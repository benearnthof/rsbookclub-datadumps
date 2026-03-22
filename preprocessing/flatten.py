#!/usr/bin/env python3
"""
flatten.py:  convert comment trees into structured per-thread documents

Input:
    --submissions   rsbc_submissions_pruned.jsonl
    --comments      rsbc_comments_pruned.jsonl
    --output        threads.jsonl            (one JSON object per line)
    --min-comments  skip threads with fewer than N comments (default: 0)
    --exclude-removed  skip threads where submission was deleted/removed

Output schema (one object per thread):
{
    "thread_id":  "nmda50",
    "metadata": {
        "title":          str,
        "author":         str,
        "date":           "YYYY-MM-DD",
        "created_utc":    int,
        "edited":         false | int,
        "score":          int,
        "upvote_ratio":   float,
        "flair":          str | null,
        "url":            str,
        "permalink":      str,
        "domain":         str,
        "is_self":        bool,
        "distinguished":  str | null,
        "stickied":       bool,
        "removed_by_category": str | null,
        "is_removed":     bool,
        "num_comments_reported": int,   # from submission field
        "num_comments_found":    int,   # actually in our dataset
        "depth_max":      int,
        "date_last_comment": "YYYY-MM-DD" | null
    },
    "segments": [
        {
            "id":       "nmda50_title" | "nmda50_body" | "<comment_id>",
            "type":     "title" | "body" | "comment",
            "author":   str,
            "depth":    int,            # 0 = post, 1+ = comment nesting level
            "score":    int | null,
            "created_utc": int | null,
            "edited":   false | int | null,
            "is_removed": bool,
            "is_submitter": bool,       # OP commenting (comments only)
            "distinguished": str | null,
            "stickied": bool,
            "parent_id": str | null,    # null for title/body segments
            "text":     str
        },
        ...
    ]
}

Notes:
- Segments are ordered depth-first (DFS pre-order) so conversational context
  is always adjacent for LLM / NER consumption.
- Orphaned comments (parent not in dataset) are promoted to depth 1.
- The "text" field is deliberately NOT included at the top level — use the
  provided adapter functions (or flatten_to_text()) to render as needed.
- Compatible with Label Studio's text classification / NER import format
  via the included to_labelstudio() adapter.

Usage:
    python3 flatten.py \\
        --submissions rsbc_submissions_pruned.jsonl \\
        --comments    rsbc_comments_pruned.jsonl \\
        --output      threads.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

log = logging.getLogger("flatten")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))

def utc_to_date(ts) -> str | None:
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError):
        return None


def clean_text(text) -> str:
    """Normalise whitespace; return empty string for deleted/removed markers."""
    if not text or text in ("[deleted]", "[removed]"):
        return ""
    return " ".join(str(text).split())


def load_submissions(path: Path) -> dict[str, dict]:
    subs = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                subs[obj["id"]] = obj
            except (json.JSONDecodeError, KeyError):
                pass
    log.info(f"Loaded {len(subs):,} submissions")
    return subs


def load_comments(path: Path) -> dict[str, dict]:
    """Returns {comment_id: comment_obj}."""
    comments = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                comments[obj["id"]] = obj
            except (json.JSONDecodeError, KeyError):
                pass
    log.info(f"Loaded {len(comments):,} comments")
    return comments


# link threads by their IDs
def build_tree(
    submission_id: str,
    comments: dict[str, dict],
) -> tuple[list[dict], int]:
    """
    Build a DFS-ordered flat list of comment dicts, each annotated with
    `_depth` (1-based relative to submission).

    Orphaned comments (parent deleted/not in dataset) are promoted to depth 1.

    Returns (ordered_comments, max_depth).
    """
    # Index comments that belong to this thread
    thread_comments = {
        cid: c for cid, c in comments.items()
        if c.get("link_id", "") == f"t3_{submission_id}"
    }

    if not thread_comments:
        return [], 0

    # Build parent: children map
    children: dict[str, list[str]] = defaultdict(list)
    for cid, c in thread_comments.items():
        pid = str(c.get("parent_id") or "")
        # Normalise parent id to bare id (strip t1_/t3_ prefix)
        bare_pid = pid.split("_", 1)[1] if "_" in pid else pid
        children[bare_pid].append(cid)

    # Sort children by created_utc for deterministic ordering
    for pid in children:
        children[pid].sort(key=lambda cid: thread_comments.get(cid, {}).get("created_utc", 0))

    # DFS traversal starting from submission root
    result: list[dict] = []
    max_depth = 0

    def dfs(node_id: str, depth: int) -> None:
        nonlocal max_depth
        if node_id in thread_comments:
            c = thread_comments[node_id]
            c["_depth"] = depth
            result.append(c)
            max_depth = max(max_depth, depth)
        for child_id in children.get(node_id, []):
            dfs(child_id, depth + 1)

    # Start from submission root (t3_ children) and orphans
    root_children = children.get(submission_id, [])
    visited = set()

    for cid in root_children:
        dfs(cid, 1)
        visited.add(cid)

    # Promote orphans (parent not in thread_comments and not the submission)
    for cid in thread_comments:
        if cid not in visited:
            bare_pid = str(thread_comments[cid].get("parent_id") or "").split("_", 1)[-1]
            if bare_pid not in thread_comments and bare_pid != submission_id:
                dfs(cid, 1)   # promote to top level
                visited.add(cid)

    return result, max_depth


def build_segments(sub: dict, ordered_comments: list[dict]) -> list[dict]:
    sid = sub["id"]
    segments = []

    # title
    segments.append({
        "id":           f"{sid}_title",
        "type":         "title",
        "author":       sub.get("author", ""),
        "depth":        0,
        "score":        sub.get("score"),
        "created_utc":  sub.get("created_utc"),
        "edited":       sub.get("edited", False),
        "is_removed":   bool(sub.get("_removed", False)),
        "is_submitter": True,
        "distinguished": sub.get("distinguished"),
        "stickied":     bool(sub.get("stickied", False)),
        "parent_id":    None,
        "text":         clean_text(sub.get("title", "")),
    })

    # body (textposts)
    if sub.get("is_self", False):
        body_text = clean_text(sub.get("selftext", ""))
        segments.append({
            "id":           f"{sid}_body",
            "type":         "body",
            "author":       sub.get("author", ""),
            "depth":        0,
            "score":        sub.get("score"),
            "created_utc":  sub.get("created_utc"),
            "edited":       sub.get("edited", False),
            "is_removed":   bool(sub.get("_removed", False)),
            "is_submitter": True,
            "distinguished": sub.get("distinguished"),
            "stickied":     bool(sub.get("stickied", False)),
            "parent_id":    None,
            "text":         body_text,
        })

    # comments
    for c in ordered_comments:
        segments.append({
            "id":           c["id"],
            "type":         "comment",
            "author":       c.get("author", ""),
            "depth":        c.get("_depth", 1),
            "score":        c.get("score"),
            "created_utc":  c.get("created_utc"),
            "edited":       c.get("edited", False),
            "is_removed":   bool(c.get("_removed", False)),
            "is_submitter": bool(c.get("is_submitter", False)),
            "distinguished": c.get("distinguished"),
            "stickied":     bool(c.get("stickied", False)),
            "parent_id":    c.get("parent_id"),
            "text":         clean_text(c.get("body", "")),
        })

    return segments


def build_thread(
    sub: dict,
    comments: dict[str, dict],
) -> dict:
    sid = sub["id"]
    ordered_comments, depth_max = build_tree(sid, comments)
    segments = build_segments(sub, ordered_comments)

    # Date of last comment
    comment_dates = [
        c.get("created_utc") for c in ordered_comments if c.get("created_utc")
    ]
    last_comment_utc = max(comment_dates) if comment_dates else None

    return {
        "thread_id": sid,
        "metadata": {
            "title":                 sub.get("title", ""),
            "author":                sub.get("author", ""),
            "date":                  utc_to_date(sub.get("created_utc")),
            "created_utc":           sub.get("created_utc"),
            "edited":                sub.get("edited", False),
            "score":                 sub.get("score"),
            "upvote_ratio":          sub.get("upvote_ratio"),
            "flair":                 sub.get("link_flair_text"),
            "url":                   sub.get("url", ""),
            "permalink":             sub.get("permalink", ""),
            "domain":                sub.get("domain", ""),
            "is_self":               bool(sub.get("is_self", False)),
            "distinguished":         sub.get("distinguished"),
            "stickied":              bool(sub.get("stickied", False)),
            "removed_by_category":   sub.get("removed_by_category"),
            "is_removed":            bool(sub.get("_removed", False)),
            "num_comments_reported": sub.get("num_comments", 0),
            "num_comments_found":    len(ordered_comments),
            "depth_max":             depth_max,
            "date_last_comment":     utc_to_date(last_comment_utc),
        },
        "segments": segments,
    }


# convert to readable .txt format (corpus.txt) ~60MB
def flatten_to_text(thread: dict, indent: int = 2, max_depth: int = 8) -> str:
    """
    Render a thread document as plain text with visual tree structure.
    Suitable for LLM prompts, DAPT corpus, and human review.
    """
    m    = thread["metadata"]
    segs = thread["segments"]
    lines = []

    # Header
    flair_str = f"  [flair: {m['flair']}]" if m["flair"] else ""
    lines.append(f"[THREAD] {m['title']}{flair_str}")
    lines.append(
        f"[DATE] {m['date']}  "
        f"[AUTHOR] u/{m['author']}  "
        f"[SCORE] {m['score']}"
    )
    lines.append("")

    for seg in segs:
        if not seg["text"]:
            continue
        depth  = min(seg["depth"], max_depth)
        pad    = " " * (depth * indent)

        if seg["type"] == "title":
            continue   # already in header
        elif seg["type"] == "body":
            lines.append(seg["text"])
            lines.append("")
            lines.append("─" * 60)
        else:
            # comment
            submitter_tag = " [OP]" if seg["is_submitter"] else ""
            mod_tag       = " [MOD]" if seg["distinguished"] == "moderator" else ""
            score_str     = f" +{seg['score']}" if seg.get("score") else ""
            lines.append(
                f"{pad}[u/{seg['author']}{submitter_tag}{mod_tag}{score_str}]"
            )
            # Wrap comment text at same indent level
            for text_line in seg["text"].splitlines():
                lines.append(f"{pad}{text_line}")
            lines.append("")

    # Footer
    lines.append("─" * 60)
    lines.append(
        f"[THREAD END] {thread['thread_id']} | "
        f"{m['num_comments_found']} comments | "
        f"{m['date']}: {m['date_last_comment'] or '?'}"
    )

    return "\n".join(lines)


# For manual labeling in labelstudio
def to_labelstudio(thread: dict) -> list[dict]:
    """
    Convert a thread to a list of Label Studio NER tasks —
    one task per non-empty, non-removed segment.

    Import the resulting list via Label Studio's JSON import.
    """
    tasks = []
    for seg in thread["segments"]:
        if not seg["text"] or seg["is_removed"]:
            continue
        tasks.append({
            "data": {
                "text":       seg["text"],
                "segment_id": seg["id"],
                "thread_id":  thread["thread_id"],
                "type":       seg["type"],
                "author":     seg["author"],
                "depth":      seg["depth"],
                "date":       thread["metadata"]["date"],
                "title":      thread["metadata"]["title"],
            }
        })
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Flatten Reddit threads to structured JSON")
    parser.add_argument("--submissions",      required=True,            type=Path)
    parser.add_argument("--comments",         required=True,            type=Path)
    parser.add_argument("--output",           required=True,            type=Path)
    parser.add_argument("--min-comments",     default=0,                type=int,
                        help="Skip threads with fewer than N comments")
    parser.add_argument("--exclude-removed",  action="store_true",
                        help="Skip threads where the submission itself was removed")
    parser.add_argument("--export-text",      type=Path, default=None,
                        help="Also write a plain-text corpus file (one thread per doc)")
    parser.add_argument("--export-labelstudio", type=Path, default=None,
                        help="Also write a Label Studio import JSON file")
    args = parser.parse_args()

    subs     = load_submissions(args.submissions)
    comments = load_comments(args.comments)

    # Group comments by submission id for fast lookup
    by_thread: dict[str, dict[str, dict]] = defaultdict(dict)
    for cid, c in comments.items():
        link = c.get("link_id", "")
        sid  = link.split("_", 1)[1] if "_" in link else link
        by_thread[sid][cid] = c

    args.output.parent.mkdir(parents=True, exist_ok=True)

    total = skipped_removed = skipped_comments = written = 0
    ls_tasks: list[dict] = []

    text_fh = open(args.export_text, "w", encoding="utf-8") if args.export_text else None

    with open(args.output, "w", encoding="utf-8") as out_fh:
        for sid, sub in tqdm(subs.items(), desc="threads", unit="thread"):
            total += 1

            if args.exclude_removed and sub.get("_removed"):
                skipped_removed += 1
                continue

            thread_comments = by_thread.get(sid, {})
            if len(thread_comments) < args.min_comments:
                skipped_comments += 1
                continue

            thread = build_thread(sub, thread_comments)
            out_fh.write(json.dumps(thread, ensure_ascii=False) + "\n")
            written += 1

            if text_fh:
                text_fh.write(flatten_to_text(thread))
                text_fh.write("\n\n" + "═" * 60 + "\n\n")

            if args.export_labelstudio is not None:
                ls_tasks.extend(to_labelstudio(thread))

    if text_fh:
        text_fh.close()

    if args.export_labelstudio is not None:
        with open(args.export_labelstudio, "w", encoding="utf-8") as lsfh:
            json.dump(ls_tasks, lsfh, ensure_ascii=False, indent=2)
        log.info(f"Label Studio tasks: {len(ls_tasks):,}: {args.export_labelstudio}")

    log.info(
        f"\nDone.  total={total:,}  written={written:,}  "
        f"skipped_removed={skipped_removed:,}  "
        f"skipped_min_comments={skipped_comments:,}"
    )
    if args.export_text:
        log.info(f"Text corpus: {args.export_text}")
    log.info(f"Threads file: {args.output}")


if __name__ == "__main__":
    main()
