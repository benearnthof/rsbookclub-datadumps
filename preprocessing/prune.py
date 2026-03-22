#!/usr/bin/env python3
"""
prune.py: strip irrelevant fields from rsbookclub JSONL files
After merging we're left with two large .jsonl files that contain a ton of irrelevant fields.
We keep only the fields needed for:
  - Thread tree reconstruction  (id, link_id, parent_id)
  - NER/text analysis           (title, selftext, body)
  - Timestamps                  (created_utc)
  - Basic metadata              (author, score, distinguished, stickied)
  - Deleted/removed detection   (removed_by_category, selftext/body value)

Usage:
    python3 prune.py submissions  <input.jsonl> <output.jsonl>
    python3 prune.py comments     <input.jsonl> <output.jsonl>
    python3 prune.py auto         <input.jsonl> <output.jsonl>  # detects type
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SUBMISSION_FIELDS = {
    # thread ids
    "id",               # base36 post id — foreign key for all comments
    "name",             # full name t3_<id>
    "permalink",        # for reference / URL reconstruction
    "url",              # external link if not a self-post
    # content
    "title",            # always present, primary NER surface
    "selftext",         # body text; "[deleted]" / "[removed]" if gone
    "is_self",          # True = text post, False = link post
    "domain",           # source domain for link posts
    # authorship & moderation
    "author",           # username; "[deleted]" if gone
    "distinguished",    # "moderator" | null
    "stickied",         # pinned posts
    "removed_by_category",  # "moderator" | "deleted" | null
    # engagement (useful for weighting NER targets)
    "score",
    "upvote_ratio",
    "num_comments",
    # timestamps
    "created_utc",
    # flair — keep text only, drop css/color/template noise
    "link_flair_text",
}

COMMENT_FIELDS = {
    # thread ids
    "id",               # base36 comment id
    "name",             # full name t1_<id>
    "link_id",          # t3_<submission_id> — which post this belongs to
    "parent_id",        # t1_<comment_id> or t3_<post_id>: tree position
    "permalink",
    # content
    "body",             # comment text; "[deleted]" / "[removed]" if gone
    # authorship & moderation
    "author",
    "is_submitter",     # True if OP is commenting: useful context for NER
    "distinguished",    # "moderator" | null
    "stickied",         # automod / mod sticky comments
    # engagement
    "score",
    "controversiality", # 0 or 1
    # timestamps
    "created_utc",
    "edited",
}


def is_removed(obj: dict, kind: str) -> bool:
    """Return True if the post/comment content has been wiped."""
    text_field = "selftext" if kind == "submission" else "body"
    text = obj.get(text_field, "")
    return text in ("[deleted]", "[removed]")


def prune(obj: dict, fields: set[str]) -> dict:
    return {k: v for k, v in obj.items() if k in fields}


def detect_kind(obj: dict) -> str:
    """Detect whether a record is a submission or comment from its fields."""
    if "title" in obj:
        return "submission"
    if "body" in obj:
        return "comment"
    raise ValueError("Cannot detect record type — neither 'title' nor 'body' present")


def process(kind_arg: str, input_path: Path, output_path: Path) -> None:
    total = kept = removed = bad = 0

    with open(input_path, encoding="utf-8") as in_fh, \
         open(output_path, "w", encoding="utf-8") as out_fh:

        for line in in_fh:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
                kind = kind_arg if kind_arg != "auto" else detect_kind(obj)
                fields = SUBMISSION_FIELDS if kind == "submission" else COMMENT_FIELDS
                pruned = prune(obj, fields)
                # Tag deleted/removed so downstream can filter without re-checking
                pruned["_removed"] = is_removed(obj, kind)
                out_fh.write(json.dumps(pruned, ensure_ascii=False) + "\n")
                kept += 1
                if pruned["_removed"]:
                    removed += 1

            except (json.JSONDecodeError, ValueError) as exc:
                bad += 1
                print(f"  WARN line {total}: {exc}", file=sys.stderr)

    print(
        f"{input_path.name}  →  {output_path.name}\n"
        f"  total={total:,}  kept={kept:,}  "
        f"deleted/removed={removed:,} ({removed/kept*100:.1f}%)  "
        f"bad={bad:,}"
    )


def main() -> None:
    if len(sys.argv) != 4 or sys.argv[1] not in ("submissions", "comments", "auto"):
        print(__doc__)
        sys.exit(1)
    kind       = "submission" if sys.argv[1] == "submissions" else \
                 "comment"    if sys.argv[1] == "comments"    else "auto"
    input_path  = Path(sys.argv[2])
    output_path = Path(sys.argv[3])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    process(kind, input_path, output_path)

if __name__ == "__main__":
    main()
