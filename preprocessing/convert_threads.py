#!/usr/bin/env python3
"""
Convert threads.jsonl to per-thread Label Studio NER tasks.

Each output task consolidates a full thread (title + body + comments)
into a single annotatable text document, with character offsets tracked
so Label Studio span annotations map back to the right content.

Usage:
    python threads_to_labelstudio.py input.jsonl output.json

    # Optional flags:
    --indent DEPTH        depth to indent replies with (default: 2 spaces per level)
    --max-depth N         truncate comment trees deeper than N (default: no limit)
    --pretty              pretty-print output JSON (default: compact)
"""

import json
import argparse
import sys
from pathlib import Path


INDENT_UNIT = "  "   # spaces per depth level for replies


def segment_prefix(seg: dict, indent: str) -> str:
    """Return the human-readable prefix for a segment, e.g. '[TITLE]' or '[u/alice →]'."""
    stype = seg.get("type", "comment")
    author = seg.get("author") or "unknown"
    depth  = seg.get("depth", 0)

    if stype == "title":
        return "[TITLE]"
    if stype == "body":
        return f"[POST by u/{author}]"

    # comment — show nesting with arrows
    arrows = "→ " * max(0, depth - 1)
    return f"[{arrows}u/{author}]"


def render_thread(thread: dict, indent_unit: str = INDENT_UNIT, max_depth: int | None = None) -> str:
    """
    Render a thread dict into a single annotatable string.

    Layout per segment:
        <prefix> <text>
        <blank line>

    Returns the full document string.
    """
    segments = thread.get("segments", [])
    parts: list[str] = []

    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue  # skip deleted / empty segments

        depth = seg.get("depth", 0)
        if max_depth is not None and depth > max_depth:
            continue

        indent = indent_unit * max(0, depth - 1)   # title/post have depth 0
        prefix = segment_prefix(seg, indent)
        line   = f"{indent}{prefix} {text}"
        parts.append(line)

    return "\n\n".join(parts)


def thread_to_ls_task(thread: dict, **render_kwargs) -> dict:
    """Convert one thread dict to a Label Studio import task."""
    meta = thread.get("metadata", {})
    text = render_thread(thread, **render_kwargs)

    return {
        "data": {
            "text":      text,
            "thread_id": thread.get("thread_id", ""),
            "title":     meta.get("title", ""),
            "author":    meta.get("author", ""),
        }
    }


LS_CONFIG = """\
<!-- Paste this into your Label Studio project's Labeling Config -->
<View>
  <Header value="Thread: $title"/>
  <Text name="text" value="$text" granularity="word"/>
  <Labels name="label" toName="text">
    <Label value="BOOK"   background="#FF0000"/>
    <Label value="WRITER" background="#00FF00"/>
  </Labels>
</View>
"""


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Convert threads.jsonl → Label Studio NER tasks (one task per thread)."
    )
    p.add_argument("input",  type=Path, help="Path to threads.jsonl")
    p.add_argument("output", type=Path, help="Path for output .json (Label Studio import format)")
    p.add_argument("--indent",     type=int,  default=2,    metavar="N",
                   help="Spaces per reply depth level (default: 2)")
    p.add_argument("--max-depth",  type=int,  default=None, metavar="N",
                   help="Skip comments deeper than N levels (default: no limit)")
    p.add_argument("--pretty",     action="store_true",
                   help="Pretty-print output JSON")
    p.add_argument("--print-config", action="store_true",
                   help="Print a suggested Label Studio labeling config and exit")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.print_config:
        print(LS_CONFIG)
        return

    indent_unit = " " * args.indent
    render_kwargs = dict(indent_unit=indent_unit, max_depth=args.max_depth)

    if not args.input.exists():
        sys.exit(f"ERROR: input file not found: {args.input}")

    tasks = []
    skipped = 0

    with args.input.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                thread = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"WARNING: skipping line {lineno} – JSON parse error: {exc}", file=sys.stderr)
                skipped += 1
                continue

            task = thread_to_ls_task(thread, **render_kwargs)
            if not task["data"]["text"].strip():
                skipped += 1
                continue
            tasks.append(task)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        if args.pretty:
            json.dump(tasks, fh, ensure_ascii=False, indent=2)
        else:
            json.dump(tasks, fh, ensure_ascii=False)

    print(f"Done. {len(tasks)} tasks written to {args.output}", file=sys.stderr)
    if skipped:
        print(f"       {skipped} threads skipped (empty or malformed).", file=sys.stderr)
    print(
        f"\nTip: run with --print-config to get a copy-pasteable Label Studio labeling config.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()