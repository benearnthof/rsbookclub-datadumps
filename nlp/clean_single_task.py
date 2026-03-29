#!/usr/bin/env python3
"""
# NOTE: this is not strictly necessary since we can just split the longest threads into 
# chunks for more consistent/less catastrophic results. 
Clean a Label Studio task by removing noise text (e.g. [u/username] & "." comment spam,)
while preserving and remapping all label character offsets.

Usage:
    python clean_task_text.py <TASK_ID>
    python clean_task_text.py <TASK_ID> --dry-run    # preview only, no writes
    python clean_task_text.py <TASK_ID> --pattern "custom regex"
"""
import re
import sys
import argparse
from label_studio_sdk import LabelStudio

API_URL = "http://localhost:8080"
API_KEY = ""

# Matches: [u/username] .   [u/username].   [u/username] (with trailing whitespace/newlines)
DEFAULT_NOISE_PATTERN = r'\[u/[^\]]+\]\s*\.?\s*'


def build_offset_map(text, removed_spans):
    """
    Build a character-level map: old index → new index (None if removed).
    """
    removed = bytearray(len(text))
    for start, end in removed_spans:
        for i in range(start, end):
            removed[i] = 1

    offset_map = []
    new_pos = 0
    for is_removed in removed:
        if is_removed:
            offset_map.append(None)
        else:
            offset_map.append(new_pos)
            new_pos += 1
    offset_map.append(new_pos)  # sentinel
    return offset_map


def remap_result(result, offset_map):
    """
    Remap start/end offsets. Returns None if the span overlaps a removed region.
    """
    old_start = result["value"].get("start")
    old_end = result["value"].get("end")

    if old_start is None or old_end is None:
        return result  # non-span label, keep as-is

    for i in range(old_start, old_end):
        if i < len(offset_map) and offset_map[i] is None:
            return None  # overlaps removed noise

    new_start = offset_map[old_start]
    new_end = offset_map[old_end]

    if new_start is None or new_end is None:
        return None

    updated = dict(result)
    updated["value"] = {**result["value"], "start": new_start, "end": new_end}
    return updated


def clean_text_and_remap(text, pattern):
    removed_spans = [(m.start(), m.end()) for m in re.finditer(pattern, text)]
    offset_map = build_offset_map(text, removed_spans)
    clean_text = re.sub(pattern, "", text)
    return clean_text, offset_map, removed_spans


def process_results(results, offset_map):
    kept, dropped = [], 0
    for r in results:
        remapped = remap_result(r, offset_map)
        if remapped is None:
            dropped += 1
        else:
            kept.append(remapped)
    return kept, dropped


def main():
    parser = argparse.ArgumentParser(
        description="Remove noise text from a Label Studio task while preserving label offsets."
    )
    parser.add_argument("task_id", type=int, help="Label Studio Task ID to clean")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, no writes")
    parser.add_argument("--pattern", type=str, default=DEFAULT_NOISE_PATTERN,
                        help=f"Regex pattern for noise to remove (default: {DEFAULT_NOISE_PATTERN!r})")
    args = parser.parse_args()

    client = LabelStudio(base_url=API_URL, api_key=API_KEY)

    print(f"Fetching task {args.task_id}...")
    task = client.tasks.get(id=args.task_id)
    print(f"Thread ID             : {task.data.get('thread_id')}")

    original_text = task.data.get("text", "")
    print(f"Text length (original): {len(original_text):,} chars")

    clean_text, offset_map, removed_spans = clean_text_and_remap(original_text, args.pattern)
    print(f"Text length (cleaned) : {len(clean_text):,} chars")
    print(f"Noise spans removed   : {len(removed_spans)}")

    if not removed_spans:
        print("Nothing matched the noise pattern. Exiting.")
        sys.exit(0)

    print("\nSample removed spans (first 5):")
    for start, end in removed_spans[:5]:
        print(f"  [{start}:{end}] {original_text[start:end]!r}")

    # Prefer human annotations, fall back to predictions
    source = None
    if task.annotations:
        valid = [a for a in task.annotations if not a.get("was_cancelled")]
        if valid:
            source = "annotation"
            source_obj = valid[0]
            results = source_obj.get("result", [])
            source_id = source_obj["id"]
    if source is None:
        if not task.predictions:
            print("No annotations or predictions found. Exiting.")
            sys.exit(0)
        source = "prediction"
        source_obj = task.predictions[0]
        results = source_obj.result
        source_id = source_obj.id

    print(f"\nLabel source          : {source} (id={source_id})")
    print(f"Labels (original)     : {len(results)}")

    clean_results, dropped = process_results(results, offset_map)
    print(f"Labels (after remap)  : {len(clean_results)} kept, {dropped} dropped (overlapped noise)")

    if args.dry_run:
        print("\nDry run — no changes written.")
        print("\nSample remapped labels (first 5):")
        for r in clean_results[:5]:
            v = r["value"]
            print(f"  [{v['start']}:{v['end']}] {v.get('text', '')!r} → {v.get('labels', [])}")
        sys.exit(0)

    # Update task text via SDK
    updated_task = client.tasks.update(id=task.id, data={**task.data, "text": clean_text})
    print(f"\nTask text updated (new length: {len(updated_task.data['text']):,} chars).")

    # Update labels via SDK
    if source == "annotation":
        client.annotations.update(id=source_id, result=clean_results)
    else:
        client.predictions.update(id=source_id, result=clean_results)
    print(f"Labels updated.")

    print(f"\nDone. Task {task.id} cleaned: {len(results)} → {len(clean_results)} labels, "
          f"{len(original_text):,} → {len(clean_text):,} chars.")


if __name__ == "__main__":
    main()