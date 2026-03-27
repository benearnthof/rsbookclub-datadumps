#!/usr/bin/env python3
"""
Remove faulty labels (e.g. "the", "The", ...) from a Label Studio task's predictions.

Usage:
    python clean_labels.py TASK_ID
    python clean_labels.py TASK_ID --dry-run
"""
import sys
import argparse
from collections import Counter
from label_studio_sdk import LabelStudio # type: ignore

API_URL = "http://localhost:8080"
API_KEY = ""

BAD_TERMS = set(["t", "de", "f.","the", "THE", "The", "of", "Of", "of the", "in", "a", "A"])


def main():
    parser = argparse.ArgumentParser(description="Remove faulty labels from a Label Studio task.")
    parser.add_argument("task_id", type=int, help="Label Studio Task ID to clean")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without updating")
    args = parser.parse_args()

    client = LabelStudio(base_url=API_URL, api_key=API_KEY)

    print(f"Fetching task {args.task_id}...")
    task = client.tasks.get(id=args.task_id)
    print(f"Thread ID: {task.data.get('thread_id')}")

    if not task.predictions:
        print("No predictions found.")
        sys.exit(0)

    preds = task.predictions[0]
    original_count = len(preds.result)

    # Show top offenders before filtering
    counts = Counter(x["value"]["text"] for x in preds.result)
    bad_found = {k: v for k, v in counts.items() if k in BAD_TERMS}
    if bad_found:
        print("\nBad terms found:")
        for term, count in sorted(bad_found.items(), key=lambda x: -x[1]):
            print(f"  {repr(term):15s} x{count}")

    clean_result = [r for r in preds.result if r["value"]["text"] not in BAD_TERMS]
    removed_count = original_count - len(clean_result)

    print(f"\nOriginal: {original_count} Removed: {removed_count} Remaining: {len(clean_result)}")

    if args.dry_run:
        print("no changes written.")
        sys.exit(0)

    if removed_count == 0:
        print("Nothing to remove.")
        sys.exit(0)

    print("Updating prediction...")
    updated_pred = client.predictions.update(id=preds.id, result=clean_result)
    print(f"Done. Prediction {preds.id} now has {len(updated_pred.result)} labels.")


if __name__ == "__main__":
    main()
