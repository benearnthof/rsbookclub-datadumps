#!/usr/bin/env python3
"""
Tag Label Studio tasks whose annotations/predictions contain suspicious
short-string label candidates, so they can be reviewed before acceptance.

For each task that contains one or more candidate strings, the task's `meta`
field is updated with:
    { "needs_review": ["i", "m"] }

If the task already carries a `needs_review` key, the found candidates are
*merged* into the existing list (deduplication, sorted), so repeated runs
with different candidate sets are safe.

Usage examples
--------------
# Dry-run: see what would be tagged, touch nothing
python tag_review_candidates.py --candidates "i,m,o,A,C,G" --dry-run

# Actually tag tasks (exact-case matching)
python tag_review_candidates.py --candidates "i,m,o,A,C,G"

# Case-insensitive: treats "V" and "v" as the same candidate
python tag_review_candidates.py --candidates "v,i,m,o" --ignore-case

# Clear the needs_review meta key from ALL tasks (undo)
python tag_review_candidates.py --clear
"""
import argparse
from collections import defaultdict
from label_studio_sdk import LabelStudio  # type: ignore

API_URL    = "http://localhost:8080"
API_KEY    = ""
PROJECT_ID = "8"
PAGE_SIZE  = 1000


def fetch_all_tasks(client: LabelStudio):
    page, all_tasks = 1, []
    while True:
        batch = client.tasks.list(project=PROJECT_ID, page=page, page_size=PAGE_SIZE)
        if not batch.items:
            break
        all_tasks.extend(batch.items)
        print(f"  Fetched page {page} ({len(all_tasks)} tasks so far)…", end="\r")
        if len(batch.items) < PAGE_SIZE:
            break
        page += 1
    print(f"\n  Total tasks: {len(all_tasks)}")
    return all_tasks


def get_results(task):
    """Prefer human annotations; fall back to model predictions."""
    if task.annotations:
        valid = [a for a in task.annotations if not a.get("was_cancelled")]
        if valid:
            return valid[0].get("result", [])
    if task.predictions:
        return task.predictions[0].result
    return []


def extract_label_texts(results) -> list[str]:
    """Return every non-empty text string from a result list."""
    texts = []
    for r in results:
        text = r.get("value", {}).get("text", "").strip()
        if text:
            texts.append(text)
    return texts


def find_matches(label_texts: list[str], candidates: set[str], ignore_case: bool) -> list[str]:
    """
    Return the sorted list of candidates that appear at least once among
    the label texts extracted from a single task.
    """
    if ignore_case:
        normalised = {t.lower() for t in label_texts}
        return sorted({c for c in candidates if c.lower() in normalised})
    else:
        label_set = set(label_texts)
        return sorted(c for c in candidates if c in label_set)


def merged_review_list(existing_meta: dict, new_candidates: list[str]) -> list[str]:
    """Merge new candidates into whatever is already in needs_review."""
    existing = existing_meta.get("needs_review", [])
    if isinstance(existing, str):          # guard against accidental string value
        existing = [existing]
    merged = sorted(set(existing) | set(new_candidates))
    return merged


def run_tagging(client, candidates: set[str], ignore_case: bool, dry_run: bool):
    print(f"Fetching tasks from project {PROJECT_ID}…")
    tasks = fetch_all_tasks(client)

    tagged: dict[int, list[str]] = {}   # task_id → candidates found
    skipped_no_labels = 0
    skipped_has_meta = 0

    print("Scanning label texts for candidates…")
    for task in tasks:
        # Tasks that already carry any metadata have been previously reviewed —
        # don't overwrite or re-tag them.
        if isinstance(task.meta, dict) and task.meta:
            skipped_has_meta += 1
            continue
        results = get_results(task)
        if not results:
            skipped_no_labels += 1
            continue
        label_texts = extract_label_texts(results)
        matches = find_matches(label_texts, candidates, ignore_case)
        if matches:
            tagged[task.id] = matches

    print(f"\n  Tasks skipped (already have meta): {skipped_has_meta}")
    print(f"  Tasks with no labels (skipped)  : {skipped_no_labels}")
    print(f"  Tasks matching ≥1 candidate     : {len(tagged)}")

    if not tagged:
        print("  Nothing to tag.")
        return

    # per-candidate breakdown
    per_candidate: dict[str, int] = defaultdict(int)
    for matches in tagged.values():
        for c in matches:
            per_candidate[c] += 1
    print("\n  Candidate breakdown:")
    for c in sorted(per_candidate, key=lambda x: per_candidate[x], reverse=True):
        print(f"    {repr(c):>10}  →  {per_candidate[c]} task(s)")

    if dry_run:
        print("\n  [DRY RUN] No changes written. Re-run without --dry-run to apply.")
        return

    print(f"\nTagging {len(tagged)} tasks…")
    errors = 0
    for i, (task_id, matches) in enumerate(tagged.items(), 1):
        # fetch current meta so we can merge safely
        try:
            task_obj = client.tasks.get(task_id)
            # The SDK can return True/False instead of a dict for tasks whose
            # meta field is a bare boolean — guard against that explicitly.
            raw_meta = task_obj.meta
            current_meta = raw_meta if isinstance(raw_meta, dict) else {}
            new_review_list = merged_review_list(current_meta, matches)
            updated_meta = {**current_meta, "needs_review": new_review_list}
            client.tasks.update(task_id, meta=updated_meta)
            print(f"  [{i}/{len(tagged)}] Task {task_id}: needs_review={new_review_list}", end="\r")
        except Exception as exc:
            print(f"\n  [ERROR] Task {task_id}: {exc}")
            errors += 1

    print(f"\n\nDone.  {len(tagged) - errors} tasks tagged,  {errors} errors.")


def run_clear(client, dry_run: bool):
    """Remove the needs_review key from every task that has it."""
    print(f"Fetching tasks from project {PROJECT_ID}…")
    tasks = fetch_all_tasks(client)

    to_clear = [t for t in tasks if isinstance(t.meta, dict) and t.meta.get("needs_review")]
    print(f"  Tasks with needs_review set: {len(to_clear)}")

    if not to_clear:
        print("  Nothing to clear.")
        return

    if dry_run:
        print("  [DRY RUN] No changes written.")
        return

    errors = 0
    for i, task in enumerate(to_clear, 1):
        try:
            raw_meta = task.meta
            existing = raw_meta if isinstance(raw_meta, dict) else {}
            cleaned = {k: v for k, v in existing.items() if k != "needs_review"}
            client.tasks.update(task.id, meta=cleaned)
            print(f"  [{i}/{len(to_clear)}] Cleared task {task.id}", end="\r")
        except Exception as exc:
            print(f"\n  [ERROR] Task {task.id}: {exc}")
            errors += 1

    print(f"\n\nDone.  {len(to_clear) - errors} tasks cleared,  {errors} errors.")


def main():
    parser = argparse.ArgumentParser(
        description="Tag Label Studio tasks that contain suspicious short-string label candidates.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--candidates", "-c",
        type=str,
        default="",
        help='Comma-separated list of label strings to flag, e.g. "i,m,o,A,C"',
    )
    parser.add_argument(
        "--ignore-case",
        action="store_true",
        help="Treat candidates case-insensitively (e.g. 'v' also matches 'V')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview affected tasks without writing anything to Label Studio",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Remove the needs_review meta key from all tasks instead of tagging",
    )
    args = parser.parse_args()

    client = LabelStudio(base_url=API_URL, api_key=API_KEY)

    if args.clear:
        run_clear(client, dry_run=args.dry_run)
        return

    if not args.candidates:
        parser.error("--candidates is required unless --clear is used.")

    candidates = {c.strip() for c in args.candidates.split(",") if c.strip()}
    print(f"Candidates to flag : {sorted(candidates)}")
    print(f"Case-insensitive   : {args.ignore_case}")
    print(f"Dry run            : {args.dry_run}\n")

    run_tagging(client, candidates, ignore_case=args.ignore_case, dry_run=args.dry_run)


if __name__ == "__main__":
    main()