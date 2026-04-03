#!/usr/bin/env python3
"""
Query Label Studio API to compute label summary statistics across all tasks.
Prefers human annotations over model predictions where available.

Usage:
    python label_stats.py                        # top 50 overall
    python label_stats.py --top 100              # top 100
    python label_stats.py --by-label             # breakdown per label type
    python label_stats.py --min-count 5          # only terms appearing >= 5 times
    python label_stats.py --csv stats.csv        # dump full stats to CSV
    python label_stats.py --unreviewed           # only tasks with no human annotations
    python label_stats.py --unique-threads       # count distinct threads per entity, not total occurrences
"""
import argparse
import csv
from collections import Counter, defaultdict
from label_studio_sdk import LabelStudio  # type: ignore

API_URL    = "http://localhost:8080"
API_KEY    = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6ODA4MTgwMzIxMCwiaWF0IjoxNzc0NjAzMjEwLCJqdGkiOiJjYzMxYmJiMmQ2NmE0NjA0YjljMDMxOTFkODBjODlmZCIsInVzZXJfaWQiOiIxIn0.MhEGA4uVO1zA3HSbkiEoZISVkBiOL4FfNu8GKElik2A"
PROJECT_ID = "8"
PAGE_SIZE  = 1000

THREAD_ID_FIELD = "thread_id"


def fetch_all_tasks(client):
    page = 1
    all_tasks = []
    while True:
        tasks = client.tasks.list(project=PROJECT_ID, page=page, page_size=PAGE_SIZE)
        if not tasks.items:
            break
        all_tasks.extend(tasks.items)
        print(f"  Fetched page {page} ({len(all_tasks)} tasks so far)...", end="\r")
        if len(tasks.items) < PAGE_SIZE:
            break
        page += 1
    print(f"\n  Done. Total tasks fetched: {len(all_tasks)}")
    return all_tasks


def is_reviewed(task) -> bool:
    """Return True if the task has at least one non-cancelled human annotation."""
    if not task.annotations:
        return False
    return any(not a.get("was_cancelled") for a in task.annotations)


def get_results(task):
    """Prefer annotations if present, fall back to model predictions."""
    if task.annotations:
        valid = [a for a in task.annotations if not a.get("was_cancelled")]
        if valid:
            return valid[0].get("result", []), "annotation"
    if task.predictions:
        return task.predictions[0].result, "prediction"
    return [], None


def collect_stats(tasks, unreviewed_only: bool, unique_threads: bool):
    """
    Returns
    -------
    overall : Counter  {text: total_count}
    by_label : dict    {label: Counter{text: count}}
    records  : list    one dict per (text, label, thread_id) combination:
                         text, label, thread_id, doc_length, count
    skipped, skipped_reviewed, source_counts
    """
    overall      = Counter()
    by_label     = defaultdict(Counter)
    raw_counts: Counter = Counter()   # (text, label, thread_id) -> count
    doc_lengths: dict   = {}          # thread_id -> full document character length

    skipped          = 0
    skipped_reviewed = 0
    source_counts    = Counter()

    for task in tasks:
        if unreviewed_only and is_reviewed(task):
            skipped_reviewed += 1
            continue

        results, source = get_results(task)
        if source is None:
            skipped += 1
            continue
        source_counts[source] += 1

        data       = task.data or {}
        thread_id  = data.get(THREAD_ID_FIELD, task.id)
        doc_length = len(data.get("text", ""))
        doc_lengths[thread_id] = doc_length

        seen_in_task: set[tuple] = set()   # (text, label) pairs for unique-thread mode

        for result in results:
            value  = result["value"]
            text   = value.get("text", "").strip()
            labels = value.get("labels", [])
            if not text:
                continue

            for label in labels:
                key = (text, label)
                if unique_threads and key in seen_in_task:
                    continue
                seen_in_task.add(key)

                overall[text] += 1
                by_label[label][text] += 1
                raw_counts[(text, label, thread_id)] += 1

    # Build the flat record list from raw_counts
    records = [
        {
            "text":       text,
            "label":      label,
            "thread_id":  thread_id,
            "doc_length": doc_lengths.get(thread_id, 0),
            "count":      count,
        }
        for (text, label, thread_id), count in raw_counts.items()
    ]
    # Sort by count descending for a predictable default order
    records.sort(key=lambda r: r["count"], reverse=True)

    return overall, by_label, records, skipped, skipped_reviewed, source_counts


def print_table(counter, title, top_n, min_count):
    rows = [(text, count) for text, count in counter.most_common(top_n) if count >= min_count]
    if not rows:
        print("  (no results)\n")
        return
    max_text_len = max(len(t) for t, _ in rows)
    col_width = max(max_text_len, 10)
    print(f"\n{'─' * (col_width + 20)}")
    print(f"  {title}")
    print(f"{'─' * (col_width + 20)}")
    print(f"  {'Text':<{col_width}}  {'Count':>8}  {'Bar'}")
    print(f"  {'─'*col_width}  {'─'*8}  {'─'*30}")
    max_count = rows[0][1]
    for text, count in rows:
        bar_len = int((count / max_count) * 30)
        bar = "█" * bar_len
        print(f"  {repr(text):<{col_width}}  {count:>8}  {bar}")
    print()


def write_csv(records, filepath, min_count):
    """
    Write one row per (text, label, thread_id) combination.
    """
    filtered   = [r for r in records if r["count"] >= min_count]
    fieldnames = ["text", "label", "thread_id", "doc_length", "count"]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(filtered)
    print(f"  CSV written to: {filepath} ({len(filtered)} rows)")


def main():
    parser = argparse.ArgumentParser(description="Label Studio label frequency statistics.")
    parser.add_argument("--top",            type=int, default=50,   help="Number of top terms to show (default: 50)")
    parser.add_argument("--by-label",       action="store_true",    help="Show breakdown per label type")
    parser.add_argument("--min-count",      type=int, default=1,    help="Minimum count to include a term")
    parser.add_argument("--csv",            type=str, default=None, metavar="FILE",
                        help="Dump per-(text, label, thread_id) stats to CSV (e.g. stats.csv)")
    parser.add_argument("--unreviewed",     action="store_true",
                        help="Only include tasks that have not yet received a human annotation")
    parser.add_argument("--unique-threads", action="store_true",
                        help="Count each (text, label) pair at most once per task/thread")
    args = parser.parse_args()

    client = LabelStudio(base_url=API_URL, api_key=API_KEY)

    print(f"Fetching all tasks from project {PROJECT_ID}...")
    tasks = fetch_all_tasks(client)

    scope      = "unreviewed tasks only" if args.unreviewed else "all tasks"
    count_mode = "unique threads" if args.unique_threads else "total occurrences"
    print(f"Computing statistics ({scope}, {count_mode})...")

    overall, by_label, records, skipped, skipped_reviewed, source_counts = collect_stats(
        tasks, args.unreviewed, args.unique_threads
    )

    if args.unreviewed:
        print(f"  Skipped (already reviewed) : {skipped_reviewed}")
    print(f"  From annotations : {source_counts['annotation']}")
    print(f"  From predictions : {source_counts['prediction']}")
    print(f"  Skipped (no labels)    : {skipped}")
    print(f"  Unique entity texts    : {len(overall)}")
    print(f"  Total label instances  : {sum(overall.values())}")

    if args.csv:
        write_csv(records, args.csv, args.min_count)

    print_table(
        overall,
        f"Top {args.top} entity texts by {count_mode} ({scope})",
        args.top,
        args.min_count,
    )

    if args.by_label:
        for label_type, counter in sorted(by_label.items()):
            print_table(
                counter,
                f"Top {args.top} — label type: {label_type} ({count_mode})",
                args.top,
                args.min_count,
            )


if __name__ == "__main__":
    main()