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
"""
import argparse
import csv
from collections import Counter, defaultdict
from label_studio_sdk import LabelStudio

API_URL = "http://localhost:8080"
API_KEY = ""
PROJECT_ID = "8"
PAGE_SIZE = 100


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


def get_results(task):
    """Prefer human annotations if present, fall back to model predictions."""
    if task.annotations:
        valid = [a for a in task.annotations if not a.get("was_cancelled")]
        if valid:
            return valid[0].get("result", []), "annotation"
    if task.predictions:
        return task.predictions[0].result, "prediction"
    return [], None


def collect_stats(tasks):
    overall = Counter()
    by_label = defaultdict(Counter)
    skipped = 0
    source_counts = Counter()

    for task in tasks:
        results, source = get_results(task)
        if source is None:
            skipped += 1
            continue
        source_counts[source] += 1
        for result in results:
            text = result["value"].get("text", "").strip()
            labels = result["value"].get("labels", [])
            if not text:
                continue
            overall[text] += 1
            for label in labels:
                by_label[label][text] += 1

    return overall, by_label, skipped, source_counts


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


def write_csv(overall, filepath, min_count):
    rows = [(text, count) for text, count in overall.most_common() if count >= min_count]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["string", "count"])
        writer.writerows(rows)
    print(f"  CSV written to: {filepath} ({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser(description="Label Studio label frequency statistics.")
    parser.add_argument("--top", type=int, default=50, help="Number of top terms to show (default: 50)")
    parser.add_argument("--by-label", action="store_true", help="Show breakdown per label type")
    parser.add_argument("--min-count", type=int, default=1, help="Minimum count to include a term")
    parser.add_argument("--csv", type=str, default=None, metavar="FILE", help="Dump full frequency stats to CSV (e.g. stats.csv)")
    args = parser.parse_args()

    client = LabelStudio(base_url=API_URL, api_key=API_KEY)

    print(f"Fetching all tasks from project {PROJECT_ID}...")
    tasks = fetch_all_tasks(client)

    print("Computing statistics...")
    overall, by_label, skipped, source_counts = collect_stats(tasks)

    print(f"  From human annotations : {source_counts['annotation']}")
    print(f"  From model predictions : {source_counts['prediction']}")
    print(f"  Skipped (no labels)    : {skipped}")
    print(f"  Unique entity texts    : {len(overall)}")
    print(f"  Total label instances  : {sum(overall.values())}")

    if args.csv:
        write_csv(overall, args.csv, args.min_count)

    print_table(overall, f"Top {args.top} most frequent entity texts (overall)", args.top, args.min_count)

    if args.by_label:
        for label_type, counter in sorted(by_label.items()):
            print_table(counter, f"Top {args.top} — label type: {label_type}", args.top, args.min_count)


if __name__ == "__main__":
    main()