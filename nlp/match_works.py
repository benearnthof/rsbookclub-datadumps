#!/usr/bin/env python3
"""
Match OpenLibrary works against Label Studio tasks.
This can be used for prelabeling, but we only use this to get a rough estimate
for the number of spans that are of length 20 or longer & were not labeled.
Anything shorter requires context to disambiguate common phrases from book titles.
This should just serve as a rough quality check for false negatives. 

Part 1: ingest to construct small database from OLDD dumps:
    python match_works.py ingest --works ol_dump_works_2026-02-28.txt --db ol_works.db

Part 2: search through the database by ahocorasick string matching:
    # TODO: Do this batch-wise so we don't have to construct the state machine anew for every single thread.
    python match_works.py search --db ol_works.db --task 11199
    python match_works.py search --db ol_works.db --task 11199 --push
    python match_works.py search --db ol_works.db --all-tasks
    python match_works.py search --db ol_works.db --all-tasks --push
"""

import re
import sys
import json
import uuid
import sqlite3
import argparse
from label_studio_sdk import LabelStudio

try:
    import ahocorasick
except ImportError:
    print("Missing dependency: pip install pyahocorasick")
    sys.exit(1)

API_URL    = "http://localhost:8080"
API_KEY    = ""
PROJECT_ID = "8"
LABEL_TYPE = "BOOK"
BATCH_SIZE = 1_000_000  # The more RAM the larger this number the faster this script runs

# L20: min len 20 -> yields 390 candidates for longest task (11199), not all correct but usable.

def ingest(works_path, db_path, min_len):
    print(f"Ingesting {works_path} → {db_path}  (min_len={min_len})")
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("""
        CREATE TABLE IF NOT EXISTS works (
            title       TEXT NOT NULL,
            ol_key      TEXT NOT NULL,
            title_lower TEXT NOT NULL
        )
    """)
    con.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_title_lower ON works(title_lower)")

    inserted = skipped = errors = 0
    batch = []

    with open(works_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % 200_000 == 0:
                print(f"  Line {i:,}  inserted={inserted:,}  skipped={skipped:,}", end="\r")
            parts = line.split("\t", 4)
            if len(parts) < 5:
                continue
            try:
                doc = json.loads(parts[4])
            except json.JSONDecodeError:
                errors += 1
                continue
            title = doc.get("title", "").strip()
            key   = doc.get("key", "")
            if len(title) < min_len:
                skipped += 1
                continue
            # Drop single-word titles
            if len(title.split()) < 2:
                skipped += 1
                continue
            batch.append((title, key, title.lower()))
            if len(batch) >= 10_000:
                con.executemany(
                    "INSERT OR IGNORE INTO works (title, ol_key, title_lower) VALUES (?,?,?)",
                    batch
                )
                con.commit()
                inserted += len(batch)
                batch.clear()

    if batch:
        con.executemany(
            "INSERT OR IGNORE INTO works (title, ol_key, title_lower) VALUES (?,?,?)",
            batch
        )
        con.commit()
        inserted += len(batch)

    row_count = con.execute("SELECT COUNT(*) FROM works").fetchone()[0]
    con.close()
    print(f"\nDone. Lines processed: {i+1:,} | Rows in DB: {row_count:,} | "
          f"Skipped (too short): {skipped:,} | Parse errors: {errors:,}")
    print(f"DB written to: {db_path}")


# part 2
def iter_title_batches(db_path, batch_size):
    con = sqlite3.connect(db_path)
    cur = con.execute("SELECT title, title_lower, ol_key FROM works ORDER BY rowid")
    while True:
        rows = cur.fetchmany(batch_size)
        if not rows:
            break
        yield rows
    con.close()


def build_automaton(rows):
    A = ahocorasick.Automaton()
    for title, title_lower, ol_key in rows:
        if title_lower not in A:
            A.add_word(title_lower, (title, ol_key))
    A.make_automaton()
    return A


def search_automaton(text, automaton, word_boundary=True):
    haystack = text.lower()
    hits = []
    for end_idx, (original_title, ol_key) in automaton.iter(haystack):
        start_idx = end_idx - len(original_title) + 1
        if word_boundary:
            before = haystack[start_idx - 1] if start_idx > 0 else " "
            after  = haystack[end_idx + 1]   if end_idx + 1 < len(haystack) else " "
            if re.match(r'\w', before) or re.match(r'\w', after):
                continue
        hits.append((start_idx, end_idx + 1, text[start_idx:end_idx + 1], original_title, ol_key))
    return hits


def keep_longest(hits):
    hits = sorted(hits, key=lambda x: (x[0], -(x[1] - x[0])))
    kept, last_end = [], -1
    for h in hits:
        if h[0] >= last_end:
            kept.append(h)
            last_end = h[1]
    return kept


def get_existing(task):
    """Returns (span_list, source_type, source_obj)."""
    if task.annotations:
        valid = [a for a in task.annotations if not a.get("was_cancelled")]
        if valid:
            spans = [(r["value"]["start"], r["value"]["end"])
                     for r in valid[0].get("result", []) if "start" in r["value"]]
            return spans, "annotation", valid[0]
    if task.predictions:
        p = task.predictions[0]
        spans = [(r["value"]["start"], r["value"]["end"])
                 for r in p.result if "start" in r["value"]]
        return spans, "prediction", p
    return [], None, None


def is_novel(hit, existing_spans, threshold=0.5):
    start, end = hit[0], hit[1]
    hit_len = end - start
    for ex_s, ex_e in existing_spans:
        overlap = max(0, min(end, ex_e) - max(start, ex_s))
        if overlap / hit_len >= threshold:
            return False
    return True


def make_result(start, end, text, label):
    return {
        "id": uuid.uuid4().hex[:8],
        "from_name": "label",
        "to_name": "text",
        "type": "labels",
        "value": {"start": start, "end": end, "text": text, "labels": [label]}
    }


def search_task(client, task, db_path, push, word_boundary):
    text = task.data.get("text", "")
    existing_spans, source_type, source_obj = get_existing(task)

    all_hits = []
    for b, batch in enumerate(iter_title_batches(db_path, BATCH_SIZE), 1):
        A = build_automaton(batch)
        hits = search_automaton(text, A, word_boundary=word_boundary)
        all_hits.extend(hits)
        del A  # release before next batch
        print(f"  Batch {b}: {len(hits)} raw hits (running total: {len(all_hits)})", end="\r")

    print(f"\n  Raw hits           : {len(all_hits):,}")
    all_hits = keep_longest(all_hits)
    print(f"  After dedup        : {len(all_hits):,}")
    novel = [h for h in all_hits if is_novel(h, existing_spans)]
    print(f"  Novel (unlabeled)  : {len(novel):,}")

    # Length histogram
    lengths = [end - start for start, end, *_ in novel]
    buckets = [(6,10), (11,15), (16,20), (21,30), (31,50), (51,100), (101, 99999)]
    print(f"\n  Length distribution of novel candidates:")
    for lo, hi in buckets:
        count = sum(1 for l in lengths if lo <= l <= hi)
        bar = "\u2588" * min(count // max(1, len(lengths) // 30), 30)
        label_str = f"{lo}-{hi}" if hi < 99999 else f"{lo}+"
        print(f"    {label_str:>8} chars:  {count:>5}  {bar}")

    print(f"\n  Top 20 novel candidates (longest first):")
    for start, end, matched, _, _ in sorted(novel, key=lambda x: -(x[1]-x[0]))[:20]:
        display = matched[:70] + "..." if len(matched) > 70 else matched
        print(f"    [{start}:{end}]  {display!r}")

    print(f"\n  Bottom 20 novel candidates (shortest first):")
    for start, end, matched, _, _ in sorted(novel, key=lambda x: x[1]-x[0])[:20]:
        print(f"    [{start}:{end}] ({end-start} chars)  {matched!r}")

    if not push or not novel:
        if not push:
            print(f"\n  Add --push to write {len(novel)} candidates to Label Studio.")
        return len(novel)

    new_results = [make_result(s, e, m, LABEL_TYPE) for s, e, m, _, _ in novel]
    if source_obj and source_type == "prediction":
        merged = list(source_obj.result) + new_results
        client.predictions.update(id=source_obj.id, result=merged)
        print(f"  Pushed: prediction {source_obj.id} updated → {len(merged)} labels.")
    elif source_obj and source_type == "annotation":
        merged = source_obj.get("result", []) + new_results
        client.annotations.update(id=source_obj["id"], result=merged)
        print(f"  Pushed: annotation {source_obj['id']} updated → {len(merged)} labels.")
    else:
        client.predictions.create(task=task.id, result=new_results, score=0.0)
        print(f"  Pushed: new prediction created with {len(new_results)} labels.")

    return len(novel)


# cli
def main():
    parser = argparse.ArgumentParser(description="OL works → Label Studio false-negative detector.")
    sub = parser.add_subparsers(dest="phase", required=True)

    p1 = sub.add_parser("ingest", help="Parse OL dump into SQLite (run once)")
    p1.add_argument("--works",   required=True, help="Path to ol_dump_works_*.txt")
    p1.add_argument("--db",      required=True, help="Output SQLite path (e.g. ol_works.db)")
    p1.add_argument("--min-len", type=int, default=6,
                    help="Min title character length to index (default: 6)")

    p2 = sub.add_parser("search", help="Search tasks for unlabeled works")
    p2.add_argument("--db",               required=True, help="SQLite DB from ingest phase")
    p2.add_argument("--task",             type=int, default=None, help="Single task ID")
    p2.add_argument("--all-tasks",        action="store_true", help="Search all tasks in project")
    p2.add_argument("--push",             action="store_true", help="Write novel hits to Label Studio")
    p2.add_argument("--no-word-boundary", action="store_true", help="Disable whole-word boundary check")

    args = parser.parse_args()

    if args.phase == "ingest":
        ingest(args.works, args.db, args.min_len)

    elif args.phase == "search":
        if not args.task and not args.all_tasks:
            parser.error("Provide --task <id> or --all-tasks")

        client = LabelStudio(base_url=API_URL, api_key=API_KEY)
        wb = not args.no_word_boundary

        if args.task:
            task = client.tasks.get(id=args.task)
            print(f"Task {task.id} | thread: {task.data.get('thread_id')} | {len(task.data.get('text',''))} chars")
            search_task(client, task, args.db, push=args.push, word_boundary=wb)

        elif args.all_tasks:
            page, page_size, total_novel = 1, 500, 0
            while True:
                tasks = client.tasks.list(project=PROJECT_ID, page=page, page_size=page_size)
                if not tasks.items:
                    break
                for task in tasks.items:
                    print(f"\n{'─'*60}")
                    print(f"Task {task.id} | {task.data.get('thread_id')}")
                    total_novel += search_task(client, task, args.db, push=args.push, word_boundary=wb)
                if len(tasks.items) < page_size:
                    break
                page += 1
            print(f"\nTotal novel candidates across all tasks: {total_novel:,}")


if __name__ == "__main__":
    main()
