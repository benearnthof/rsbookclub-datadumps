#!/usr/bin/env python3
"""
Push accepted.json entities to Label Studio.
# preliminary updates after accepting 708 book entities:
# 551 tasks to update 
# 708 new BOOK spans
# 436 superseded BOOK spans
# Since we began verifying the longest candidates first, the false positive rate would
# only increase the shorter the matches get. 

Usage
  python push_accepted.py

  Dry run: print what would change without touching Label Studio:
  python push_accepted.py --dry-run

Options
  --accepted   PATH   accepted.json  (default: accepted.json)
  --dry-run           Print plan without writing to Label Studio
"""

import argparse
import json
import uuid
from collections import defaultdict
from pathlib import Path

from label_studio_sdk import LabelStudio # type: ignore

API_URL = "http://localhost:8080"
API_KEY = ""


def make_result(start: int, end: int, text: str, label: str) -> dict:
    return {
        "id":        uuid.uuid4().hex[:8],
        "from_name": "label",
        "to_name":   "text",
        "type":      "labels",
        "value": {"start": start, "end": end, "text": text, "labels": [label]},
    }


def spans_overlap(s1: int, e1: int, s2: int, e2: int) -> bool:
    return max(s1, s2) < min(e1, e2)


def get_source(task) -> tuple[str | None, int | None, list[dict]]:
    """
    Return (source_type, source_id, result_list) for the first usable
    annotation or prediction on the task.
    """
    if task.annotations:
        valid = [a for a in task.annotations if not a.get("was_cancelled")]
        if valid:
            a = valid[0]
            return "annotation", a["id"], list(a.get("result", []))
    if task.predictions:
        p = task.predictions[0]
        return "prediction", p.id, list(p.result)
    return None, None, []


def merge(existing: list[dict], new_candidates: list[dict]) -> tuple[list[dict], int]:
    """
    Drop existing BOOK spans that overlap any accepted candidate, keep all
    WRITER spans, then append the new BOOK spans.

    Returns (merged_result, n_dropped).
    """
    # Build (start, end) pairs for accepted candidates
    new_spans = [(c["start"], c["end"]) for c in new_candidates]

    kept    = []
    dropped = 0
    for r in existing:
        v      = r.get("value", {})
        labels = v.get("labels", [])
        if "BOOK" in labels:
            rs, re = v.get("start", 0), v.get("end", 0)
            if any(spans_overlap(rs, re, ns, ne) for ns, ne in new_spans):
                dropped += 1
                continue   # superseded by verified OL match
        kept.append(r)

    new_results = [
        make_result(c["start"], c["end"], c["text"], "BOOK")
        for c in new_candidates
    ]
    return kept + new_results, dropped


def run(accepted_path: Path, dry_run: bool) -> None:
    with accepted_path.open(encoding="utf-8") as fh:
        accepted: list[dict] = json.load(fh)

    # Group by task_id
    by_task: dict[int, list[dict]] = defaultdict(list)
    for rec in accepted:
        by_task[rec["task_id"]].append(rec)

    print(f"Accepted entities : {len(accepted):,}")
    print(f"Tasks to update   : {len(by_task):,}")
    if dry_run:
        print("DRY RUN :nothing will be written.\n")
    else:
        print()

    ls = LabelStudio(base_url=API_URL, api_key=API_KEY)

    updated   = 0
    created   = 0
    errors    = []
    total_dropped = 0
    total_added   = 0

    for i, (task_id, candidates) in enumerate(sorted(by_task.items()), 1):
        print(f"  [{i:>4}/{len(by_task)}] task {task_id}  "
              f"({len(candidates)} new span{'s' if len(candidates) != 1 else ''})",
              end="  ", flush=True)
        try:
            task = ls.tasks.get(id=task_id)
            source_type, source_id, existing = get_source(task)
            merged, dropped = merge(existing, candidates)

            total_dropped += dropped
            total_added   += len(candidates)

            print(f"dropped {dropped} old BOOK span{'s' if dropped != 1 else ''},  "
                  f"total spans → {len(merged)}")

            if dry_run:
                continue

            if source_type == "annotation":
                ls.annotations.update(id=source_id, result=merged)
                updated += 1
            elif source_type == "prediction":
                ls.predictions.update(id=source_id, result=merged)
                updated += 1
            else:
                ls.predictions.create(task=task_id, result=merged, score=0.0)
                created += 1

        except Exception as exc:
            print(f"ERROR: {exc}")
            errors.append(f"task {task_id}: {exc}")

    print(f"\n{'─'*50}")
    if dry_run:
        print(f"[dry-run] Would add    : {total_added:,} BOOK spans")
        print(f"[dry-run] Would drop   : {total_dropped:,} overlapping old BOOK spans")
        print(f"[dry-run] Would update : {len(by_task):,} tasks")
    else:
        print(f"Added    : {total_added:,} new BOOK spans")
        print(f"Dropped  : {total_dropped:,} superseded BOOK spans")
        print(f"Updated  : {updated:,} existing sources")
        print(f"Created  : {created:,} new predictions")
        if errors:
            print(f"Errors   : {len(errors):,}")
            for e in errors[:10]:
                print(f"  {e}")
    print(f"{'─'*50}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--accepted",
                   type=Path, default=Path("accepted.json"),
                   help="accepted.json (default: accepted.json)")
    p.add_argument("--dry-run",
                   action="store_true",
                   help="Print plan without writing to Label Studio")
    args = p.parse_args()

    run(accepted_path=args.accepted, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

