"""
enrich_work_titles.py
─────────────────────
Read edition_to_work_keys.csv, group by surface string, look up every
work key in the OL SQLite database to fetch its title, and write one
consolidated row per surface to an output CSV.

Input columns  (edition_to_work_keys.csv):
    surface            – raw candidate string
    edition_key        – /books/OL…M
    work_keys          – pipe-separated /works/OL…W  (may be empty)
    work_key_primary   – single /works/OL…W           (may be empty)

Output columns  (enriched_surfaces.csv):
    surface
    edition_keys       – pipe-separated, deduplicated, order-preserved
    work_keys          – pipe-separated, deduplicated, order-preserved
    work_keys_primary  – pipe-separated, deduplicated, order-preserved
    work_titles        – JSON object  {"/works/OL…W": "Title text", …}
                         (empty string if no work keys resolved)

Usage:
    python enrich_work_titles.py \
        --input  edition_to_work_keys.csv \
        --db     ol.db \
        --output enriched_surfaces.csv

Optional flags:
    --missing-title  TEXT   Placeholder for work keys not found in db
                            (default: "[NOT FOUND]")
    --batch-log      INT    Log progress every N surfaces (default: 1000)
"""

import argparse
import csv
import json
import logging
import sys
import time
from collections import OrderedDict
from pathlib import Path

# ol_db must be importable – either on PYTHONPATH or in the same directory.
from ol_db import OLDatabase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_pipe(cell: str) -> list[str]:
    """Split a pipe-delimited cell into a list, skipping blanks."""
    return [v.strip() for v in cell.split("|") if v.strip()]


def dedup_ordered(items):
    """Remove duplicates while preserving first-seen order."""
    seen = set()
    out = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


# ── main logic ────────────────────────────────────────────────────────────────

def load_and_group(input_path: Path) -> OrderedDict:
    """
    Read the CSV and group rows by surface string.

    Returns an OrderedDict keyed by surface, each value being:
        {
            "edition_keys":     [str, …],   # ordered, deduped across rows
            "work_keys":        [str, …],
            "work_keys_primary":[str, …],
        }
    """
    groups: OrderedDict = OrderedDict()

    with input_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"surface", "edition_key", "work_keys", "work_key_primary"}
        if not required.issubset(set(reader.fieldnames or [])):
            missing = required - set(reader.fieldnames or [])
            log.error("CSV missing columns: %s", missing)
            sys.exit(1)

        for row in reader:
            surface = row["surface"].strip()
            if not surface:
                continue

            if surface not in groups:
                groups[surface] = {
                    "edition_keys":      [],
                    "work_keys":         [],
                    "work_keys_primary": [],
                }
            g = groups[surface]

            ek = row["edition_key"].strip()
            if ek:
                g["edition_keys"].append(ek)

            for wk in parse_pipe(row["work_keys"]):
                g["work_keys"].append(wk)

            pk = row["work_key_primary"].strip()
            if pk:
                g["work_keys_primary"].append(pk)

    # Deduplicate within each group
    for g in groups.values():
        g["edition_keys"]      = dedup_ordered(g["edition_keys"])
        g["work_keys"]         = dedup_ordered(g["work_keys"])
        g["work_keys_primary"] = dedup_ordered(g["work_keys_primary"])

    log.info("Loaded %d unique surfaces from %s", len(groups), input_path.name)
    return groups


def collect_all_work_keys(groups: OrderedDict) -> set[str]:
    """Gather every distinct work key that needs a title lookup."""
    keys = set()
    for g in groups.values():
        keys.update(g["work_keys"])
        keys.update(g["work_keys_primary"])
    return keys


def fetch_titles(db: OLDatabase, work_keys: set[str], missing_placeholder: str) -> dict[str, str]:
    """
    Bulk-fetch titles for all work keys.

    Returns dict: work_key → title string (or placeholder if not found).
    """
    title_map: dict[str, str] = {}
    total = len(work_keys)
    log.info("Fetching titles for %d distinct work keys …", total)
    t0 = time.perf_counter()

    for i, wk in enumerate(work_keys, 1):
        work = db.get_work(wk)
        title_map[wk] = work.title if work else missing_placeholder
        if i % 50_000 == 0:
            log.info("  … %d / %d titles fetched", i, total)

    log.info("Title fetch complete in %.1fs", time.perf_counter() - t0)
    return title_map


def write_output(
    output_path: Path,
    groups: OrderedDict,
    title_map: dict[str, str],
    batch_log: int,
) -> None:
    FIELDNAMES = [
        "surface",
        "edition_keys",
        "work_keys",
        "work_keys_primary",
        "work_titles",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, quoting=csv.QUOTE_ALL)
        writer.writeheader()

        for i, (surface, g) in enumerate(groups.items(), 1):
            # Build {work_key: title} for every work key associated with
            # this surface (union of work_keys and work_keys_primary).
            all_wks = dedup_ordered(g["work_keys"] + g["work_keys_primary"])
            titles_obj = {wk: title_map.get(wk, "") for wk in all_wks}

            writer.writerow({
                "surface":           surface,
                "edition_keys":      "|".join(g["edition_keys"]),
                "work_keys":         "|".join(g["work_keys"]),
                "work_keys_primary": "|".join(g["work_keys_primary"]),
                "work_titles":       json.dumps(titles_obj, ensure_ascii=False) if titles_obj else "",
            })

            if i % batch_log == 0:
                log.info("  … %d surfaces written", i)

    log.info("Output written → %s  (%d rows)", output_path.resolve(), len(groups))


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Enrich edition_to_work_keys.csv with OL work titles"
    )
    p.add_argument("--input",  default="edition_to_work_keys.csv",
                   help="Input CSV (default: edition_to_work_keys.csv)")
    p.add_argument("--db",     default="ol.db",
                   help="OL SQLite database (default: ol.db)")
    p.add_argument("--output", default="enriched_surfaces.csv",
                   help="Output CSV (default: enriched_surfaces.csv)")
    p.add_argument("--missing-title", default="[NOT FOUND]",
                   help="Placeholder for work keys absent from the DB")
    p.add_argument("--batch-log", type=int, default=1000,
                   help="Log progress every N surfaces (default: 1000)")
    return p.parse_args()


def main():
    args = parse_args()
    input_path  = Path(args.input)
    db_path     = Path(args.db)
    output_path = Path(args.output)

    if not input_path.exists():
        log.error("Input file not found: %s", input_path)
        sys.exit(1)
    if not db_path.exists():
        log.error("Database not found: %s", db_path)
        sys.exit(1)

    # ── 1. Group CSV rows by surface ──────────────────────────────────────
    groups = load_and_group(input_path)

    # ── 2. Open DB and batch-fetch all titles ─────────────────────────────
    log.info("Opening database: %s", db_path)
    db = OLDatabase(db_path)
    try:
        all_work_keys = collect_all_work_keys(groups)
        title_map     = fetch_titles(db, all_work_keys, args.missing_title)
    finally:
        db.close()

    # ── 3. Write enriched output ──────────────────────────────────────────
    write_output(output_path, groups, title_map, args.batch_log)

    # ── 4. Quick sanity numbers ───────────────────────────────────────────
    found   = sum(1 for t in title_map.values() if t != args.missing_title)
    missing = len(title_map) - found
    log.info("─" * 55)
    log.info("Surfaces:         %d", len(groups))
    log.info("Distinct wk keys: %d", len(title_map))
    log.info("  titles found:   %d", found)
    log.info("  titles missing: %d", missing)
    log.info("Done.")


if __name__ == "__main__":
    main()