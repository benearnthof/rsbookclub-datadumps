"""
extract_work_keys.py
--------------------
Single-pass scan of the Open Library data dump to map edition keys → work keys.

Usage (from the folder containing the dump):
    python extract_work_keys.py

Edit the three paths below to match your setup.
"""

import csv
import json
import sys
import time
from pathlib import Path

# ── configure these ──────────────────────────────────────────────────────────
DUMP_PATH   = Path(r"C:\Users\Bene\Desktop\rsbookclub_output\rsbookclub-datadumps\ol_cdump_2026-02-28.txt")
INPUT_CSV   = Path(r"typesense_ol_keys.csv")   # your CSV with the ol_keys column
OUTPUT_CSV  = Path(r"edition_to_work_keys.csv")
# ─────────────────────────────────────────────────────────────────────────────

def load_edition_keys(csv_path: Path) -> dict[str, str]:
    """Return {edition_key: surface_title} from the ol_keys pipe-separated column."""
    edition_to_surface: dict[str, str] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            surface = row.get("surface", "")
            for key in row.get("ol_keys", "").split("|"):
                key = key.strip()
                if key:
                    # Normalise: store as "/books/OL12345M" form
                    if not key.startswith("/"):
                        key = f"/books/{key}"
                    edition_to_surface[key] = surface
    return edition_to_surface


def scan_dump(dump_path: Path, target_keys: set[str]) -> dict[str, list[str]]:
    """
    Stream the dump, return {edition_key: [work_key, ...]} for every hit.
    The dump columns (tab-separated) are:
      0: type   1: key   2: revision   3: last_modified   4: json
    """
    results: dict[str, list[str]] = {}
    found = 0
    needed = len(target_keys)

    print(f"Scanning {dump_path}  ({dump_path.stat().st_size / 1e9:.1f} GB)")
    print(f"Looking for {needed} unique edition keys …\n")

    t0 = time.time()
    lines_read = 0

    with dump_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            lines_read += 1

            # Fast pre-check: skip non-edition lines without splitting
            if "\t/books/" not in line:
                continue

            parts = line.split("\t", 4)
            if len(parts) < 5:
                continue

            rec_type, key = parts[0], parts[1]
            if rec_type != "/type/edition":
                continue
            if key not in target_keys:
                continue

            # Parse JSON only for matching lines
            try:
                data = json.loads(parts[4])
                work_keys = [w["key"] for w in data.get("works", [])]
            except (json.JSONDecodeError, KeyError):
                work_keys = []

            results[key] = work_keys
            found += 1

            elapsed = time.time() - t0
            #print(f"  [{found}/{needed}]  {key}  →  {work_keys}   "
            #      f"(line {lines_read:,}, {elapsed:.0f}s)")

            if found == needed:
                print("\nAll keys found — stopping early.")
                break

            # Progress heartbeat every million lines
            if lines_read % 1_000_000 == 0:
                pct = found / needed * 100
                print(f"  … {lines_read/1e6:.0f}M lines scanned, "
                      f"{found}/{needed} found ({pct:.0f}%), "
                      f"{(time.time()-t0)/60:.1f} min elapsed")

    return results


def write_output(
    output_path: Path,
    edition_to_surface: dict[str, str],
    results: dict[str, list[str]],
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["surface", "edition_key", "work_keys", "work_key_primary"])
        for edition_key, surface in sorted(edition_to_surface.items()):
            work_keys = results.get(edition_key, [])
            writer.writerow([
                surface,
                edition_key,
                "|".join(work_keys),
                work_keys[0] if work_keys else "",
            ])

    print(f"\nOutput written to: {output_path}")
    found_count = sum(1 for v in results.values() if v)
    total = len(edition_to_surface)
    print(f"Matched {found_count} / {total} edition keys to at least one work key.")
    missing = [k for k, v in results.items() if not v]
    no_hit  = [k for k in edition_to_surface if k not in results]
    if missing:
        print(f"  {len(missing)} editions found in dump but had no 'works' field.")
    if no_hit:
        print(f"  {len(no_hit)} edition keys not found in dump at all.")


def main() -> None:
    if not DUMP_PATH.exists():
        sys.exit(f"Dump not found: {DUMP_PATH}")
    if not INPUT_CSV.exists():
        sys.exit(f"Input CSV not found: {INPUT_CSV}")

    edition_to_surface = load_edition_keys(INPUT_CSV)
    print(f"Loaded {len(edition_to_surface)} unique edition keys from {INPUT_CSV}\n")

    target_keys = set(edition_to_surface.keys())
    results = scan_dump(DUMP_PATH, target_keys)
    write_output(OUTPUT_CSV, edition_to_surface, results)


if __name__ == "__main__":
    main()