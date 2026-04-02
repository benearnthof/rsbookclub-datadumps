"""
resolve_canonical.py
────────────────────
Convert enriched_surfaces.csv into a clean surface → canonical title lookup.

Strategy (in order of confidence):
  1. EXACT  – surface normalises to match a candidate title exactly.
  2. FUZZY  – rank candidate titles from work_titles JSON by
              token-sort-ratio similarity; accept if score ≥ FUZZY_THRESHOLD.
  3. FTS    – when no candidates exist (work_titles empty), run
              ol_db.search_works_fts() and pick the best fuzzy match.
  4. UNRESOLVED – nothing scored high enough; record for manual review.

Output files:
  canonical_lookup.csv      – surface, canonical_title, work_key, method, score
  unresolved_surfaces.csv   – surfaces that need manual review

Dependencies:  pip install rapidfuzz
Usage:
    python resolve_canonical.py \
        --input  enriched_surfaces.csv \
        --db     ol.db \
        --output canonical_lookup.csv \
        --unresolved unresolved_surfaces.csv
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

from rapidfuzz import fuzz
from ol_db import OLDatabase, normalize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
FUZZY_THRESHOLD  = 72   # min token_sort_ratio to accept a candidate
FTS_LIMIT        = 20   # how many FTS results to consider for fallback
FTS_MIN_SCORE    = 60   # lower bar for FTS matches (surface may be abbreviated)


# ── Similarity scoring ────────────────────────────────────────────────────────

def score_pair(surface: str, title: str) -> float:
    """
    Combine multiple rapidfuzz metrics to handle:
      - different word orders   (token_sort_ratio)
      - surface being a subset  (token_set_ratio)
      - straight similarity     (ratio)
    Returns the maximum of the three, 0-100.
    """
    norm_s = normalize(surface)
    norm_t = normalize(title)
    return max(
        fuzz.token_sort_ratio(norm_s, norm_t),
        fuzz.token_set_ratio(norm_s, norm_t),
        fuzz.ratio(norm_s, norm_t),
    )


def best_candidate(surface: str, title_map: dict[str, str]) -> tuple[str, str, float]:
    """
    Given a {work_key: title} map, return (best_work_key, best_title, score).
    Returns ("", "", 0.0) if title_map is empty.
    """
    best_key, best_title, best_score = "", "", 0.0
    for wk, title in title_map.items():
        if not title or title == "[NOT FOUND]":
            continue
        s = score_pair(surface, title)
        if s > best_score:
            best_score = s
            best_key   = wk
            best_title = title
    return best_key, best_title, best_score


# ── Resolution logic ──────────────────────────────────────────────────────────

def resolve_row(surface: str, work_titles_raw: str, db: OLDatabase) -> dict:
    """
    Returns a result dict:
        canonical_title, work_key, method, score
    method ∈ {"exact", "fuzzy", "fts", "unresolved"}
    """
    norm_surface = normalize(surface)

    # ── Parse candidate titles ────────────────────────────────────────────
    title_map: dict[str, str] = {}
    if work_titles_raw:
        try:
            title_map = json.loads(work_titles_raw)
        except json.JSONDecodeError:
            log.warning("Bad JSON for surface %r – treating as no candidates", surface)

    # ── 1. EXACT match ────────────────────────────────────────────────────
    for wk, title in title_map.items():
        if title and normalize(title) == norm_surface:
            return {"canonical_title": title, "work_key": wk,
                    "method": "exact", "score": 100.0}

    # ── 2. FUZZY match from existing candidates ───────────────────────────
    if title_map:
        wk, title, score = best_candidate(surface, title_map)
        if score >= FUZZY_THRESHOLD:
            return {"canonical_title": title, "work_key": wk,
                    "method": "fuzzy", "score": round(score, 1)}

    # ── 3. FTS fallback (covers empty work_titles AND low-confidence fuzzy) ─
    fts_results = db.search_works_fts(surface, limit=FTS_LIMIT)
    if fts_results:
        fts_map = {w.ol_key: w.title for w in fts_results if w.title}
        wk, title, score = best_candidate(surface, fts_map)
        if score >= FTS_MIN_SCORE:
            return {"canonical_title": title, "work_key": wk,
                    "method": "fts", "score": round(score, 1)}

    # ── 4. Unresolved ─────────────────────────────────────────────────────
    # Store best available info even if below threshold, for manual review
    best_hint = ""
    best_hint_score = 0.0
    best_hint_key = ""
    if title_map:
        wk, title, score = best_candidate(surface, title_map)
        best_hint, best_hint_score, best_hint_key = title, score, wk
    return {"canonical_title": best_hint, "work_key": best_hint_key,
            "method": "unresolved", "score": round(best_hint_score, 1)}


# ── I/O helpers ───────────────────────────────────────────────────────────────

def process(input_path: Path, db: OLDatabase,
            output_path: Path, unresolved_path: Path) -> None:

    out_fields        = ["surface", "canonical_title", "work_key", "method", "score"]
    unresolved_fields = ["surface", "best_candidate_title", "best_candidate_key",
                         "best_score", "work_titles_raw"]

    counts = {"exact": 0, "fuzzy": 0, "fts": 0, "unresolved": 0}
    t0 = time.perf_counter()

    with (
        input_path.open(newline="", encoding="utf-8")          as fin,
        output_path.open("w", newline="", encoding="utf-8")    as fout,
        unresolved_path.open("w", newline="", encoding="utf-8") as funres,
    ):
        reader   = csv.DictReader(fin)
        writer   = csv.DictWriter(fout,   fieldnames=out_fields,        quoting=csv.QUOTE_ALL)
        unwriter = csv.DictWriter(funres, fieldnames=unresolved_fields,  quoting=csv.QUOTE_ALL)
        writer.writeheader()
        unwriter.writeheader()

        for i, row in enumerate(reader, 1):
            surface          = row["surface"].strip()
            work_titles_raw  = row.get("work_titles", "").strip()

            result = resolve_row(surface, work_titles_raw, db)
            method = result["method"]
            counts[method] += 1

            writer.writerow({
                "surface":         surface,
                "canonical_title": result["canonical_title"],
                "work_key":        result["work_key"],
                "method":          method,
                "score":           result["score"],
            })

            if method == "unresolved":
                unwriter.writerow({
                    "surface":               surface,
                    "best_candidate_title":  result["canonical_title"],
                    "best_candidate_key":    result["work_key"],
                    "best_score":            result["score"],
                    "work_titles_raw":       work_titles_raw,
                })

            if i % 500 == 0:
                log.info("  … %d surfaces processed", i)

    elapsed = time.perf_counter() - t0
    total   = sum(counts.values())
    log.info("─" * 55)
    log.info("Total surfaces:  %d  in %.1fs", total, elapsed)
    log.info("  exact:         %d  (%.1f%%)", counts["exact"],
             100 * counts["exact"] / max(total, 1))
    log.info("  fuzzy:         %d  (%.1f%%)", counts["fuzzy"],
             100 * counts["fuzzy"] / max(total, 1))
    log.info("  fts fallback:  %d  (%.1f%%)", counts["fts"],
             100 * counts["fts"] / max(total, 1))
    log.info("  unresolved:    %d  (%.1f%%)", counts["unresolved"],
             100 * counts["unresolved"] / max(total, 1))
    log.info("Lookup  → %s", output_path.resolve())
    log.info("Review  → %s", unresolved_path.resolve())


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Resolve surface strings to canonical OL titles"
    )
    p.add_argument("--input",      default="enriched_surfaces.csv")
    p.add_argument("--db",         default="ol.db")
    p.add_argument("--output",     default="canonical_lookup.csv")
    p.add_argument("--unresolved", default="unresolved_surfaces.csv")
    p.add_argument("--fuzzy-threshold", type=float, default=FUZZY_THRESHOLD,
                   help=f"Min score to accept fuzzy candidate (default {FUZZY_THRESHOLD})")
    p.add_argument("--fts-min-score",   type=float, default=FTS_MIN_SCORE,
                   help=f"Min score to accept FTS result (default {FTS_MIN_SCORE})")
    return p.parse_args()


def main():
    args = parse_args()

    # Allow threshold overrides from CLI
    global FUZZY_THRESHOLD, FTS_MIN_SCORE
    FUZZY_THRESHOLD = args.fuzzy_threshold
    FTS_MIN_SCORE   = args.fts_min_score

    input_path     = Path(args.input)
    db_path        = Path(args.db)
    output_path    = Path(args.output)
    unresolved_path = Path(args.unresolved)

    for p in (input_path, db_path):
        if not p.exists():
            log.error("Not found: %s", p)
            sys.exit(1)

    log.info("Opening database: %s", db_path)
    db = OLDatabase(db_path)
    try:
        process(input_path, db, output_path, unresolved_path)
    finally:
        db.close()


if __name__ == "__main__":
    main()