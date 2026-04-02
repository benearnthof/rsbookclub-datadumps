"""
merge_entity_lookup.py
──────────────────────
Merge three canonical-resolution CSVs into a single entity_lookup.csv.

Sources (in descending priority when a surface appears in multiple files):
  1. resolved_books.csv   – exact / normalized matches with author info
  2. fuzzy_resolved.csv   – fuzzy matches and aliases with author info
  3. canonical_lookup.csv – FTS / fuzzy fallback produced by resolve_canonical.py

Priority rules
──────────────
For a surface that appears in more than one file the row with the highest
*normalised* score wins.  Scores are normalised to [0, 1]:
  • resolved_books  : confidence   (already 0-1)
  • fuzzy_resolved  : fuzzy_score  / 100
  • canonical_lookup: score        / 100

Tie-break (same normalised score): priority order above (1 > 2 > 3).

Unresolved entries (empty canonical_title, method == "unresolved", score 0)
are kept but flagged so downstream consumers can filter easily.

Output columns
──────────────
  surface           – original surface string
  canonical_title   – best resolved title (may be empty if unresolved)
  ol_key            – /works/OL…W  (empty for aliases or unresolved)
  author_name       – from resolved_books / fuzzy_resolved; empty otherwise
  method            – exact | normalized | fuzzy | alias | fts | unresolved
  score             – normalised 0-1 confidence
  ambiguous         – True / False / ""  (empty when not provided)
  notes             – original notes field, prefixed with source file tag
  source            – which file the winning row came from

Usage:
    python merge_entity_lookup.py \
        --resolved   output/resolved_books.csv \
        --fuzzy      output/fuzzy_resolved.csv \
        --canonical  canonical_lookup.csv \
        --output     entity_lookup.csv
"""

import argparse
import csv
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Unified row ───────────────────────────────────────────────────────────────

@dataclass
class Entry:
    surface:         str
    canonical_title: str
    ol_key:          str
    author_name:     str
    method:          str
    score:           float        # normalised 0-1
    ambiguous:       str          # "True" | "False" | ""
    notes:           str
    source:          str
    source_priority: int          # lower = higher priority; for tie-breaking


OUTPUT_FIELDS = [
    "surface", "canonical_title", "ol_key", "author_name",
    "method", "score", "ambiguous", "notes", "source",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _str_bool(val: str) -> str:
    """Normalise various bool representations to 'True'/'False'/''."""
    v = val.strip().lower()
    if v in ("true", "1", "yes"):
        return "True"
    if v in ("false", "0", "no"):
        return "False"
    return ""


def _safe_float(val: str, divisor: float = 1.0) -> float:
    try:
        return float(val.strip()) / divisor
    except (ValueError, ZeroDivisionError):
        return 0.0


def _is_unresolved(e: Entry) -> bool:
    return e.method == "unresolved" or (not e.canonical_title and e.method != "alias")


# ── Readers ───────────────────────────────────────────────────────────────────

def read_resolved_books(path: Path) -> list[Entry]:
    """
    Columns: surface, method, ol_key, canonical_title,
             author_name, confidence, ambiguous, notes
    score field: confidence (0-1 already)
    """
    entries = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            entries.append(Entry(
                surface         = row["surface"].strip(),
                canonical_title = row["canonical_title"].strip(),
                ol_key          = row.get("ol_key", "").strip(),
                author_name     = row.get("author_name", "").strip(),
                method          = row.get("method", "").strip(),
                score           = _safe_float(row.get("confidence", "0")),
                ambiguous       = _str_bool(row.get("ambiguous", "")),
                notes           = row.get("notes", "").strip(),
                source          = path.name,
                source_priority = 1,
            ))
    log.info("  resolved_books:   %d rows", len(entries))
    return entries


def read_fuzzy_resolved(path: Path) -> list[Entry]:
    """
    Columns: surface, method, ol_key, canonical_title,
             author_name, fuzzy_score, ambiguous, notes
    score field: fuzzy_score (0-100) → divide by 100
    """
    entries = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            entries.append(Entry(
                surface         = row["surface"].strip(),
                canonical_title = row["canonical_title"].strip(),
                ol_key          = row.get("ol_key", "").strip(),
                author_name     = row.get("author_name", "").strip(),
                method          = row.get("method", "").strip(),
                score           = _safe_float(row.get("fuzzy_score", "0"), divisor=100.0),
                ambiguous       = _str_bool(row.get("ambiguous", "")),
                notes           = row.get("notes", "").strip(),
                source          = path.name,
                source_priority = 2,
            ))
    log.info("  fuzzy_resolved:   %d rows", len(entries))
    return entries


def read_canonical_lookup(path: Path) -> list[Entry]:
    """
    Columns: surface, canonical_title, work_key, method, score
    score field: score (0-100) → divide by 100
    No author_name or ambiguous columns.
    """
    entries = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            entries.append(Entry(
                surface         = row["surface"].strip(),
                canonical_title = row["canonical_title"].strip(),
                ol_key          = row.get("work_key", "").strip(),
                author_name     = "",
                method          = row.get("method", "").strip(),
                score           = _safe_float(row.get("score", "0"), divisor=100.0),
                ambiguous       = "",
                notes           = row.get("notes", "").strip(),
                source          = path.name,
                source_priority = 3,
            ))
    log.info("  canonical_lookup: %d rows", len(entries))
    return entries


# ── Merge logic ───────────────────────────────────────────────────────────────

def merge(all_entries: list[Entry]) -> list[Entry]:
    """
    For each surface keep the single best Entry by:
      1. highest normalised score
      2. lowest source_priority (tie-break)
    Unresolved entries from lower-priority sources are discarded if a
    better (even partially resolved) entry exists from any source.
    """
    best: dict[str, Entry] = {}

    for e in all_entries:
        key = e.surface

        if key not in best:
            best[key] = e
            continue

        incumbent = best[key]

        # Prefer any resolved entry over an unresolved one
        inc_unres = _is_unresolved(incumbent)
        new_unres = _is_unresolved(e)

        if inc_unres and not new_unres:
            best[key] = e
            continue
        if not inc_unres and new_unres:
            continue

        # Both resolved (or both unresolved): compare scores, then priority
        if e.score > incumbent.score:
            best[key] = e
        elif e.score == incumbent.score and e.source_priority < incumbent.source_priority:
            best[key] = e

    return list(best.values())


# ── Writer ────────────────────────────────────────────────────────────────────

def write_output(entries: list[Entry], path: Path) -> None:
    # Sort: resolved first (by score desc), then unresolved alpha
    resolved   = sorted(
        [e for e in entries if not _is_unresolved(e)],
        key=lambda e: (-e.score, e.surface.lower()),
    )
    unresolved = sorted(
        [e for e in entries if _is_unresolved(e)],
        key=lambda e: e.surface.lower(),
    )
    ordered = resolved + unresolved

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for e in ordered:
            writer.writerow({
                "surface":         e.surface,
                "canonical_title": e.canonical_title,
                "ol_key":          e.ol_key,
                "author_name":     e.author_name,
                "method":          e.method,
                "score":           f"{e.score:.4f}",
                "ambiguous":       e.ambiguous,
                "notes":           e.notes,
                "source":          e.source,
            })

    log.info("Wrote %d rows → %s", len(ordered), path.resolve())


# ── Stats ─────────────────────────────────────────────────────────────────────

def print_stats(entries: list[Entry]) -> None:
    from collections import Counter
    method_counts  = Counter(e.method for e in entries)
    source_counts  = Counter(e.source for e in entries)
    n_unresolved   = sum(1 for e in entries if _is_unresolved(e))
    n_ambiguous    = sum(1 for e in entries if e.ambiguous == "True")
    n_total        = len(entries)

    log.info("─" * 55)
    log.info("Total unique surfaces: %d", n_total)
    log.info("  resolved:    %d  (%.1f%%)", n_total - n_unresolved,
             100 * (n_total - n_unresolved) / max(n_total, 1))
    log.info("  unresolved:  %d  (%.1f%%)", n_unresolved,
             100 * n_unresolved / max(n_total, 1))
    log.info("  ambiguous:   %d", n_ambiguous)
    log.info("By method:")
    for method, count in method_counts.most_common():
        log.info("    %-15s %d", method, count)
    log.info("Winning source:")
    for source, count in source_counts.most_common():
        log.info("    %-35s %d", source, count)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Merge entity lookup CSVs")
    p.add_argument("--resolved",  required=True, help="resolved_books.csv")
    p.add_argument("--fuzzy",     required=True, help="fuzzy_resolved.csv")
    p.add_argument("--canonical", required=True, help="canonical_lookup.csv")
    p.add_argument("--output",    default="entity_lookup.csv")
    return p.parse_args()


def main():
    args  = parse_args()
    paths = {
        "resolved":  Path(args.resolved),
        "fuzzy":     Path(args.fuzzy),
        "canonical": Path(args.canonical),
    }
    for label, p in paths.items():
        if not p.exists():
            log.error("File not found (%s): %s", label, p)
            sys.exit(1)

    log.info("Reading source files …")
    all_entries = (
        read_resolved_books(paths["resolved"])
        + read_fuzzy_resolved(paths["fuzzy"])
        + read_canonical_lookup(paths["canonical"])
    )
    log.info("Total rows across all sources: %d", len(all_entries))

    log.info("Merging …")
    merged = merge(all_entries)

    write_output(merged, Path(args.output))
    print_stats(merged)


if __name__ == "__main__":
    main()

