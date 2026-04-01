"""
disambiguate_books.py
─────────────────────
Resolves BOOK entity surface forms from the Label Studio export to
Open Library work keys, using a four-stage cascade:

  1. Acronym lookup        (acronyms.json, subreddit-specific, exact)
  2. Exact DB lookup       (works.title == surface form)
  3. Normalised DB lookup  (unicode/punct/case folded)
  4. Failed → failed_lookup.csv  for manual review

Thread context is used to boost or confirm candidates:
  - A WRITER entity in the same annotation that matches an author
    of a candidate work raises confidence to 1.0.
  - Multiple works returned by a lookup are ranked by
    (a) author cross-match, (b) edition-count proxy (OL revision),
    (c) surface-form similarity score.

Outputs
───────
  resolved_books.csv      – one row per unique surface form
  failed_lookup.csv       – surface forms with no confident match
  entity_links.csv        – one row per raw entity occurrence,
                            joined to resolved_books

Usage
─────
  python disambiguate_books.py \
      --dataset  dataset-at-2026-03-31.json \
      --db       ol.db \
      --acronyms acronyms.json \
      --out-dir  ./output

  # Quick smoke-test on first 25 unique surface forms:
  python disambiguate_books.py ... --smoke-test
"""

import argparse
import csv
import json
import logging
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ol_db.py must be on the path (same directory or installed)
from oldb import OLDatabase, Work, Author, normalize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROGRESS_INTERVAL = 100   # update bar postfix every N resolved surface forms
SMOKE_TEST_LIMIT  = 25    # number of unique surface forms in smoke-test mode


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class RawEntity:
    """One labeled span from the Label Studio export."""
    task_id:   int
    thread_id: str
    label:     str          # 'BOOK' or 'WRITER'
    text:      str          # surface form as labeled
    start:     int
    end:       int


@dataclass
class Resolution:
    surface:        str
    method:         str             # 'acronym' | 'exact' | 'normalized' | 'failed'
    ol_key:         Optional[str]   # /works/OL…W
    canonical_title: Optional[str]
    author_name:    Optional[str]   # first author if available
    confidence:     float           # 0.0 – 1.0
    ambiguous:      bool = False
    notes:          str  = ""


# ── Dataset parsing ──────────────────────────────────────────────────────────

def load_entities(dataset_path: Path) -> list[RawEntity]:
    """Extract all BOOK and WRITER spans from the Label Studio JSON export."""
    with dataset_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    entities: list[RawEntity] = []
    for task in data:
        task_id   = task.get("id", -1)
        thread_id = task.get("data", {}).get("thread_id", "")

        # Prefer completed annotations; fall back to drafts
        annotations = task.get("annotations") or []
        completed   = [a for a in annotations if not a.get("was_cancelled")]
        source      = completed[0] if completed else None
        if source is None:
            drafts = task.get("drafts") or []
            source = drafts[0] if drafts else None
        if source is None:
            continue

        results = source.get("result") or []
        for span in results:
            val = span.get("value", {})
            labels = val.get("labels", [])
            if not labels:
                continue
            label = labels[0]
            if label not in ("BOOK", "WRITER"):
                continue
            entities.append(RawEntity(
                task_id   = task_id,
                thread_id = thread_id,
                label     = label,
                text      = val.get("text", "").strip(),
                start     = val.get("start", 0),
                end       = val.get("end", 0),
            ))

    log.info("Loaded %d raw entities from %s", len(entities), dataset_path.name)
    return entities


def build_thread_index(entities: list[RawEntity]) -> dict[str, list[RawEntity]]:
    """task_id → list of entities in that task (for context lookups)."""
    idx: dict[str, list[RawEntity]] = {}
    for e in entities:
        idx.setdefault(str(e.task_id), []).append(e)
    return idx


# ── Context helpers ──────────────────────────────────────────────────────────

def writer_names_in_task(task_id: int,
                          thread_idx: dict[str, list[RawEntity]]) -> set[str]:
    """Return normalised WRITER surface forms co-occurring in the same task."""
    spans = thread_idx.get(str(task_id), [])
    return {normalize(e.text) for e in spans if e.label == "WRITER"}


def author_matches_context(work: Work,
                            db: OLDatabase,
                            context_writers: set[str]) -> bool:
    """True if any author of `work` matches a WRITER entity in the same task."""
    if not context_writers:
        return False
    for ak in work.author_keys:
        author = db.get_author(ak)
        if author and normalize(author.name) in context_writers:
            return True
        # Also check last name only
        if author:
            last = normalize(author.name.split()[-1]) if author.name else ""
            if last and last in context_writers:
                return True
    return False


# ── Candidate ranking ────────────────────────────────────────────────────────

def _simple_similarity(a: str, b: str) -> float:
    """Rough character-overlap similarity (Dice on bigrams), 0–1."""
    def bigrams(s):
        s = normalize(s)
        return set(s[i:i+2] for i in range(len(s) - 1))
    bg_a, bg_b = bigrams(a), bigrams(b)
    if not bg_a or not bg_b:
        return 1.0 if normalize(a) == normalize(b) else 0.0
    return 2 * len(bg_a & bg_b) / (len(bg_a) + len(bg_b))


def rank_candidates(surface: str,
                    candidates: list[Work],
                    db: OLDatabase,
                    context_writers: set[str]) -> list[tuple[Work, float]]:
    """
    Score each candidate work and return sorted (work, score) pairs.
    Score components (each 0–1, weighted sum):
      - author_ctx   0.50  : author found in thread context
      - title_sim    0.35  : bigram similarity of canonical title to surface
      - has_subjects 0.15  : work has subject metadata (data quality proxy)
    """
    scored = []
    for w in candidates:
        author_ctx   = 1.0 if author_matches_context(w, db, context_writers) else 0.0
        title_sim    = _simple_similarity(surface, w.title)
        has_subjects = 0.15 if w.subjects else 0.0
        score = 0.50 * author_ctx + 0.35 * title_sim + has_subjects
        scored.append((w, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# ── Resolution stages ─────────────────────────────────────────────────────────

def resolve_via_acronym(surface: str,
                        acronyms: dict,
                        db: OLDatabase) -> Optional[Resolution]:
    entry = acronyms.get(surface)
    if not entry:
        return None
    canonical_title = entry.get("BOOK")
    author_name     = entry.get("WRITER")
    if not canonical_title:
        return None

    # Try to find the OL key for the canonical title
    works = db.find_works_normalized(canonical_title)
    ol_key = works[0].ol_key if works else None

    return Resolution(
        surface         = surface,
        method          = "acronym",
        ol_key          = ol_key,
        canonical_title = canonical_title,
        author_name     = author_name,
        confidence      = 1.0,
        notes           = f"acronym→'{canonical_title}'",
    )


def resolve_via_exact(surface: str,
                      db: OLDatabase,
                      context_writers: set[str]) -> Optional[Resolution]:
    works = db.find_works_exact(surface)
    if not works:
        return None
    return _build_resolution(surface, works, db, context_writers, method="exact")


def resolve_via_normalized(surface: str,
                            db: OLDatabase,
                            context_writers: set[str]) -> Optional[Resolution]:
    works = db.find_works_normalized(surface)
    if not works:
        return None
    return _build_resolution(surface, works, db, context_writers, method="normalized")


def _build_resolution(surface: str,
                      candidates: list[Work],
                      db: OLDatabase,
                      context_writers: set[str],
                      method: str) -> Resolution:
    if len(candidates) == 1:
        w = candidates[0]
        author = db.get_author(w.author_keys[0]) if w.author_keys else None
        ctx_match = author_matches_context(w, db, context_writers)
        confidence = 1.0 if ctx_match else 0.80
        return Resolution(
            surface         = surface,
            method          = method,
            ol_key          = w.ol_key,
            canonical_title = w.title,
            author_name     = author.name if author else None,
            confidence      = confidence,
            notes           = "ctx_confirmed" if ctx_match else "single_match",
        )

    # Multiple candidates — rank them
    ranked = rank_candidates(surface, candidates, db, context_writers)
    best_work, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    author = db.get_author(best_work.author_keys[0]) if best_work.author_keys else None
    ambiguous = (best_score - second_score) < 0.20 and best_score < 0.80

    return Resolution(
        surface         = surface,
        method          = method,
        ol_key          = best_work.ol_key,
        canonical_title = best_work.title,
        author_name     = author.name if author else None,
        confidence      = round(best_score, 3),
        ambiguous       = ambiguous,
        notes           = f"{len(candidates)}_candidates;top_score={best_score:.2f}",
    )


# ── Main pipeline ────────────────────────────────────────────────────────────

def _make_postfix(n_acronym: int, n_exact: int,
                  n_norm: int, n_failed: int) -> dict:
    """Compact tqdm postfix dict showing live resolution breakdown."""
    return {
        "acr": n_acronym,
        "exact": n_exact,
        "norm": n_norm,
        "fail": n_failed,
    }


def disambiguate_books(entities:    list[RawEntity],
                       thread_idx:  dict[str, list[RawEntity]],
                       db:          OLDatabase,
                       acronyms:    dict,
                       smoke_test:  bool = False) -> dict[str, Resolution]:
    """
    Resolve every unique BOOK surface form.
    Returns surface_form → Resolution mapping.

    If smoke_test=True, only the first SMOKE_TEST_LIMIT unique surface forms
    (alphabetical) are processed so you can check for errors and estimate
    per-form runtime before committing to the full run.
    """
    book_entities = [e for e in entities if e.label == "BOOK"]
    n_unique = len({e.text for e in book_entities})
    log.info("Raw BOOK entities: %d  |  Unique surface forms: %d",
             len(book_entities), n_unique)

    # Group occurrences by surface form, collecting task_ids for context
    surface_tasks: dict[str, set[int]] = {}
    for e in book_entities:
        surface_tasks.setdefault(e.text, set()).add(e.task_id)

    all_surfaces = sorted(surface_tasks.items())   # deterministic order

    if smoke_test:
        all_surfaces = all_surfaces[:SMOKE_TEST_LIMIT]
        log.info("── SMOKE TEST: processing first %d surface forms ──",
                 len(all_surfaces))

    resolutions: dict[str, Resolution] = {}
    n_acronym = n_exact = n_norm = n_failed = 0
    t_start   = time.perf_counter()

    # ── Progress bar setup ───────────────────────────────────────────────────
    use_tqdm = tqdm is not None
    if not use_tqdm:
        log.warning("tqdm not installed — install with `pip install tqdm` "
                    "for a progress bar. Falling back to log lines.")

    bar = (
        tqdm(total=len(all_surfaces),
             unit="form",
             desc="Disambiguating",
             dynamic_ncols=True,
             bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                        "[{elapsed}<{remaining}, {rate_fmt}] {postfix}")
        if use_tqdm else None
    )

    for i, (surface, task_ids) in enumerate(all_surfaces, 1):

        # Build union of writer context across all tasks this surface appears in
        context_writers: set[str] = set()
        for tid in task_ids:
            context_writers |= writer_names_in_task(tid, thread_idx)

        # Stage 1: acronym
        res = resolve_via_acronym(surface, acronyms, db)
        if res:
            n_acronym += 1
            resolutions[surface] = res
        else:
            # Stage 2: exact
            res = resolve_via_exact(surface, db, context_writers)
            if res:
                n_exact += 1
                resolutions[surface] = res
            else:
                # Stage 3: normalized
                res = resolve_via_normalized(surface, db, context_writers)
                if res:
                    n_norm += 1
                    resolutions[surface] = res
                else:
                    # Stage 4: failed
                    n_failed += 1
                    resolutions[surface] = Resolution(
                        surface         = surface,
                        method          = "failed",
                        ol_key          = None,
                        canonical_title = None,
                        author_name     = None,
                        confidence      = 0.0,
                        notes           = "no_match",
                    )

        # ── Update progress ──────────────────────────────────────────────────
        if bar is not None:
            bar.update(1)
            if i % PROGRESS_INTERVAL == 0 or i == len(all_surfaces):
                bar.set_postfix(_make_postfix(n_acronym, n_exact,
                                              n_norm, n_failed))
        elif i % PROGRESS_INTERVAL == 0 or i == len(all_surfaces):
            elapsed  = time.perf_counter() - t_start
            rate     = i / elapsed if elapsed > 0 else 0
            remaining = (len(all_surfaces) - i) / rate if rate > 0 else 0
            log.info(
                "[%d/%d] %.0f form/s  ETA ~%.0fs  |  "
                "acr=%d  exact=%d  norm=%d  fail=%d",
                i, len(all_surfaces), rate, remaining,
                n_acronym, n_exact, n_norm, n_failed,
            )

    if bar is not None:
        bar.set_postfix(_make_postfix(n_acronym, n_exact, n_norm, n_failed))
        bar.close()

    elapsed = time.perf_counter() - t_start
    total   = n_acronym + n_exact + n_norm + n_failed

    # ── Final summary ────────────────────────────────────────────────────────
    log.info("─" * 60)
    log.info("Resolution summary (%d unique forms in %.1fs, %.0f form/s)",
             total, elapsed, total / elapsed if elapsed > 0 else 0)
    log.info("  acronym    %5d  (%5.1f%%)", n_acronym, 100 * n_acronym / total if total else 0)
    log.info("  exact      %5d  (%5.1f%%)", n_exact,   100 * n_exact   / total if total else 0)
    log.info("  normalized %5d  (%5.1f%%)", n_norm,    100 * n_norm    / total if total else 0)
    log.info("  failed     %5d  (%5.1f%%)", n_failed,  100 * n_failed  / total if total else 0)
    log.info("─" * 60)

    if smoke_test:
        log.info("Smoke-test complete.  Extrapolated full-run ETA: ~%.0f s  (~%.1f min)",
                 elapsed / total * n_unique if total else 0,
                 elapsed / total * n_unique / 60 if total else 0)

    return resolutions


# ── Output writers ────────────────────────────────────────────────────────────

RESOLVED_FIELDS = [
    "surface", "method", "ol_key", "canonical_title",
    "author_name", "confidence", "ambiguous", "notes",
]

FAILED_FIELDS = ["surface", "notes"]

LINKS_FIELDS = [
    "task_id", "thread_id", "label", "text", "start", "end",
    "method", "ol_key", "canonical_title", "author_name", "confidence", "ambiguous",
]


def write_outputs(resolutions: dict[str, Resolution],
                  entities:    list[RawEntity],
                  out_dir:     Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved  = {s: r for s, r in resolutions.items() if r.method != "failed"}
    failed    = {s: r for s, r in resolutions.items() if r.method == "failed"}

    # resolved_books.csv
    with (out_dir / "resolved_books.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=RESOLVED_FIELDS)
        w.writeheader()
        for r in sorted(resolved.values(), key=lambda x: x.surface.lower()):
            w.writerow({f: getattr(r, f) for f in RESOLVED_FIELDS})
    log.info("Wrote resolved_books.csv (%d entries)", len(resolved))

    # failed_lookup.csv
    with (out_dir / "failed_lookup.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["surface", "notes"])
        w.writeheader()
        for s in sorted(failed.keys()):
            w.writerow({"surface": s, "notes": failed[s].notes})
    log.info("Wrote failed_lookup.csv (%d entries)", len(failed))

    # entity_links.csv — one row per raw occurrence
    book_entities = [e for e in entities if e.label == "BOOK"]
    with (out_dir / "entity_links.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LINKS_FIELDS)
        w.writeheader()
        for e in book_entities:
            r = resolutions.get(e.text)
            w.writerow({
                "task_id":        e.task_id,
                "thread_id":      e.thread_id,
                "label":          e.label,
                "text":           e.text,
                "start":          e.start,
                "end":            e.end,
                "method":         r.method          if r else "",
                "ol_key":         r.ol_key          if r else "",
                "canonical_title":r.canonical_title if r else "",
                "author_name":    r.author_name     if r else "",
                "confidence":     r.confidence      if r else 0.0,
                "ambiguous":      r.ambiguous       if r else False,
            })
    log.info("Wrote entity_links.csv (%d rows)", len(book_entities))


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Disambiguate BOOK entities via Open Library")
    p.add_argument("--dataset",    required=True, help="Label Studio JSON export")
    p.add_argument("--db",         required=True, help="ol.db SQLite database")
    p.add_argument("--acronyms",   required=True, help="acronyms.json")
    p.add_argument("--out-dir",    default="./output", help="Output directory")
    p.add_argument("--smoke-test", action="store_true",
                   help=f"Process only the first {SMOKE_TEST_LIMIT} unique surface forms "
                        "to check for errors and estimate full-run time.")
    return p.parse_args()


def main():
    args = parse_args()

    dataset_path  = Path(args.dataset)
    db_path       = Path(args.db)
    acronyms_path = Path(args.acronyms)
    out_dir       = Path(args.out_dir)

    for p in (dataset_path, db_path, acronyms_path):
        if not p.exists():
            log.error("File not found: %s", p)
            sys.exit(1)

    with acronyms_path.open("r", encoding="utf-8") as fh:
        acronyms = json.load(fh)
    log.info("Loaded %d acronym entries", len(acronyms))

    db = OLDatabase(db_path)

    entities   = load_entities(dataset_path)
    thread_idx = build_thread_index(entities)

    resolutions = disambiguate_books(
        entities, thread_idx, db, acronyms,
        smoke_test=args.smoke_test,
    )

    if args.smoke_test:
        # Print a human-readable table of results instead of writing files
        log.info("Smoke-test results (not writing output files):")
        header = f"{'SURFACE':<35}  {'METHOD':<12}  {'CONF':>5}  {'AMB':>3}  CANONICAL TITLE"
        log.info(header)
        log.info("-" * len(header))
        for surface, r in sorted(resolutions.items()):
            log.info("%-35s  %-12s  %5.2f  %3s  %s",
                     surface[:35],
                     r.method,
                     r.confidence,
                     "Y" if r.ambiguous else "N",
                     r.canonical_title or "—")
    else:
        write_outputs(resolutions, entities, out_dir)

    db.close()
    log.info("Done.")



if __name__ == "__main__":
    main()