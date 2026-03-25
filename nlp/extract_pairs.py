#!/usr/bin/env python3
"""
extract_pairs.py
BOOK-WRITER pair extraction from soft-labeled Reddit entity data.

Reads extractions.jsonl (one thread per line), queries Open Library to
validate writers and confirm book–author relationships, then writes a
consolidated (each pair retains all thread_ids containing them) pairs.jsonl.

Each output record has one of three additional labels:
  PAIR      – writer confirmed by BOTH thread co-occurrence AND Open Library
  OLDD      – writer sourced from Open Library only (no matching writer
               appeared in the same thread as the book)
  UNMATCHED – book entity with no OL record found at all

Every record retains the full list of thread_ids in which the pair was
observed, for rough downstream mention-frequency and trend analysis.

Usage:
With a pre-built local index (fastest, build first via oldd_utils.py):

    python extract_pairs.py \\
        --input  extractions.jsonl \\
        --output pairs.jsonl \\
        --index  ol_index.db

Without a local index (falls back to the OL Search API, results cached):
    Do not do this, please download the full data dumps instead, they are only ~4.5GB total)
    python extract_pairs.py --input extractions.jsonl --output pairs.jsonl
"""

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from oldd_utils import (
    DEFAULT_CACHE,
    DEFAULT_INDEX,
    PAIR_THRESHOLD,
    OLAuthor,
    OLClient,
    fuzzy_score,
    normalize,
    writer_matches,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


@dataclass
class Pair:
    """A single BOOK-WRITER pair extracted from the entity data."""
    book:          str
    writer:        Optional[str]   # canonical OL name (or None for UNMATCHED)
    writer_raw:    Optional[str]   # original thread text (PAIR only)
    source:        str             # "PAIR" | "OLDD" | "UNMATCHED"
    match_score:   float           # fuzzy match score used in pairing
    ol_work_key:   Optional[str]   # e.g. "/works/OL1W"
    ol_author_key: Optional[str]   # e.g. "/authors/OL1A"
    thread_ids:    list[str]       = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "book":          self.book,
            "writer":        self.writer,
            "writer_raw":    self.writer_raw,
            "source":        self.source,
            "match_score":   self.match_score,
            "ol_work_key":   self.ol_work_key,
            "ol_author_key": self.ol_author_key,
            "thread_ids":    self.thread_ids,
            "thread_count":  len(self.thread_ids),
        }


# extract pairs from threads
def pair_thread(
    thread_id:         str,
    books:             list[str],
    writers:           list[str],
    ol:                OLClient,
    validated_writers: dict[str, Optional[OLAuthor]],
) -> list[Pair]:
    """
    Produce PAIR / OLDD / UNMATCHED records for a single thread.

    Parameters
    ----------
    thread_id:
        Reddit thread identifier stored on every output record.
    books:
        BOOK entity texts from the thread.
    writers:
        WRITER entity texts from the thread.
    ol:
        OLClient instance (shared across all threads).
    validated_writers:
        Cross-thread cache: raw writer text -> OLAuthor (or None).
        Updated in-place so each unique writer string is only looked up once.
    """
    # Validate all writers in this thread, populating the shared cache
    for raw in writers:
        if raw not in validated_writers:
            validated_writers[raw] = ol.validate_author(raw)

    # Only keep writers that OL confirmed as real people
    real_writers: dict[str, OLAuthor] = {
        raw: canon
        for raw, canon in validated_writers.items()
        if raw in writers and canon is not None
    }

    pairs: list[Pair] = []

    for book in books:
        ol_works = ol.search_works(book)

        best_pair: Optional[Pair] = None
        best_oldd: Optional[Pair] = None

        for work in ol_works:
            for ol_author in work.authors:

                # try to confirm via thread co-occurrence
                for raw_writer, canon in real_writers.items():
                    score = fuzzy_score(normalize(raw_writer), ol_author.name_norm)
                    # Compare canonical name too (handles abbreviations/surnames)
                    score = max(score, fuzzy_score(canon.name_norm, ol_author.name_norm))

                    if score >= PAIR_THRESHOLD:
                        if best_pair is None or score > best_pair.match_score:
                            best_pair = Pair(
                                book          = book,
                                writer        = canon.name,
                                writer_raw    = raw_writer,
                                source        = "PAIR",
                                match_score   = round(score, 1),
                                ol_work_key   = work.ol_key,
                                ol_author_key = ol_author.ol_key,
                                thread_ids    = [thread_id],
                            )

            # OLDD fallback: first (best-ranked) OL author
            if best_oldd is None and work.authors:
                top = work.authors[0]
                best_oldd = Pair(
                    book          = book,
                    writer        = top.name,
                    writer_raw    = None,
                    source        = "OLDD",
                    match_score   = round(
                        fuzzy_score(normalize(book), work.title_norm), 1
                    ),
                    ol_work_key   = work.ol_key,
                    ol_author_key = top.ol_key,
                    thread_ids    = [thread_id],
                )

        if best_pair:
            pairs.append(best_pair)
        elif best_oldd:
            pairs.append(best_oldd)
        else:
            pairs.append(Pair(
                book          = book,
                writer        = None,
                writer_raw    = None,
                source        = "UNMATCHED",
                match_score   = 0.0,
                ol_work_key   = None,
                ol_author_key = None,
                thread_ids    = [thread_id],
            ))

    return pairs

# consolidation across threads
_SOURCE_RANK = {"PAIR": 0, "OLDD": 1, "UNMATCHED": 2}

def consolidate(all_pairs: list[Pair]) -> list[dict]:
    """
    Merge duplicate (book, writer) pairs across all threads.

    Deduplication key: (normalised book title, normalised writer name).
    Per group:
      - Source priority:  PAIR > OLDD > UNMATCHED
      - thread_ids:       union of every thread that mentioned the pair
      - thread_count:     len(thread_ids) — primary mention-frequency signal
      - match_score:      mean across all observations in the group
    """
    groups: dict[tuple, list[Pair]] = defaultdict(list)
    for p in all_pairs:
        bnorm = normalize(p.book)
        wnorm = normalize(p.writer) if p.writer else "__none__"
        groups[(bnorm, wnorm)].append(p)

    merged: list[dict] = []
    for records in groups.values():
        records.sort(key=lambda r: _SOURCE_RANK.get(r.source, 9))
        best = records[0].to_dict()

        all_tids: list[str] = []
        for r in records:
            all_tids.extend(r.thread_ids)
        best["thread_ids"]   = sorted(set(all_tids))
        best["thread_count"] = len(best["thread_ids"])
        best["match_score"]  = round(
            sum(r.match_score for r in records) / len(records), 1
        )
        merged.append(best)

    merged.sort(key=lambda r: (-r["thread_count"], r["book"]))
    return merged


def run(args: argparse.Namespace) -> None:
    ol = OLClient(index_path=args.index, cache_path=args.cache)

    log.info("Reading %s …", args.input)
    threads: list[dict] = []
    with open(args.input, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                threads.append(json.loads(line))
            except json.JSONDecodeError as exc:
                log.warning("Line %d: skipping bad JSON – %s", lineno, exc)

    log.info("Loaded %d threads", len(threads))

    validated_writers: dict[str, Optional[OLAuthor]] = {}
    all_pairs: list[Pair] = []

    for i, thread in enumerate(threads, 1):
        tid      = thread.get("thread_id", f"thread_{i}")
        entities = thread.get("entities", [])
        books    = [e["text"] for e in entities if e.get("label") == "BOOK"]
        writers  = [e["text"] for e in entities if e.get("label") == "WRITER"]

        if not books:
            continue

        all_pairs.extend(
            pair_thread(tid, books, writers, ol, validated_writers)
        )

        if i % 500 == 0:
            log.info(
                "  %d / %d threads  |  %d raw pairs  |  %d writers validated",
                i, len(threads), len(all_pairs), len(validated_writers),
            )

    log.info("Raw pairs collected:   %d", len(all_pairs))
    final = consolidate(all_pairs)
    log.info("Consolidated pairs:    %d", len(final))

    # breakdown by source
    by_source: dict[str, int] = defaultdict(int)
    for p in final:
        by_source[p["source"]] += 1
    log.info("Breakdown:")
    for src in ("PAIR", "OLDD", "UNMATCHED"):
        log.info("  %-12s %d", src, by_source[src])

    # write to disk
    with open(args.output, "w", encoding="utf-8") as fh:
        for p in final:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    log.info("Output → %s", args.output)

    # print a sample of pairs
    sample = [p for p in final if p["source"] == "PAIR"][:15]
    if sample:
        print("\n Sample pairs")
        for p in sample:
            raw   = p.get("writer_raw") or ""
            canon = p["writer"] or ""
            label = (
                f"{raw!r} → {canon!r}"
                if raw and raw.lower() != canon.lower()
                else repr(canon)
            )
            tids  = ", ".join(p["thread_ids"][:3])
            ellip = " …" if len(p["thread_ids"]) > 3 else ""
            print(
                f"  {p['book']!r:40s}  ✓  {label:35s}"
                f"  n={p['thread_count']}  [{tids}{ellip}]"
            )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--input",
        default="extractions.jsonl",
        help="Input JSONL file (default: extractions.jsonl)",
    )
    p.add_argument(
        "--output",
        default="pairs.jsonl",
        help="Output JSONL file (default: pairs.jsonl)",
    )
    p.add_argument(
        "--index",
        default=None,
        metavar="PATH",
        help=(
            "Local OL SQLite index built by oldd_utils.py "
            "(optional but strongly recommended for 10k+ docs)"
        ),
    )
    p.add_argument(
        "--cache",
        default=DEFAULT_CACHE,
        metavar="PATH",
        help=f"SQLite cache for OL API responses (default: {DEFAULT_CACHE})",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=PAIR_THRESHOLD,
        metavar="SCORE",
        help=f"Fuzzy match threshold 0-100 (default: {PAIR_THRESHOLD})",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.getLogger().setLevel(args.log_level)
    import oldd_utils
    oldd_utils.PAIR_THRESHOLD = args.threshold
    run(args)