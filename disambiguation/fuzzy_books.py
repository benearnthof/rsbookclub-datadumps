"""
fuzzy_books.py
──────────────
Stage-4 fuzzy resolver for BOOK surface forms that failed the exact /
normalised lookup in disambiguate_books.py.

Pipeline
────────
  4-0  Scripture detection  → label=SCRIPTURE, skip
  4-1  Alias expansion      → hand-curated aliases.json, confidence 1.0
  4-2  Number normalisation → "100 Yrs" → "One Hundred Years", etc.
  4-3  Token-index pre-filter from ol.db (candidate retrieval)
  4-4  RapidFuzz re-ranking (token_sort_ratio + partial_ratio)

The TokenIndex is cached to disk as a gzip-compressed pickle so reruns
skip the build step entirely.  Use --index-cache to set the path
(default: token_index.pkl.gz next to the DB file).

Outputs
───────
  fuzzy_resolved.csv   – matches above threshold
  still_failed.csv     – below threshold or no candidates

Requirements
────────────
  pip install rapidfuzz tqdm

Usage
─────
  python fuzzy_books.py \
      --failed      output/failed_lookup.csv \
      --db          ol.db \
      --aliases     aliases.json \
      --out-dir     ./output \
      --threshold   85 \
      --index-cache token_index.pkl.gz
"""

import argparse
import csv
import gzip
import json
import logging
import pickle
import re
import sqlite3
import sys
from pathlib import Path
from typing import Optional

try:
    from rapidfuzz import fuzz
except ImportError:
    print("ERROR: rapidfuzz not installed.  pip install rapidfuzz")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("ERROR: tqdm not installed.  pip install tqdm")
    sys.exit(1)

from ol_db import OLDatabase, normalize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── 4-0  Scripture detection ──────────────────────────────────────────────────

_SCRIPTURE_RE = re.compile(r"^(\d\s+)?[A-Z][a-z]+\s+\d+:\d+", re.ASCII)
_SCRIPTURE_BOOKS = {
    "genesis","exodus","leviticus","numbers","deuteronomy","joshua","judges",
    "ruth","samuel","kings","chronicles","ezra","nehemiah","esther","job",
    "psalms","psalm","proverbs","ecclesiastes","isaiah","jeremiah","lamentations",
    "ezekiel","daniel","hosea","joel","amos","obadiah","jonah","micah","nahum",
    "habakkuk","zephaniah","haggai","zechariah","malachi","matthew","mark",
    "luke","john","acts","romans","corinthians","galatians","ephesians",
    "philippians","colossians","thessalonians","timothy","titus","philemon",
    "hebrews","james","peter","jude","revelation","quran","surah",
}

def is_scripture(surface: str) -> bool:
    if _SCRIPTURE_RE.match(surface):
        return True
    first = surface.split()[0].rstrip(",.").lower() if surface else ""
    return first in _SCRIPTURE_BOOKS


# ── 4-1  Alias expansion ──────────────────────────────────────────────────────

DEFAULT_ALIASES: dict[str, str] = {
    "1,000 plateaus":                    "A Thousand Plateaus",
    "1000 plateaus":                     "A Thousand Plateaus",
    "1,000 years of non-linear history": "A Thousand Years of Nonlinear History",
    "1000 years of nonlinear history":   "A Thousand Years of Nonlinear History",
    "1,001 nights":                      "One Thousand and One Nights",
    "1001 nights":                       "One Thousand and One Nights",
    "arabian nights":                    "One Thousand and One Nights",
    "100 years in solitude":             "One Hundred Years of Solitude",
    "100 years of solitude":             "One Hundred Years of Solitude",
    "100 yrs of solitude":               "One Hundred Years of Solitude",
    "100 yrs":                           "One Hundred Years of Solitude",
    "100 years of loneliness":           "One Hundred Years of Solitude",
    "100 years of course":               "One Hundred Years of Solitude",
    "1000 cranes":                       "Thousand Cranes",
    "10 of december":                    "Tenth of December",
    "10 days that shook the world":      "Ten Days That Shook the World",
    "100 days of sodom":                 "The 120 Days of Sodom",
    "120 days of sodom":                 "The 120 Days of Sodom",
    "100 years war on palestine":        "The Hundred Years' War on Palestine",
    "100 artists manifestos":            "100 Artists' Manifestos",
    "1001 books to read before you die": "1001 Books You Must Read Before You Die",
    "1001 before you die":               "1001 Books You Must Read Before You Die",
    "1001 books":                        "1001 Books You Must Read Before You Die",
    "10 myths about israel":             "Ten Myths About Israel",
}

def load_aliases(path: Optional[Path]) -> dict[str, str]:
    aliases = dict(DEFAULT_ALIASES)
    if path and path.exists():
        with path.open("r", encoding="utf-8") as fh:
            extra = json.load(fh)
        aliases.update({normalize(k): v for k, v in extra.items()})
        log.info("Loaded %d aliases from %s", len(extra), path.name)
    return {normalize(k): v for k, v in aliases.items()}


# ── 4-2  Number normalisation ─────────────────────────────────────────────────

_NUM_MAP = {
    "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
    "11": "eleven", "12": "twelve", "13": "thirteen", "14": "fourteen",
    "15": "fifteen", "16": "sixteen", "17": "seventeen", "18": "eighteen",
    "19": "nineteen", "20": "twenty", "30": "thirty", "40": "forty",
    "50": "fifty", "60": "sixty", "70": "seventy", "80": "eighty",
    "90": "ninety", "100": "one hundred", "1000": "one thousand",
    "1001": "one thousand and one", "1000000": "one million",
}

_ABBR_MAP = {
    r"\byrs\b": "years",
    r"\byr\b":  "year",
    r"\bvol\b": "volume",
    r"\bst\b":  "saint",
    r"\bdr\b":  "doctor",
    r"\bmr\b":  "mister",
    r"\bno\b":  "number",
    r"\bpt\b":  "part",
    r"\bch\b":  "chapter",
}

def _strip_commas_in_numbers(s: str) -> str:
    return re.sub(r"(\d),(\d{3})", r"\1\2", s)

def number_normalize(surface: str) -> str:
    s      = _strip_commas_in_numbers(surface.lower())
    tokens = [_NUM_MAP.get(t, t) for t in s.split()]
    s      = " ".join(tokens)
    for pat, repl in _ABBR_MAP.items():
        s = re.sub(pat, repl, s, flags=re.IGNORECASE)
    return s.strip()


# ── 4-3  Token-index candidate retrieval ─────────────────────────────────────

_INDEX_STOP = {
    "a","an","the","of","in","on","at","and","or","but","for","with","to",
    "by","from","is","it","its","as","be","was","are","were","been",
    "do","does","did","has","have","had","not","no","so","if","my",
    "his","her","our","their","your","we","he","she","they","i",
}

class TokenIndex:
    """
    Inverted index over work titles for fast candidate retrieval.

    Persistence
    ───────────
    Pass `cache_path` (a .pkl.gz file) to enable disk caching.  On first
    run the index is built from the DB and saved; subsequent runs load it
    in seconds.  The cache is automatically invalidated when the DB file
    is newer than the cache file.
    """

    def __init__(self, db: OLDatabase, min_token_len: int = 4,
                 cache_path: Optional[Path] = None):
        self._idx: dict[str, list[tuple[str, str]]] = {}
        self._min = min_token_len

        if cache_path and self._cache_valid(cache_path, db):
            self._load(cache_path)
        else:
            self._build(db)
            if cache_path:
                self._save(cache_path)

    # ── Cache helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _cache_valid(cache_path: Path, db: OLDatabase) -> bool:
        if not cache_path.exists():
            log.info("No token index cache found — will build from DB.")
            return False
        db_mtime    = Path(db._path).stat().st_mtime
        cache_mtime = cache_path.stat().st_mtime
        if db_mtime > cache_mtime:
            log.info("Token index cache is older than DB — rebuilding.")
            return False
        log.info("Token index cache is up to date: %s", cache_path)
        return True

    def _load(self, cache_path: Path):
        log.info("Loading token index from cache: %s", cache_path)
        with gzip.open(cache_path, "rb") as fh:
            self._idx = pickle.load(fh)
        log.info("Token index loaded: %d tokens", len(self._idx))

    def _save(self, cache_path: Path):
        log.info("Saving token index to cache: %s", cache_path)
        with gzip.open(cache_path, "wb", compresslevel=3) as fh:
            pickle.dump(self._idx, fh, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = cache_path.stat().st_size / 1024 / 1024
        log.info("Token index cached -> %s (%.1f MB)", cache_path, size_mb)

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build(self, db: OLDatabase):
        log.info("Building token index from DB ...")
        con: sqlite3.Connection = db._con
        total = con.execute(
            "SELECT COUNT(*) FROM works WHERE title IS NOT NULL"
        ).fetchone()[0]
        cur = con.execute("SELECT ol_key, title FROM works WHERE title IS NOT NULL")
        n   = 0
        with tqdm(cur, total=total, desc="Building token index",
                  unit=" works", unit_scale=True, dynamic_ncols=True) as bar:
            for row in bar:
                ol_key, title = row[0], row[1]
                for tok in self._tokenize(title):
                    self._idx.setdefault(tok, []).append((ol_key, title))
                n += 1
        log.info("Token index built: %d works, %d tokens", n, len(self._idx))

    # ── Query ─────────────────────────────────────────────────────────────────

    def _tokenize(self, title: str) -> set[str]:
        tokens = normalize(title).split()
        return {t for t in tokens if t not in _INDEX_STOP and len(t) >= self._min}

    def candidates(self, surface: str,
                   max_per_token: int = 300) -> list[tuple[str, str]]:
        tokens = self._tokenize(surface)
        seen:   set[str]              = set()
        result: list[tuple[str, str]] = []
        for tok in tokens:
            for item in self._idx.get(tok, [])[:max_per_token]:
                if item[0] not in seen:
                    seen.add(item[0])
                    result.append(item)
        return result


class FTSFallback:
    """Use the FTS5 index already in the DB — lower RAM, slower per query."""
    def __init__(self, db: OLDatabase):
        self._db = db

    def candidates(self, surface: str, limit: int = 200) -> list[tuple[str, str]]:
        works = self._db.search_works_fts(surface, limit=limit)
        return [(w.ol_key, w.title) for w in works]


# ── 4-4  RapidFuzz re-ranking ─────────────────────────────────────────────────

def fuzzy_score(query: str, candidate_title: str) -> float:
    """60% token_sort_ratio + 40% partial_ratio, both 0-100."""
    q   = normalize(query)
    c   = normalize(candidate_title)
    tsr = fuzz.token_sort_ratio(q, c)
    pr  = fuzz.partial_ratio(q, c)
    return 0.60 * tsr + 0.40 * pr


def resolve_fuzzy(surface: str,
                  candidates: list[tuple[str, str]],
                  db: OLDatabase,
                  threshold: float) -> Optional[dict]:
    if not candidates:
        return None

    scored = [(ol_key, title, fuzzy_score(surface, title))
              for ol_key, title in candidates]
    scored.sort(key=lambda x: x[2], reverse=True)

    best_key, best_title, best_score = scored[0]
    if best_score < threshold:
        return None

    second_score = scored[1][2] if len(scored) > 1 else 0.0
    ambiguous    = (best_score - second_score) < 5.0 and best_score < 90.0

    work        = db.get_work(best_key)
    author_name = None
    if work and work.author_keys:
        author = db.get_author(work.author_keys[0])
        if author:
            author_name = author.name

    return {
        "ol_key":          best_key,
        "canonical_title": best_title,
        "author_name":     author_name,
        "fuzzy_score":     round(best_score, 1),
        "ambiguous":       ambiguous,
        "n_candidates":    len(candidates),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

RESOLVED_FIELDS = [
    "surface", "method", "ol_key", "canonical_title",
    "author_name", "fuzzy_score", "ambiguous", "notes",
]
FAILED_FIELDS = ["surface", "notes"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--failed",      required=True,  help="failed_lookup.csv from stage 3")
    p.add_argument("--db",          required=True,  help="ol.db SQLite database")
    p.add_argument("--aliases",     default=None,   help="aliases.json (optional)")
    p.add_argument("--out-dir",     default="./output")
    p.add_argument("--threshold",   type=float, default=85.0,
                   help="Minimum RapidFuzz score to accept a match (default: 85)")
    p.add_argument("--index-cache", default=None,
                   help="Path to gzip-pickle cache for the token index "
                        "(e.g. token_index.pkl.gz).  Created on first run, "
                        "reused on subsequent runs.  Invalidated automatically "
                        "if the DB is newer than the cache.")
    p.add_argument("--use-fts",     action="store_true",
                   help="Use FTS5 fallback instead of token index (low RAM, slower)")
    return p.parse_args()


def main():
    args         = parse_args()
    failed_path  = Path(args.failed)
    db_path      = Path(args.db)
    aliases_path = Path(args.aliases) if args.aliases else None
    out_dir      = Path(args.out_dir)
    threshold    = args.threshold
    cache_path   = Path(args.index_cache) if args.index_cache else None

    for p in (failed_path, db_path):
        if not p.exists():
            log.error("Not found: %s", p)
            sys.exit(1)

    with failed_path.open("r", encoding="utf-8") as fh:
        surfaces = [row["surface"] for row in csv.DictReader(fh)]
    log.info("Loaded %d failed surfaces", len(surfaces))

    aliases = load_aliases(aliases_path)
    db      = OLDatabase(db_path)

    if args.use_fts:
        log.info("Using FTS5 fallback (lower RAM, slower)")
        retriever = FTSFallback(db)
    else:
        retriever = TokenIndex(db, cache_path=cache_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_rows: list[dict] = []
    failed_rows:   list[dict] = []
    n_scripture = n_alias = n_fuzzy = n_failed = 0

    for surface in tqdm(surfaces, desc="Resolving surfaces",
                        unit=" forms", dynamic_ncols=True):

        # 4-0  Scripture
        if is_scripture(surface):
            n_scripture += 1
            failed_rows.append({"surface": surface, "notes": "SCRIPTURE"})
            continue

        # 4-1  Alias
        norm = normalize(surface)
        if norm in aliases:
            canonical = aliases[norm]
            works     = db.find_works_normalized(canonical)
            ol_key    = works[0].ol_key if works else None
            author    = None
            if works and works[0].author_keys:
                a = db.get_author(works[0].author_keys[0])
                if a:
                    author = a.name
            resolved_rows.append({
                "surface":         surface,
                "method":          "alias",
                "ol_key":          ol_key or "",
                "canonical_title": canonical,
                "author_name":     author or "",
                "fuzzy_score":     100.0,
                "ambiguous":       False,
                "notes":           f"alias->'{canonical}'",
            })
            n_alias += 1
            continue

        # 4-2 + 4-3 + 4-4  Normalise -> retrieve -> fuzzy rank
        normalised_surface = number_normalize(surface)
        candidates         = retriever.candidates(normalised_surface)
        if normalised_surface != normalize(surface):
            candidates += retriever.candidates(surface)

        match = resolve_fuzzy(normalised_surface, candidates, db, threshold)
        if match:
            n_fuzzy += 1
            resolved_rows.append({
                "surface":         surface,
                "method":          "fuzzy",
                "ol_key":          match["ol_key"],
                "canonical_title": match["canonical_title"],
                "author_name":     match["author_name"] or "",
                "fuzzy_score":     match["fuzzy_score"],
                "ambiguous":       match["ambiguous"],
                "notes":           f"score={match['fuzzy_score']};n_cands={match['n_candidates']}",
            })
        else:
            n_failed += 1
            failed_rows.append({
                "surface": surface,
                "notes":   f"below_threshold({threshold})" if candidates else "no_candidates",
            })

    log.info(
        "Stage-4 summary — scripture: %d | alias: %d | fuzzy: %d | still_failed: %d",
        n_scripture, n_alias, n_fuzzy, n_failed,
    )

    with (out_dir / "fuzzy_resolved.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=RESOLVED_FIELDS)
        w.writeheader()
        w.writerows(resolved_rows)
    log.info("Wrote fuzzy_resolved.csv (%d rows)", len(resolved_rows))

    with (out_dir / "still_failed.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FAILED_FIELDS)
        w.writeheader()
        w.writerows(failed_rows)
    log.info("Wrote still_failed.csv (%d rows)", len(failed_rows))


if __name__ == "__main__":
    main()