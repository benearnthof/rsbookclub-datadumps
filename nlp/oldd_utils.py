#!/usr/bin/env python3
"""
oldd_utils.py
Open Library Data Dump utilities.

Covers:
  - Text normalisation and fuzzy author matching
  - SQLite index builder (from OL dump files)
  - OLClient: unified query interface (local index or OL Search API fallback)
  - API response cache (SQLite-backed)

Run as a script to build the local index:

    python oldd_utils.py \\
        --works-dump   ol_dump_works_2026-02-28.txt.gz \\
        --authors-dump ol_dump_authors_2026-02-28.txt.gz \\
        --index        ol_index.db

The resulting ol_index.db is then passed to extract_pairs.py via --index.
"""

import argparse
import gzip
import json
import logging
import re
import sqlite3
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from rapidfuzz import fuzz as _rfuzz

    def fuzzy_score(a: str, b: str) -> float:
        return _rfuzz.token_set_ratio(a, b)

except ImportError:
    from difflib import SequenceMatcher

    def fuzzy_score(a: str, b: str) -> float:  # type: ignore[misc]
        return SequenceMatcher(None, a, b).ratio() * 100


log = logging.getLogger(__name__)


DEFAULT_WORKS_DUMP   = "ol_dump_works_2026-02-28.txt.gz"
DEFAULT_AUTHORS_DUMP = "ol_dump_authors_2026-02-28.txt.gz"
DEFAULT_INDEX        = "ol_index.db"
DEFAULT_CACHE        = "ol_api_cache.db"

OL_SEARCH_URL        = "https://openlibrary.org/search.json"
OL_AUTHOR_SEARCH     = "https://openlibrary.org/search/authors.json"
OL_API_DELAY         = 0.5      # polite delay between API calls (seconds)

PAIR_THRESHOLD       = 72       # minimum fuzzy score (0–100) to accept a match
MIN_TITLE_LEN        = 3        # skip OL lookups for very short book titles
INDEX_BATCH          = 10_000   # rows per SQLite batch-insert during indexing
MAX_OL_CANDIDATES    = 3        # OL work results to consider per book title


# text normalization
_PUNCT = re.compile(r"[^\w\s]")
_SPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, ASCII-fold, strip punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = _PUNCT.sub(" ", text.lower())
    return _SPACE.sub(" ", text).strip()


def name_contains(query_norm: str, candidate_norm: str) -> bool:
    """
    True if every token in *query* appears somewhere in *candidate*.
    Handles surname-only references: "sebald" ⊆ "w g sebald".
    Allows prefix matching on tokens to handle initials.
    """
    qt = query_norm.split()
    ct = set(candidate_norm.split())
    return all(any(t == c or c.startswith(t) for c in ct) for t in qt)


def writer_matches(raw: str, canonical: str, threshold: float = PAIR_THRESHOLD) -> bool:
    """
    True if *raw* (e.g. a thread mention like "sebald" or "Bolaño") plausibly
    refers to the *canonical* OL author name.
    """
    rn = normalize(raw)
    cn = normalize(canonical)
    if rn == cn:
        return True
    if name_contains(rn, cn) or name_contains(cn, rn):
        return True
    return fuzzy_score(rn, cn) >= threshold


@dataclass
class OLAuthor:
    ol_key:    str
    name:      str
    name_norm: str


@dataclass
class OLWork:
    ol_key:     str
    title:      str
    title_norm: str
    authors:    list[OLAuthor] = field(default_factory=list)


# sqlite stuff
def open_cache(path: str) -> sqlite3.Connection:
    """Open (or create) the API response cache DB."""
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS api_cache (
            cache_key  TEXT PRIMARY KEY,
            payload    TEXT,
            cached_at  REAL
        )""")
    conn.commit()
    return conn


def cache_get(conn: sqlite3.Connection, key: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT payload FROM api_cache WHERE cache_key = ?", (key,)
    ).fetchone()
    return json.loads(row[0]) if row else None


def cache_set(conn: sqlite3.Connection, key: str, data: dict) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO api_cache(cache_key, payload, cached_at) VALUES(?,?,?)",
        (key, json.dumps(data), time.time()),
    )
    conn.commit()


def open_index(path: str) -> sqlite3.Connection:
    """Open a pre-built OL local index in read-only mode."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


# build DB index from data dump files
def build_index(
    works_dump:   str = DEFAULT_WORKS_DUMP,
    authors_dump: str = DEFAULT_AUTHORS_DUMP,
    index_path:   str = DEFAULT_INDEX,
) -> None:
    """
    Stream through OL dump files and build a local SQLite index.

    Dump line format (tab-separated):
        <type>  <key>  <revision>  <last_modified>  <json>

    This is a one-time operation; the resulting DB is used by OLClient
    for all subsequent extract_pairs.py runs.
    """
    log.info("Building OL index → %s", index_path)
    conn = sqlite3.connect(index_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS authors (
            ol_key     TEXT PRIMARY KEY,
            name       TEXT NOT NULL,
            name_norm  TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS works (
            ol_key     TEXT PRIMARY KEY,
            title      TEXT NOT NULL,
            title_norm TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS work_authors (
            work_key   TEXT NOT NULL,
            author_key TEXT NOT NULL,
            PRIMARY KEY (work_key, author_key)
        );
        CREATE INDEX IF NOT EXISTS idx_authors_norm ON authors(name_norm);
        CREATE INDEX IF NOT EXISTS idx_works_norm   ON works(title_norm);
        CREATE INDEX IF NOT EXISTS idx_wa_work      ON work_authors(work_key);
        CREATE INDEX IF NOT EXISTS idx_wa_author    ON work_authors(author_key);
    """)
    conn.commit()

    # writers
    log.info("Indexing authors from %s …", authors_dump)
    author_rows: list[tuple] = []
    n_authors = 0
    opener = gzip.open if authors_dump.endswith(".gz") else open
    with opener(authors_dump, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t", 4)
            if len(parts) < 5:
                continue
            try:
                obj = json.loads(parts[4])
            except json.JSONDecodeError:
                continue
            name = obj.get("name", "").strip()
            key  = parts[1]
            if not name or not key:
                continue
            author_rows.append((key, name, normalize(name)))
            if len(author_rows) >= INDEX_BATCH:
                conn.executemany(
                    "INSERT OR IGNORE INTO authors VALUES(?,?,?)", author_rows
                )
                conn.commit()
                n_authors += len(author_rows)
                author_rows = []
                if n_authors % 500_000 == 0:
                    log.info("  … %d authors indexed", n_authors)
    if author_rows:
        conn.executemany("INSERT OR IGNORE INTO authors VALUES(?,?,?)", author_rows)
        conn.commit()
        n_authors += len(author_rows)
    log.info("Authors indexed: %d", n_authors)

    # works
    log.info("Indexing works from %s …", works_dump)
    work_rows: list[tuple] = []
    wa_rows:   list[tuple] = []
    n_works = 0
    opener = gzip.open if works_dump.endswith(".gz") else open
    with opener(works_dump, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t", 4)
            if len(parts) < 5:
                continue
            try:
                obj = json.loads(parts[4])
            except json.JSONDecodeError:
                continue
            title = obj.get("title", "").strip()
            key   = parts[1]
            if not title or not key:
                continue
            work_rows.append((key, title, normalize(title)))
            for ae in obj.get("authors", []):
                ak = ae.get("author", {}).get("key") if isinstance(ae, dict) else None
                if ak:
                    wa_rows.append((key, ak))
            if len(work_rows) >= INDEX_BATCH:
                conn.executemany("INSERT OR IGNORE INTO works VALUES(?,?,?)", work_rows)
                conn.executemany("INSERT OR IGNORE INTO work_authors VALUES(?,?)", wa_rows)
                conn.commit()
                n_works += len(work_rows)
                work_rows, wa_rows = [], []
                if n_works % 500_000 == 0:
                    log.info("  … %d works indexed", n_works)
    if work_rows:
        conn.executemany("INSERT OR IGNORE INTO works VALUES(?,?,?)", work_rows)
        conn.executemany("INSERT OR IGNORE INTO work_authors VALUES(?,?)", wa_rows)
        conn.commit()
        n_works += len(work_rows)
    log.info("Works indexed: %d", n_works)

    conn.close()
    log.info("Index build complete → %s", index_path)


class OLClient:
    """
    Unified query interface to Open Library.

    Uses a local SQLite index when available (built by build_index()).
    Falls back to the OL Search API with a SQLite-backed response cache.
    """

    def __init__(
        self,
        index_path: Optional[str] = None,
        cache_path: str = DEFAULT_CACHE,
    ):
        self._index: Optional[sqlite3.Connection] = None
        self._cache = open_cache(cache_path)
        self._last_api_call = 0.0

        if index_path and Path(index_path).exists():
            self._index = open_index(index_path)
            log.info("Using local OL index: %s", index_path)
        else:
            log.info("No local index found – using OL Search API (cached at %s)", cache_path)

        session = requests.Session()
        session.mount("https://", HTTPAdapter(max_retries=Retry(
            total=4, backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )))
        session.headers["User-Agent"] = (
            "BookWriterPairExtractor/1.0 (literary EDA research)"
        )
        self._session = session

    # api fallback: Just for good measure, please download the data dumps instead
    # https://openlibrary.org/developers/dumps
    def _api_get(self, url: str, params: dict) -> dict:
        cache_key = url + "?" + "&".join(
            f"{k}={v}" for k, v in sorted(params.items())
        )
        cached = cache_get(self._cache, cache_key)
        if cached is not None:
            return cached
        elapsed = time.time() - self._last_api_call
        if elapsed < OL_API_DELAY:
            time.sleep(OL_API_DELAY - elapsed)
        resp = self._session.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        self._last_api_call = time.time()
        cache_set(self._cache, cache_key, data)
        return data

    # writer validation
    def validate_author(self, raw_name: str) -> Optional[OLAuthor]:
        """
        Return the best OL author match for *raw_name*, or None if not found.
        Used to (a) confirm a WRITER entity is real, (b) get the canonical name.
        """
        qnorm = normalize(raw_name)
        if not qnorm:
            return None
        return self._index_author(qnorm) if self._index else self._api_author(raw_name, qnorm)

    def _index_author(self, qnorm: str) -> Optional[OLAuthor]:
        row = self._index.execute(
            "SELECT ol_key, name, name_norm FROM authors WHERE name_norm = ?",
            (qnorm,),
        ).fetchone()
        if row:
            return OLAuthor(**row)
        tokens = qnorm.split()
        if not tokens:
            return None
        like_clauses = " AND ".join("name_norm LIKE ?" for _ in tokens)
        rows = self._index.execute(
            f"SELECT ol_key, name, name_norm FROM authors WHERE {like_clauses} LIMIT 20",
            [f"%{t}%" for t in tokens],
        ).fetchall()
        if not rows:
            return None
        best = max(rows, key=lambda r: fuzzy_score(qnorm, r["name_norm"]))
        if fuzzy_score(qnorm, best["name_norm"]) >= PAIR_THRESHOLD:
            return OLAuthor(**best)
        return None

    def _api_author(self, raw: str, qnorm: str) -> Optional[OLAuthor]:
        try:
            data = self._api_get(OL_AUTHOR_SEARCH, {"q": raw, "limit": 5})
        except requests.RequestException as exc:
            log.warning("Author API error for %r: %s", raw, exc)
            return None
        for doc in data.get("docs", []):
            name = doc.get("name", "")
            key  = doc.get("key", "")
            if name and key and writer_matches(raw, name):
                return OLAuthor(ol_key=key, name=name, name_norm=normalize(name))
        return None

    # work lookup
    def search_works(self, title: str) -> list[OLWork]:
        """Return up to MAX_OL_CANDIDATES OL works matching *title*."""
        tnorm = normalize(title)
        if len(tnorm) < MIN_TITLE_LEN:
            return []
        return self._index_works(tnorm) if self._index else self._api_works(title, tnorm)

    def _index_works(self, tnorm: str) -> list[OLWork]:
        rows = self._index.execute(
            "SELECT ol_key, title, title_norm FROM works WHERE title_norm = ? LIMIT 5",
            (tnorm,),
        ).fetchall()
        if not rows:
            tokens = tnorm.split()
            if not tokens:
                return []
            like_clauses = " AND ".join("title_norm LIKE ?" for _ in tokens)
            rows = self._index.execute(
                f"SELECT ol_key, title, title_norm FROM works WHERE {like_clauses} LIMIT 20",
                [f"%{t}%" for t in tokens],
            ).fetchall()
        if not rows:
            return []
        ranked = sorted(
            rows,
            key=lambda r: fuzzy_score(tnorm, r["title_norm"]),
            reverse=True,
        )
        results = []
        for row in ranked[:MAX_OL_CANDIDATES]:
            if fuzzy_score(tnorm, row["title_norm"]) < PAIR_THRESHOLD - 10:
                break
            work = OLWork(
                ol_key=row["ol_key"], title=row["title"], title_norm=row["title_norm"]
            )
            work.authors = self._authors_for_work(row["ol_key"])
            results.append(work)
        return results

    def _authors_for_work(self, work_key: str) -> list[OLAuthor]:
        rows = self._index.execute("""
            SELECT a.ol_key, a.name, a.name_norm
            FROM work_authors wa
            JOIN authors a ON a.ol_key = wa.author_key
            WHERE wa.work_key = ?
        """, (work_key,)).fetchall()
        return [OLAuthor(**r) for r in rows]

    def _api_works(self, title: str, tnorm: str) -> list[OLWork]:
        try:
            data = self._api_get(OL_SEARCH_URL, {
                "title": title,
                "fields": "key,title,author_name,author_key",
                "limit": MAX_OL_CANDIDATES,
            })
        except requests.RequestException as exc:
            log.warning("Work API error for %r: %s", title, exc)
            return []
        results = []
        for doc in data.get("docs", [])[:MAX_OL_CANDIDATES]:
            dtitle = doc.get("title", "")
            dkey   = doc.get("key", "")
            if not dtitle or not dkey:
                continue
            if fuzzy_score(tnorm, normalize(dtitle)) < PAIR_THRESHOLD - 10:
                continue
            authors = [
                OLAuthor(
                    ol_key=f"/authors/{ak}" if not ak.startswith("/") else ak,
                    name=an,
                    name_norm=normalize(an),
                )
                for an, ak in zip(
                    doc.get("author_name", []),
                    doc.get("author_key", []),
                )
            ]
            results.append(OLWork(
                ol_key=dkey, title=dtitle,
                title_norm=normalize(dtitle), authors=authors,
            ))
        return results


def _cli() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--works-dump",
        default=DEFAULT_WORKS_DUMP,
        help=f"OL works dump (default: {DEFAULT_WORKS_DUMP})",
    )
    p.add_argument(
        "--authors-dump",
        default=DEFAULT_AUTHORS_DUMP,
        help=f"OL authors dump (default: {DEFAULT_AUTHORS_DUMP})",
    )
    p.add_argument(
        "--index",
        default=DEFAULT_INDEX,
        help=f"Output SQLite index path (default: {DEFAULT_INDEX})",
    )
    args = p.parse_args()
    build_index(args.works_dump, args.authors_dump, args.index)


if __name__ == "__main__":
    _cli()
