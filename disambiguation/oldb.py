"""
Thin query layer over the ol.db SQLite database.

Provides:
  - exact / normalised title matching
  - FTS prefix / phrase search
  - author lookup by name (exact + FTS)
  - work ↔ author joins
  - acronym helpers

Intended to be imported by the disambiguation pipeline, not run directly.
"""

import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional


@dataclass
class Author:
    ol_key: str
    name: str
    alternate_names: list[str] = field(default_factory=list)


@dataclass
class Work:
    ol_key: str
    title: str
    author_keys: list[str] = field(default_factory=list)
    subjects: list[str] = field(default_factory=list)
    description: Optional[str] = None
    # Populated lazily via join
    authors: list[Author] = field(default_factory=list)


_STRIP_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE    = re.compile(r"\s+")

def normalize(text: str) -> str:
    """
    Lower-case, strip diacritics, remove punctuation, collapse whitespace.
    Used for deterministic matching regardless of surface variation.
    """
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.lower()
    text = _STRIP_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def initials_of(name: str) -> str:
    """
    Return an uppercase initials string for a multi-word name.
    "David Foster Wallace" → "DFW"
    """
    words = name.split()
    return "".join(w[0].upper() for w in words if w)


def title_acronym(title: str) -> str:
    """
    Return uppercase initials of significant words (skip articles / preps).
    "Infinite Jest" → "IJ"
    "The Brothers Karamazov" → "BK"  (leading 'The' stripped)
    """
    STOP = {"a", "an", "the", "of", "in", "on", "at", "and", "or", "but",
            "for", "with", "to", "by"}
    words = title.split()
    sig   = [w for w in words if w.lower() not in STOP]
    return "".join(w[0].upper() for w in sig if w)


class OLDatabase:
    def __init__(self, db_path: str | Path):
        self._path = str(db_path)
        self._con  = sqlite3.connect(self._path, check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        self._con.execute("PRAGMA query_only = ON")
        # Normalised-title cache: norm_title → list[Work]
        self._norm_cache: dict[str, list[Work]] = {}

    def close(self):
        self._con.close()


    def get_author(self, ol_key: str) -> Optional[Author]:
        row = self._con.execute(
            "SELECT ol_key, name, alternate_names FROM authors WHERE ol_key = ?",
            (ol_key,)
        ).fetchone()
        return self._row_to_author(row) if row else None

    def find_authors_exact(self, name: str) -> list[Author]:
        rows = self._con.execute(
            "SELECT ol_key, name, alternate_names FROM authors WHERE name = ?",
            (name,)
        ).fetchall()
        return [self._row_to_author(r) for r in rows]

    def find_authors_normalized(self, name: str) -> list[Author]:
        """
        Match after normalization. Iterates the FTS index for speed,
        then re-filters by normalised equality.
        Suitable for last-name-only or diacritics-variant queries.
        """
        # FTS5 prefix search on the name token(s)
        tokens = normalize(name).split()
        fts_query = " AND ".join(f'"{t}"' for t in tokens)
        rows = self._con.execute(
            """
            SELECT a.ol_key, a.name, a.alternate_names
            FROM   authors_fts af
            JOIN   authors a ON a.rowid = af.rowid
            WHERE  authors_fts MATCH ?
            LIMIT  200
            """,
            (fts_query,)
        ).fetchall()
        norm_target = normalize(name)
        return [
            self._row_to_author(r) for r in rows
            if normalize(r["name"]) == norm_target
        ]

    def search_authors_fts(self, query: str, limit: int = 20) -> list[Author]:
        rows = self._con.execute(
            """
            SELECT a.ol_key, a.name, a.alternate_names
            FROM   authors_fts af
            JOIN   authors a ON a.rowid = af.rowid
            WHERE  authors_fts MATCH ?
            ORDER  BY rank
            LIMIT  ?
            """,
            (query, limit)
        ).fetchall()
        return [self._row_to_author(r) for r in rows]


    def get_work(self, ol_key: str, with_authors: bool = False) -> Optional[Work]:
        row = self._con.execute(
            "SELECT ol_key, title, author_keys, subjects, description "
            "FROM works WHERE ol_key = ?",
            (ol_key,)
        ).fetchone()
        if not row:
            return None
        w = self._row_to_work(row)
        if with_authors:
            w.authors = self.get_authors_for_work(ol_key)
        return w

    def find_works_exact(self, title: str) -> list[Work]:
        rows = self._con.execute(
            "SELECT ol_key, title, author_keys, subjects, description "
            "FROM works WHERE title = ?",
            (title,)
        ).fetchall()
        return [self._row_to_work(r) for r in rows]

    def find_works_normalized(self, title: str) -> list[Work]:
        """Match by normalized title (handles case, punctuation, diacritics)."""
        norm = normalize(title)
        if norm in self._norm_cache:
            return self._norm_cache[norm]

        # Use FTS to get candidates quickly, then filter by norm equality
        tokens = norm.split()
        fts_query = " AND ".join(f'"{t}"' for t in tokens) if tokens else '""'
        rows = self._con.execute(
            """
            SELECT w.ol_key, w.title, w.author_keys, w.subjects, w.description
            FROM   works_fts wf
            JOIN   works w ON w.rowid = wf.rowid
            WHERE  works_fts MATCH ?
            LIMIT  500
            """,
            (fts_query,)
        ).fetchall()
        results = [
            self._row_to_work(r) for r in rows
            if normalize(r["title"]) == norm
        ]
        self._norm_cache[norm] = results
        return results

    def search_works_fts(self, query: str, limit: int = 20) -> list[Work]:
        rows = self._con.execute(
            """
            SELECT w.ol_key, w.title, w.author_keys, w.subjects, w.description
            FROM   works_fts wf
            JOIN   works w ON w.rowid = wf.rowid
            WHERE  works_fts MATCH ?
            ORDER  BY rank
            LIMIT  ?
            """,
            (query, limit)
        ).fetchall()
        return [self._row_to_work(r) for r in rows]


    def get_authors_for_work(self, work_key: str) -> list[Author]:
        rows = self._con.execute(
            """
            SELECT a.ol_key, a.name, a.alternate_names
            FROM   work_authors wa
            JOIN   authors a ON a.ol_key = wa.author_key
            WHERE  wa.work_key = ?
            """,
            (work_key,)
        ).fetchall()
        return [self._row_to_author(r) for r in rows]

    def get_works_for_author(self, author_key: str, limit: int = 50) -> list[Work]:
        rows = self._con.execute(
            """
            SELECT w.ol_key, w.title, w.author_keys, w.subjects, w.description
            FROM   work_authors wa
            JOIN   works w ON w.ol_key = wa.work_key
            WHERE  wa.author_key = ?
            LIMIT  ?
            """,
            (author_key, limit)
        ).fetchall()
        return [self._row_to_work(r) for r in rows]


    def resolve_acronym_work(self, acronym: str, limit: int = 10) -> list[Work]:
        """
        Return works whose title acronym (significant words) matches.
        E.g. acronym='IJ' → 'Infinite Jest'
        """
        acr = acronym.upper()
        # We can't query this purely in SQL without a stored column,
        # so we iterate titles in a targeted way using a LIKE heuristic
        # (first letter of acronym must start the first significant word).
        first_letter = acr[0]
        rows = self._con.execute(
            "SELECT ol_key, title, author_keys, subjects, description "
            "FROM works WHERE title LIKE ? LIMIT 50000",
            (f"{first_letter}%",)
        ).fetchall()
        results = [
            self._row_to_work(r) for r in rows
            if title_acronym(r["title"]) == acr
        ]
        return results[:limit]

    def resolve_acronym_author(self, acronym: str, limit: int = 10) -> list[Author]:
        """
        Return authors whose name initials match.
        E.g. 'DFW' → 'David Foster Wallace'
        """
        acr = acronym.upper()
        first_letter = acr[0]
        rows = self._con.execute(
            "SELECT ol_key, name, alternate_names FROM authors "
            "WHERE name LIKE ? LIMIT 50000",
            (f"{first_letter}%",)
        ).fetchall()
        results = [
            self._row_to_author(r) for r in rows
            if initials_of(r["name"]) == acr
        ]
        return results[:limit]


    @staticmethod
    def _parse_json_list(raw) -> list:
        if not raw:
            return []
        import json
        try:
            val = json.loads(raw)
            return val if isinstance(val, list) else []
        except Exception:
            return []

    def _row_to_author(self, row) -> Author:
        return Author(
            ol_key          = row["ol_key"],
            name            = row["name"] or "",
            alternate_names = self._parse_json_list(row["alternate_names"]),
        )

    def _row_to_work(self, row) -> Work:
        return Work(
            ol_key      = row["ol_key"],
            title       = row["title"] or "",
            author_keys = self._parse_json_list(row["author_keys"]),
            subjects    = self._parse_json_list(row["subjects"]),
            description = row["description"],
        )


if __name__ == "__main__":
    import sys
    db_path = sys.argv[1] if len(sys.argv) > 1 else "ol.db"
    db = OLDatabase(db_path)

    print("Writer:")
    for w in db.find_works_normalized("Paraliterary"):
        if w.author_keys:
            author = db.get_author(w.author_keys[0])
            if not author:
                author = Author("None", "None", ["None"])
        print(f"{w.ol_key}  {w.title!r}  {author.name}")

    # print("\nWorks for Writer:")
    # w = db.get_author("/authors/OL1433006A")
    # print(f"{w!r}")

    db.close()

# What seems to be robust enough is looking for writers via fts
# then selecting whomever has the most published works
# since we're interested in a top 100 list this should be sufficient
# doesnt work for "Fisher". 
# We need thread level information
# Well technically we're only interested in the top 100 books. 
# Any writer information will be available for free.
# Don't need anything fancy, just reliable book disambiguation & lookup.
# if normalized author.name exists in thread we can be pretty sure 
# we found the correct writer.
# We can brute force normalization for the top 100 writer entities
# should be quick just go by popularity and when in doubt 
# do the same we do for books but inverse

# probably need to do it iteratively. 
# -> Exact Map first
# -> Exact DB lookup + Thread Info
# -> Normalized DB Lookup + Thread Info
# -> Check how many couldn't be resolved