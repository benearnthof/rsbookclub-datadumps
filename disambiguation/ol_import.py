"""
Parse Open Library TSV dumps into SQLite.
# NOTE: Claude version of ingest_oldd.py
Usage:
    python ol_import.py \
        --authors ol_dump_authors_2026-02-28.txt \
        --works   ol_dump_works_2026-02-28.txt \
        --db      ol.db

Each dump line is tab-separated:
    col0: /type/…
    col1: key  (/authors/OL…A  or  /works/OL…W)
    col2: revision (int)
    col3: last_modified (ISO datetime)
    col4: JSON payload

Schema
authors (ol_key TEXT PK, name TEXT, alternate_names TEXT)
works   (ol_key TEXT PK, title TEXT, author_keys TEXT,
         subjects TEXT, description TEXT)
work_authors  (work_key TEXT, author_key TEXT)   -- junction, indexed

FTS virtual tables are created over authors.name and works.title
so string matching during disambiguation is fast.
"""

import argparse
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

WORK_BATCH = 50_000
AUTHOR_BATCH = 50_000

DDL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;
PRAGMA cache_size   = -65536;   -- 64 MB page cache
PRAGMA temp_store   = MEMORY;

CREATE TABLE IF NOT EXISTS authors (
    ol_key          TEXT PRIMARY KEY,
    name            TEXT,
    alternate_names TEXT    -- JSON array stored as text
);

CREATE TABLE IF NOT EXISTS works (
    ol_key      TEXT PRIMARY KEY,
    title       TEXT,
    author_keys TEXT,       -- JSON array of /authors/OL…A strings
    subjects    TEXT,       -- JSON array stored as text
    description TEXT
);

-- Normalised junction table for fast author→works / work→authors lookups
CREATE TABLE IF NOT EXISTS work_authors (
    work_key   TEXT NOT NULL REFERENCES works(ol_key),
    author_key TEXT NOT NULL REFERENCES authors(ol_key),
    PRIMARY KEY (work_key, author_key)
);

CREATE INDEX IF NOT EXISTS idx_wa_author ON work_authors(author_key);
CREATE INDEX IF NOT EXISTS idx_wa_work   ON work_authors(work_key);

-- FTS5 for fuzzy/prefix title search
CREATE VIRTUAL TABLE IF NOT EXISTS works_fts USING fts5(
    title,
    content='works',
    content_rowid='rowid',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE VIRTUAL TABLE IF NOT EXISTS authors_fts USING fts5(
    name,
    content='authors',
    content_rowid='rowid',
    tokenize='unicode61 remove_diacritics 2'
);
"""

def _description_text(raw) -> str | None:
    """OL description can be a plain string or {type, value} dict."""
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        return raw.get("value")
    return None


def iter_dump(path: Path, expected_type: str):
    """Yield parsed JSON payloads from an OL dump file."""
    skipped = 0
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 4)
            if len(parts) < 5:
                skipped += 1
                continue
            type_col, key, _rev, _ts, payload = parts
            if type_col != expected_type:
                skipped += 1
                continue
            try:
                yield key, json.loads(payload)
            except json.JSONDecodeError:
                skipped += 1
            if lineno % 500_000 == 0:
                log.info("  … %d lines parsed", lineno)
    if skipped:
        log.warning("  Skipped %d malformed/wrong-type lines in %s", skipped, path.name)


def import_authors(db: sqlite3.Connection, path: Path) -> int:
    log.info("Importing authors from %s", path)
    t0 = time.perf_counter()
    batch, total = [], 0

    cur = db.cursor()
    for key, obj in iter_dump(path, "/type/author"):
        name  = obj.get("name") or ""
        alts  = obj.get("alternate_names")
        alts_json = json.dumps(alts, ensure_ascii=False) if alts else None

        batch.append((key, name, alts_json))
        if len(batch) >= AUTHOR_BATCH:
            cur.executemany(
                "INSERT OR REPLACE INTO authors VALUES (?,?,?)", batch
            )
            total += len(batch)
            batch.clear()
            log.info("  authors committed: %d", total)

    if batch:
        cur.executemany("INSERT OR REPLACE INTO authors VALUES (?,?,?)", batch)
        total += len(batch)

    db.commit()
    log.info("Authors done – %d rows in %.1fs", total, time.perf_counter() - t0)
    return total


def import_works(db: sqlite3.Connection, path: Path) -> int:
    log.info("Importing works from %s", path)
    t0 = time.perf_counter()
    w_batch, wa_batch, total = [], [], 0

    cur = db.cursor()
    for key, obj in iter_dump(path, "/type/work"):
        title = obj.get("title") or ""

        # Extract author keys
        raw_authors = obj.get("authors") or []
        author_keys = []
        for a in raw_authors:
            if isinstance(a, dict):
                author_ref = a.get("author") or {}
                ak = author_ref.get("key") if isinstance(author_ref, dict) else None
                if ak:
                    author_keys.append(ak)

        subjects    = obj.get("subjects")
        description = _description_text(obj.get("description"))

        w_batch.append((
            key,
            title,
            json.dumps(author_keys, ensure_ascii=False) if author_keys else None,
            json.dumps(subjects,    ensure_ascii=False) if subjects    else None,
            description,
        ))

        for ak in author_keys:
            wa_batch.append((key, ak))

        if len(w_batch) >= WORK_BATCH:
            cur.executemany("INSERT OR REPLACE INTO works VALUES (?,?,?,?,?)", w_batch)
            cur.executemany(
                "INSERT OR IGNORE INTO work_authors VALUES (?,?)", wa_batch
            )
            total += len(w_batch)
            w_batch.clear()
            wa_batch.clear()
            log.info("  works committed: %d", total)

    if w_batch:
        cur.executemany("INSERT OR REPLACE INTO works VALUES (?,?,?,?,?)", w_batch)
        cur.executemany("INSERT OR IGNORE INTO work_authors VALUES (?,?)", wa_batch)
        total += len(w_batch)

    db.commit()
    log.info("Works done – %d rows in %.1fs", total, time.perf_counter() - t0)
    return total


def build_fts(db: sqlite3.Connection):
    """Populate FTS tables from the base tables."""
    log.info("Building FTS indexes …")
    t0 = time.perf_counter()
    db.execute("INSERT INTO works_fts(works_fts)   VALUES('rebuild')")
    db.execute("INSERT INTO authors_fts(authors_fts) VALUES('rebuild')")
    db.commit()
    log.info("FTS done in %.1fs", time.perf_counter() - t0)


def build_extra_indexes(db: sqlite3.Connection):
    """Add covering indexes useful for disambiguation queries."""
    log.info("Building extra indexes …")
    db.execute("CREATE INDEX IF NOT EXISTS idx_works_title   ON works(title)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_authors_name  ON authors(name)")
    db.commit()
    log.info("Extra indexes done.")


def parse_args():
    p = argparse.ArgumentParser(description="Import Open Library dumps into SQLite")
    p.add_argument("--authors", required=True, help="Path to ol_dump_authors_*.txt")
    p.add_argument("--works",   required=True, help="Path to ol_dump_works_*.txt")
    p.add_argument("--db",      default="ol.db", help="Output SQLite file (default: ol.db)")
    p.add_argument("--skip-fts", action="store_true",
                   help="Skip FTS index build (faster import, no full-text search)")
    return p.parse_args()


def main():
    args = parse_args()
    authors_path = Path(args.authors)
    works_path   = Path(args.works)
    db_path      = Path(args.db)

    for p in (authors_path, works_path):
        if not p.exists():
            log.error("File not found: %s", p)
            sys.exit(1)

    log.info("Opening database: %s", db_path)
    db = sqlite3.connect(db_path)
    db.executescript(DDL)

    import_authors(db, authors_path)
    import_works(db, works_path)
    build_extra_indexes(db)

    if not args.skip_fts:
        build_fts(db)

    # Quick sanity check
    n_authors = db.execute("SELECT COUNT(*) FROM authors").fetchone()[0]
    n_works   = db.execute("SELECT COUNT(*) FROM works").fetchone()[0]
    n_wa      = db.execute("SELECT COUNT(*) FROM work_authors").fetchone()[0]
    log.info("─" * 55)
    log.info("authors:      %10d", n_authors)
    log.info("works:        %10d", n_works)
    log.info("work_authors: %10d", n_wa)
    log.info("Database ready → %s", db_path.resolve())

    db.close()


if __name__ == "__main__":
    main()