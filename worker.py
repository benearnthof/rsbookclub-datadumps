"""
fast, parallel Reddit .zst filter
In a nutshell we use ProcessPoolExecutor to launch one process per file to be decompressed.
We directly call zstd binary via subprocess & pipe everything
source for original filter:
https://github.com/Watchful1/PushshiftDumps/blob/master/scripts/filter_file.py
We fast-reject at byte level to avoid JSON parsing & decode only the relevant lines.
Since we reject more than 99.9% of posts for small subreddits this improves performance massively.
Relevant subreddits can be adjusted via filter variables.
Source-file can be deleted to easily iterate over batches of files on machines that have limited disk space.

Usage:
    (Is demonstrated in monolith_dl.sh)
    python filter_worker.py [source_dir] [output_dir] [--workers N] [--delete-source]
    
Defaults:
    source_dir      ./submissions
    output_dir      ./filtered
    workers         os.cpu_count()
    delete-source   False
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

SUBREDDITS   = {"rsbookclub"}          # lower-cased subreddit names to keep
FILTER_FIELD = "subreddit"             # JSON field to match against
CHUNK_SIZE   = 2 ** 25                 # 32 MB read chunks  (tune to your I/O latency)
MAX_WINDOW   = 2 ** 31                 # zstd fallback decompression window

# Pre-generated filters (byte-level to skip JSON decoding)
_PROBES: frozenset[bytes] = frozenset(s.encode() for s in SUBREDDITS)

# Detect system zstd binary once at import time (safe across forks)
_ZSTD_BIN: str | None = shutil.which("zstd")

log = logging.getLogger("filter")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))

# zstd line iterator, based on code linked at top of file
def _iter_lines_pipe(file_path: Path):
    """
    Decompress via the system `zstd` binary (pure C, fastest path).
    Yields lines as raw bytes, no json decode yet.
    """
    proc = subprocess.Popen(
        [_ZSTD_BIN, "-dc", "--no-progress", str(file_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0,
    )
    buf = b""
    for chunk in iter(lambda: proc.stdout.read(CHUNK_SIZE), b""):
        lines = (buf + chunk).split(b"\n")
        for line in lines[:-1]:
            yield line
        buf = lines[-1]
    if buf:
        yield buf
    proc.wait()
    if proc.returncode not in (0, None):
        raise RuntimeError(f"zstd exited with code {proc.returncode} on {file_path}")


def _iter_lines_zstandard(file_path: Path):
    """
    Fallback: decompress via python-zstandard when the binary is absent. (optional)
    Also yields raw lines as bytes.
    """
    import zstandard 

    with open(file_path, "rb") as fh:
        reader = zstandard.ZstdDecompressor(max_window_size=MAX_WINDOW).stream_reader(fh)
        buf = b""
        while True:
            chunk = reader.read(CHUNK_SIZE)
            if not chunk:
                break
            lines = (buf + chunk).split(b"\n")
            for line in lines[:-1]:
                yield line
            buf = lines[-1]
        if buf:
            yield buf
        reader.close()


def iter_lines_zst(file_path: Path):
    if _ZSTD_BIN:
        yield from _iter_lines_pipe(file_path)
    else:
        log.warning("zstd binary not found — falling back to python-zstandard")
        yield from _iter_lines_zstandard(file_path)

# per file worker
def process_file(
    file_path: Path,
    output_dir: Path,
    subreddits: frozenset[str],
    delete_source: bool,
) -> dict:
    """
    Filter one .zst file and write matching lines as JSONL.
    Returns a summary dict safe to send across process boundaries.
    """
    file_size   = file_path.stat().st_size
    out_path    = output_dir / (file_path.stem + ".jsonl")

    total_lines = 0
    matched     = 0
    bad_lines   = 0
    last_date   = None

    with open(out_path, "w", encoding="utf-8") as out_fh:
        for raw_line in iter_lines_zst(file_path):
            total_lines += 1

            # fast-reject (bytes, no decode, no JSON parse
            # Lower-case the raw bytes once and check for the subreddit name.
            if not any(p in raw_line.lower() for p in _PROBES):
                continue

            # slow-accept (only decode lines that match filters)
            try:
                obj = json.loads(raw_line)
                last_date = datetime.utcfromtimestamp(int(obj["created_utc"]))
                if obj[FILTER_FIELD].lower() in subreddits:
                    out_fh.write(raw_line.decode("utf-8", errors="replace") + "\n")
                    matched += 1
            except (KeyError, ValueError, json.JSONDecodeError):
                bad_lines += 1

    # Clean up empty output files
    if matched == 0:
        out_path.unlink(missing_ok=True)

    if delete_source:
        try:
            file_path.unlink()
        except OSError as exc:
            log.warning(f"Could not delete {file_path}: {exc}")

    return {
        "file":        file_path.name,
        "total_lines": total_lines,
        "matched":     matched,
        "bad_lines":   bad_lines,
        "last_date":   last_date.isoformat() if last_date else "?",
        "out_path":    str(out_path) if matched else None,
    }


# wrapper
def run(
    source_dir: Path,
    output_dir: Path,
    workers: int,
    delete_source: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    zst_files = sorted(source_dir.glob("*.zst"))
    if not zst_files:
        log.error(f"No .zst files found in {source_dir}")
        sys.exit(1)

    log.info(
        f"Found {len(zst_files)} file(s) → dispatching across {workers} worker(s)"
    )

    subreddits = frozenset(SUBREDDITS)
    grand_total = grand_matched = grand_bad = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                process_file, f, output_dir, subreddits, delete_source
            ): f
            for f in zst_files
        }

        with tqdm(total=len(futures), desc="files", unit="file") as pbar:
            for future in as_completed(futures):
                src = futures[future]
                try:
                    r = future.result()
                except Exception as exc:
                    log.error(f"{src.name} failed: {exc}")
                    pbar.update(1)
                    continue

                grand_total   += r["total_lines"]
                grand_matched += r["matched"]
                grand_bad     += r["bad_lines"]

                status = (
                    f"✓ {r['file']}  "
                    f"matched={r['matched']:,}  "
                    f"total={r['total_lines']:,}  "
                    f"bad={r['bad_lines']:,}  "
                    f"last={r['last_date']}"
                )
                tqdm.write(status)
                pbar.update(1)

    log.info(
        f"\nDone.  files={len(zst_files)}  "
        f"total_lines={grand_total:,}  "
        f"matched={grand_matched:,}  "
        f"bad={grand_bad:,}"
    )

# cli
def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel Reddit .zst subreddit filter")
    parser.add_argument("source_dir",  nargs="?", default="./submissions",  type=Path)
    parser.add_argument("output_dir",  nargs="?", default="./filtered",     type=Path)
    parser.add_argument("--workers",   type=int,  default=os.cpu_count(),
                        help="Parallel worker processes (default: all cores)")
    parser.add_argument("--delete-source", action="store_true",
                        help="Delete each .zst after successful processing")
    args = parser.parse_args()

    run(
        source_dir    = args.source_dir,
        output_dir    = args.output_dir,
        workers       = args.workers,
        delete_source = args.delete_source,
    )


if __name__ == "__main__":
    main()
