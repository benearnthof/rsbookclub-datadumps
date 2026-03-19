#!/usr/bin/env python3
"""
peek.py — print the first submission from a target subreddit and exit.
Usage: python3 peek.py <file.zst> [subreddit]
       python3 peek.py /workspace/downloads/reddit/submissions/RS_2025-12.zst pics
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

CHUNK_SIZE = 2 ** 25  # 32 MB


def iter_lines(file_path: Path):
    zstd = shutil.which("zstd")
    if zstd:
        proc = subprocess.Popen(
            [zstd, "-dc", "--no-progress", str(file_path)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0,
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
    else:
        import zstandard
        with open(file_path, "rb") as fh:
            reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(fh)
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


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 peek.py <file.zst> [subreddit]")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    target    = sys.argv[2].lower() if len(sys.argv) > 2 else "pics"
    probe     = target.encode()

    print(f"Scanning {file_path.name} for first r/{target} post...\n")

    checked = 0
    for raw_line in iter_lines(file_path):
        checked += 1
        if probe not in raw_line.lower():
            continue
        try:
            obj = json.loads(raw_line)
            if obj.get("subreddit", "").lower() == target:
                print(f"Found after checking {checked:,} lines\n")
                print(json.dumps(obj, indent=2))
                return
        except (json.JSONDecodeError, KeyError):
            continue

    print(f"No posts from r/{target} found after {checked:,} lines.")


if __name__ == "__main__":
    main()
