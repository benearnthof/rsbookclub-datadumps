#!/usr/bin/env python3
"""
Interactive terminal review of OL book-title candidates.

Deduplicates by ol_key so each unique work is shown exactly once.
Fetches the source task text from Label Studio to show match-in-context.

Controls
--------
  Enter   — accept candidate -> appended to accepted.json
  c       — discard candidate, move to next
  q       — quit (progress is saved; re-running resumes from where you left off)
  b       — back one step  (re-shows the previous candidate)

Usage
-----
  python review_candidates.py candidates.json

Options
-------
  --accepted    PATH   Output file for accepted candidates  (default: accepted.json)
  --context     INT    Characters of context on each side   (default: 20)
  --min-length  INT    Skip candidates shorter than N chars (default: 0)
  --ol-key      KEY    Jump straight to a specific ol_key
"""

import argparse
import json
import os
import sys
from pathlib import Path

from label_studio_sdk import LabelStudio # type: ignore

API_URL    = "http://localhost:8080"
API_KEY    = ""

if sys.platform == "win32":
    import msvcrt

    def getch() -> str:
        """Read a single keypress without waiting for Enter (Windows)."""
        ch = msvcrt.getwch()
        # Arrow keys and other special keys on Windows send a two-byte sequence
        # starting with '\x00' or '\xe0'. Consume the second byte so the caller
        # never receives a partial sequence.
        if ch in ("\x00", "\xe0"):
            msvcrt.getwch()
            return ""
        return ch

else:
    import termios, tty

    def getch() -> str:
        """Read a single keypress without waiting for Enter (Unix/macOS)."""
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return ch


RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[31m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
DIM    = "\033[2m"


def clear_line():
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()


_text_cache: dict[int, str] = {}

def get_task_text(ls: LabelStudio, task_id: int) -> str:
    if task_id not in _text_cache:
        task = ls.tasks.get(id=task_id)
        _text_cache[task_id] = task.data.get("text", "")
    return _text_cache[task_id]


def dedup_by_ol_key(all_candidates: list[dict]) -> list[dict]:
    """
    For each unique ol_key keep the *longest* candidate as the representative
    example (longest gives the most unambiguous context snippet).
    Returns a list sorted by descending candidate length.
    """
    best: dict[str, dict] = {}
    for task_entry in all_candidates:
        task_id   = task_entry["task_id"]
        thread_id = task_entry["thread_id"]
        for cand in task_entry["candidates"]:
            key = cand["ol_key"]
            rec = {**cand, "task_id": task_id, "thread_id": thread_id}
            if key not in best or cand["length"] > best[key]["length"]:
                best[key] = rec
    return sorted(best.values(), key=lambda x: -x["length"])


def load_reviewed(accepted_path: Path) -> set[str]:
    """
    Return the set of ol_keys that have already been decided (accepted or
    explicitly skipped).  We track skipped keys in a sidecar file so a restart
    doesn't re-show discarded candidates.
    """
    seen: set[str] = set()

    if accepted_path.exists():
        with accepted_path.open(encoding="utf-8") as fh:
            try:
                data = json.load(fh)
                for rec in data:
                    seen.add(rec["ol_key"])
            except Exception:
                pass

    skip_path = accepted_path.with_suffix(".skipped.json")
    if skip_path.exists():
        with skip_path.open(encoding="utf-8") as fh:
            try:
                seen.update(json.load(fh))
            except Exception:
                pass

    return seen


def append_accepted(accepted_path: Path, record: dict) -> None:
    data: list[dict] = []
    if accepted_path.exists():
        with accepted_path.open(encoding="utf-8") as fh:
            try:
                data = json.load(fh)
            except Exception:
                pass
    data.append(record)
    with accepted_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def append_skipped(accepted_path: Path, ol_key: str) -> None:
    skip_path = accepted_path.with_suffix(".skipped.json")
    keys: list[str] = []
    if skip_path.exists():
        with skip_path.open(encoding="utf-8") as fh:
            try:
                keys = json.load(fh)
            except Exception:
                pass
    if ol_key not in keys:
        keys.append(ol_key)
    with skip_path.open("w", encoding="utf-8") as fh:
        json.dump(keys, fh, ensure_ascii=False, indent=2)


def rejected_strings_path(accepted_path: Path) -> Path:
    return accepted_path.with_suffix(".rejected_strings.json")


def load_rejected_strings(accepted_path: Path) -> set[str]:
    """Return the set of lowercased entity texts that have been manually rejected."""
    path = rejected_strings_path(accepted_path)
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as fh:
        try:
            return set(json.load(fh))
        except Exception:
            return set()


def append_rejected_string(accepted_path: Path, text: str) -> None:
    """Persist a newly rejected entity string (lowercased) to the sidecar file."""
    path = rejected_strings_path(accepted_path)
    strings: list[str] = []
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            try:
                strings = json.load(fh)
            except Exception:
                pass
    key = text.lower().strip()
    if key not in strings:
        strings.append(key)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(strings, fh, ensure_ascii=False, indent=2)


def render_snippet(text: str, start: int, end: int, ctx: int) -> str:
    """
    Return a coloured snippet:  ...dim context... BOLD+GREEN match ...dim context...
    """
    snip_start = max(0, start - ctx)
    snip_end   = min(len(text), end + ctx)

    before = text[snip_start:start]
    match  = text[start:end]
    after  = text[end:snip_end]

    ellipsis_l = "…" if snip_start > 0   else " "
    ellipsis_r = "…" if snip_end < len(text) else " "

    return (
        f"{DIM}{ellipsis_l}{before}{RESET}"
        f"{BOLD}{GREEN}{match}{RESET}"
        f"{DIM}{after}{ellipsis_r}{RESET}"
    )


def show_candidate(cand: dict, text: str, idx: int, total: int, ctx: int, cascade: int = 0) -> None:
    term_width = os.get_terminal_size().columns

    print("\033[2J\033[H", end="")   # clear screen, cursor home

    progress = f" {idx}/{total} "
    ol_info  = f" {cand['ol_key']} "
    padding  = term_width - len(progress) - len(ol_info)
    print(f"{BOLD}{CYAN}{progress}{RESET}"
          f"{'─' * max(0, padding)}"
          f"{DIM}{ol_info}{RESET}")
    print()

    print(f"  {BOLD}Title  {RESET}: {cand['text']}")
    print(f"  {BOLD}Length {RESET}: {cand['length']} chars")
    print(f"  {BOLD}Task   {RESET}: {cand['task_id']}  (thread {cand['thread_id']})")
    print(f"  {BOLD}Span   {RESET}: [{cand['start']}:{cand['end']}]")
    print()

    snippet = render_snippet(text, cand["start"], cand["end"], ctx)
    print(f"  {snippet}")
    print()

    print("─" * term_width)
    cascade_hint = (f"  {YELLOW}↳ rejecting will also drop {cascade} duplicate(s){RESET}\n"
                    if cascade else "")
    print(f"  {BOLD}[Enter]{RESET} accept   "
          f"{BOLD}[c]{RESET} discard   "
          f"{BOLD}[b]{RESET} back   "
          f"{BOLD}[q]{RESET} quit")
    if cascade_hint:
        print(cascade_hint, end="")
    print("─" * term_width, end="", flush=True)


def review(
    candidates_path: Path,
    accepted_path:   Path,
    ctx:             int,
    min_length:      int,
    start_key:       str | None,
) -> None:

    with candidates_path.open(encoding="utf-8") as fh:
        raw = json.load(fh)

    unique = dedup_by_ol_key(raw)

    # Apply min-length filter
    if min_length > 0:
        unique = [c for c in unique if c["length"] >= min_length]

    # Resume: skip already-decided keys and auto-reject known bad strings
    reviewed         = load_reviewed(accepted_path)
    rejected_strings = load_rejected_strings(accepted_path)

    unreviewed = [c for c in unique if c["ol_key"] not in reviewed]
    auto_rejected = [c for c in unreviewed
                     if c["text"].lower().strip() in rejected_strings]
    queue = sorted(
        (c for c in unreviewed if c["text"].lower().strip() not in rejected_strings),
        key=lambda x: -x["length"],
    )

    # Persist auto-rejected ol_keys so they don't re-appear on future runs
    if auto_rejected:
        for c in auto_rejected:
            append_skipped(accepted_path, c["ol_key"])

    total_unique  = len(unique)
    total_queue   = len(queue)

    print(f"{BOLD}Candidates (unique by ol_key){RESET}: {total_unique:,}")
    print(f"Already reviewed               : {total_unique - total_queue - len(auto_rejected):,}")
    print(f"Auto-rejected (known strings)  : {len(auto_rejected):,}")
    print(f"Remaining                      : {total_queue:,}")

    if not queue:
        print("\nAll candidates reviewed. Nothing to do.")
        return

    # Optional: jump to a specific ol_key
    start_idx = 0
    if start_key:
        for i, c in enumerate(queue):
            if c["ol_key"] == start_key:
                start_idx = i
                break
        else:
            print(f"  {YELLOW}Warning: --ol-key {start_key!r} not found in queue.{RESET}")

    print(f"\nStarting review … (press any key)")
    getch()

    ls = LabelStudio(base_url=API_URL, api_key=API_KEY)

    # Pre-compute how many other queue entries share the same lowercased text,
    # for the longest candidates this is basically useless
    from collections import Counter
    text_counts = Counter(c["text"].lower().strip() for c in queue)

    accepted_count = 0
    skipped_count  = 0
    auto_reject_count = 0
    history: list[int] = []   # stack of queue indices we've visited
    i = start_idx

    while 0 <= i < len(queue):
        cand = queue[i]
        # Fetch context text
        try:
            text = get_task_text(ls, cand["task_id"])
        except Exception as exc:
            print(f"\n{RED}Could not fetch task {cand['task_id']}: {exc}{RESET}")
            text = ""

        cascade = text_counts[cand["text"].lower().strip()] - 1
        show_candidate(cand, text, i + 1, total_queue, ctx, cascade)

        key = getch()

        if key in ("\r", "\n", ""):       # Enter -> accept
            append_accepted(accepted_path, cand)
            accepted_count += 1
            history.append(i)
            i += 1
            # Brief flash confirmation
            clear_line()
            sys.stdout.write(f"  {GREEN}✓ Accepted{RESET}\n")
            sys.stdout.flush()

        elif key in ("c", "C"):           # c -> discard
            append_skipped(accepted_path, cand["ol_key"])
            append_rejected_string(accepted_path, cand["text"])
            # Mark all other queue entries with the same text as skipped too
            cand_text_lower = cand["text"].lower().strip()
            cascaded = 0
            for other in queue[i + 1:]:
                if other["text"].lower().strip() == cand_text_lower:
                    append_skipped(accepted_path, other["ol_key"])
                    cascaded += 1
            if cascaded:
                auto_reject_count += cascaded
                sys.stdout.write(f"  {YELLOW}↳ auto-rejected {cascaded} duplicate string(s){RESET}\n")
                sys.stdout.flush()
            skipped_count += 1
            history.append(i)
            i += 1

        elif key in ("b", "B"):           # b -> back
            if history:
                i = history.pop() # undo
            else:
                pass

        elif key in ("q", "Q", "\x03"):   # q / Ctrl-C -> quit
            break

    print("\033[2J\033[H", end="")   # clear screen
    print(f"\n{BOLD}Session summary{RESET}")
    print(f"  Accepted      : {accepted_count}")
    print(f"  Rejected      : {skipped_count}")
    print(f"  Auto-rejected : {auto_reject_count} (cascade from rejected strings)")
    print(f"  Remaining     : {max(0, len(queue) - i - auto_reject_count)}")
    print(f"\n  Accepted candidates written to: {accepted_path}")
    print(f"  Re-run to continue from where you left off.\n")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("candidates",
                   type=Path,
                   help="candidates.json produced by match_works.py")
    p.add_argument("--accepted",
                   type=Path, default=Path("accepted.json"),
                   help="Output file for accepted candidates (default: accepted.json)")
    p.add_argument("--context",
                   type=int, default=20,
                   help="Characters of context shown on each side (default: 20)")
    p.add_argument("--min-length",
                   type=int, default=0,
                   help="Skip candidates shorter than N characters (default: 0)")
    p.add_argument("--ol-key",
                   default=None,
                   help="Jump straight to a specific ol_key")
    args = p.parse_args()

    review(
        candidates_path=args.candidates,
        accepted_path=args.accepted,
        ctx=args.context,
        min_length=args.min_length,
        start_key=args.ol_key,
    )


if __name__ == "__main__":
    main()