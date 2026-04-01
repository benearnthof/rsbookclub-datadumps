"""
query_typesense.py  (async edition)
------------------------------------
Scrapes OL keys from books-search.typesense.org for every surface string
in still_failed.csv, using a pool of concurrent Playwright pages.

Install:
    pip install playwright pandas
    playwright install chromium

Usage:
    python query_typesense.py --csv still_failed.csv --n 20 --workers 8
    python query_typesense.py --csv still_failed.csv            # full run
    python query_typesense.py --csv still_failed.csv --headed   # debug

Tune --workers to your machine / connection. 8-12 is a good starting point;
go higher if CPU/RAM allow and you see no errors.
"""

import argparse
import asyncio
import re
import random
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
from playwright.async_api import async_playwright, TimeoutError as PWTimeout


# ── config ────────────────────────────────────────────────────────────────────

BASE_URL   = "https://books-search.typesense.org/"
OUTPUT_CSV = "typesense_ol_keys.csv"
SKIP_NOTES = {"SCRIPTURE"}
OL_KEY_RE  = re.compile(r"/(OL\d+[MWA])")


# ── per-page worker ───────────────────────────────────────────────────────────

def search_url(surface: str) -> str:
    cleaned = surface.strip().strip('"').strip("'").strip()
    return f"{BASE_URL}?b%5Bquery%5D={quote(cleaned)}"


async def extract_ol_keys(page) -> list[str]:
    keys, seen = [], set()
    for el in await page.locator("a[href*='openlibrary.org']").all():
        href = await el.get_attribute("href") or ""
        m = OL_KEY_RE.search(href)
        if m and (key := m.group(1)) not in seen:
            seen.add(key)
            keys.append(key)
    return keys


async def query_surface(page, surface: str) -> dict:
    result = {"surface": surface, "ol_keys": [], "error": None}
    try:
        await page.goto(search_url(surface), wait_until="domcontentloaded",
                        timeout=20_000)
        try:
            await page.wait_for_selector("a[href*='openlibrary.org']",
                                         timeout=8_000)
        except PWTimeout:
            pass  # genuine no-results — not an error
        result["ol_keys"] = await extract_ol_keys(page)
    except PWTimeout:
        result["error"] = "timeout"
    except Exception as exc:
        result["error"] = str(exc)
    return result


# ── worker pool ───────────────────────────────────────────────────────────────

async def worker(
    worker_id: int,
    queue: asyncio.Queue,
    context,
    write_lock: asyncio.Lock,
    out_path: Path,
    counters: dict,
    total: int,
):
    page = await context.new_page()

    while True:
        try:
            row = queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        surface = row.surface
        notes   = getattr(row, "notes", "")

        result = await query_surface(page, surface)

        # ── crash-safe append ─────────────────────────────────────────
        out_row = pd.DataFrame([{
            "surface": surface,
            "ol_keys": "|".join(result["ol_keys"]),
            "n_hits":  len(result["ol_keys"]),
            "error":   result["error"] or "",
            "notes":   notes,
        }])
        async with write_lock:
            out_row.to_csv(
                out_path,
                mode="a",
                header=not out_path.exists(),
                index=False,
            )
            counters["done"] += 1
            done = counters["done"]

        # ── progress line ─────────────────────────────────────────────
        status = f"{len(result['ol_keys'])} keys" if not result["error"] \
                 else f"ERR:{result['error']}"
        pct = done / total * 100
        print(f"  [{done:>5}/{total}  {pct:4.1f}%]  w{worker_id}  "
              f"{surface[:50]:<50}  {status}")

        queue.task_done()

        # small per-worker jitter so pages don't all fire simultaneously
        await asyncio.sleep(random.uniform(0.3, 0.8))

    await page.close()


# ── main ──────────────────────────────────────────────────────────────────────

async def run(args):
    # ── load CSV ───────────────────────────────────────────────────────
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")

    if "notes" in df.columns:
        before = len(df)
        df = df[~df["notes"].isin(SKIP_NOTES)]
        print(f"Skipped {before - len(df)} SCRIPTURE rows")

    if args.n:
        df = df.head(args.n)

    # ── resume: skip already-done surfaces ────────────────────────────
    out_path = Path(args.output)
    done_surfaces = set()
    if out_path.exists():
        done_df = pd.read_csv(out_path)
        done_surfaces = set(done_df["surface"].tolist())
        print(f"Resuming — {len(done_surfaces)} surfaces already done")

    rows = [row for row in df.itertuples() if row.surface not in done_surfaces]
    if not rows:
        print("Nothing left to do.")
        return

    total = len(rows)
    print(f"Querying {total} surfaces with {args.workers} workers\n")

    # ── fill queue ────────────────────────────────────────────────────
    queue = asyncio.Queue()
    for row in rows:
        await queue.put(row)

    write_lock = asyncio.Lock()
    counters   = {"done": 0}
    t0         = time.monotonic()

    # ── launch browser + worker pool ──────────────────────────────────
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=not args.headed,
            args=["--no-sandbox"],
        )
        context = await browser.new_context(
            viewport={"width": 1280, "height": 900},
            locale="en-US",
        )

        workers = [
            asyncio.create_task(
                worker(i + 1, queue, context, write_lock,
                       out_path, counters, total)
            )
            for i in range(min(args.workers, total))
        ]
        await asyncio.gather(*workers)
        await browser.close()

    elapsed = time.monotonic() - t0
    per_q   = elapsed / total if total else 0

    # ── final summary ─────────────────────────────────────────────────
    final   = pd.read_csv(out_path)
    errors  = (final["error"] != "").sum()
    no_hits = (final["n_hits"] == 0).sum()
    with_keys = (final["n_hits"] > 0).sum()

    print(f"\n── Done in {elapsed:.1f}s  ({per_q:.2f}s/query) {'─'*30}")
    print(f"  with OL keys  : {with_keys}")
    print(f"  no results    : {no_hits}")
    print(f"  errors        : {errors}")
    print(f"  output        : {out_path}")
    if errors:
        err_surfaces = final[final["error"] != ""]["surface"].tolist()
        print(f"\n  Failed surfaces (consider retry):")
        for s in err_surfaces[:10]:
            print(f"    {s}")
        if len(err_surfaces) > 10:
            print(f"    … and {len(err_surfaces) - 10} more")
    print("─" * 50)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",     default="still_failed.csv")
    parser.add_argument("--n",       type=int, default=None,
                        help="Limit to first N rows (omit for full run)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Concurrent browser pages (default: 8)")
    parser.add_argument("--headed",  action="store_true")
    parser.add_argument("--output",  default=OUTPUT_CSV)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

# This is very ugly, we should instead do this locally. 
# TODO: add progress bar with ETA
# TODO: After we obtain the OL keys for every of these we go back to the threads the strings came from.
# Or rather, we run the disambiguation again, with a 5th step, that looks up the OL keys in the 
# generated typesense_ol_keys.csv, queries the database with them, and checks context for any 
# writers present. If there are we found a match, if not we cannot match this with elastic search
# and will thus discard the entity string at last. 
# we could probably perform this asynchronously to massively speed things up. 