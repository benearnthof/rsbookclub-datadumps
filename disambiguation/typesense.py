"""
query_typesense.py
------------------
For each surface string in still_failed.csv, loads the Typesense books
search page and collects every OpenLibrary key from the result links.

Output: typesense_ol_keys.csv  with columns:
    surface, ol_keys (pipe-separated), n_hits, notes

Install:
    pip install playwright pandas
    playwright install chromium

Usage:
    python query_typesense.py --csv still_failed.csv --n 5
    python query_typesense.py --csv still_failed.csv          # full run
    python query_typesense.py --csv still_failed.csv --headed # show browser
"""

import argparse
import re
import time
import random
from urllib.parse import quote
from pathlib import Path
import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout


# ── config ────────────────────────────────────────────────────────────────────

BASE_URL     = "https://books-search.typesense.org/"
OUTPUT_CSV   = "typesense_ol_keys.csv"
SKIP_NOTES   = {"SCRIPTURE"}
DELAY_MIN    = 1.5          # seconds between page loads
DELAY_MAX    = 3.0
OL_KEY_RE    = re.compile(r"/(OL\d+[MWA])")   # matches /OL28368134M  /OL123W etc.


# ── helpers ───────────────────────────────────────────────────────────────────

def search_url(surface: str) -> str:
    """Build the Typesense search URL for a surface string."""
    cleaned = surface.strip().strip('"').strip("'").strip()
    return f"{BASE_URL}?b%5Bquery%5D={quote(cleaned)}"


def extract_ol_keys(page) -> list[str]:
    """Return all unique OL keys found in openlibrary.org links on the page."""
    keys = []
    seen = set()
    for el in page.locator("a[href*='openlibrary.org']").all():
        href = el.get_attribute("href") or ""
        m    = OL_KEY_RE.search(href)
        if m:
            key = m.group(1)
            if key not in seen:
                seen.add(key)
                keys.append(key)
    return keys


def query_surface(page, surface: str) -> dict:
    url    = search_url(surface)
    result = {"surface": surface, "ol_keys": [], "error": None}

    try:
        page.goto(url, wait_until="domcontentloaded", timeout=20_000)
        # wait for at least one OL link — or timeout after 8 s (no results is fine)
        try:
            page.wait_for_selector("a[href*='openlibrary.org']", timeout=8_000)
        except PWTimeout:
            pass   # genuinely no results — not an error

        result["ol_keys"] = extract_ol_keys(page)
        print(f"  ✓  {len(result['ol_keys'])} OL keys")

    except PWTimeout:
        result["error"] = "timeout"
        print("  ✗  timeout")
    except Exception as exc:
        result["error"] = str(exc)
        print(f"  ✗  {exc}")

    return result


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",       default="still_failed.csv")
    parser.add_argument("--n",         type=int,   default=None,
                        help="Limit rows (omit for full run)")
    parser.add_argument("--delay-min", type=float, default=DELAY_MIN)
    parser.add_argument("--delay-max", type=float, default=DELAY_MAX)
    parser.add_argument("--headed",    action="store_true")
    parser.add_argument("--output",    default=OUTPUT_CSV)
    args = parser.parse_args()

    # ── load & filter ──────────────────────────────────────────────────────
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows")

    if "notes" in df.columns:
        before = len(df)
        df = df[~df["notes"].isin(SKIP_NOTES)]
        print(f"Skipped {before - len(df)} SCRIPTURE rows")

    if args.n:
        df = df.head(args.n)

    print(f"Querying {len(df)} surfaces  "
          f"(delay {args.delay_min}–{args.delay_max}s, headed={args.headed})\n")

    # ── resume support: skip already-done surfaces ─────────────────────────
    done = set()
    out_path = Path(args.output)
    if out_path.exists():
        done_df = pd.read_csv(out_path)
        done    = set(done_df["surface"].tolist())
        print(f"Resuming — {len(done)} surfaces already in {args.output}\n")

    rows = [row for row in df.itertuples() if row.surface not in done]
    print(f"{len(rows)} surfaces left to query\n")

    # ── browser ────────────────────────────────────────────────────────────
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=not args.headed,
            args=["--no-sandbox"],
        )
        ctx  = browser.new_context(
            viewport={"width": 1280, "height": 900},
            locale="en-US",
        )
        page = ctx.new_page()

        for i, row in enumerate(rows, start=1):
            surface = row.surface
            notes   = getattr(row, "notes", "")
            print(f"[{i}/{len(rows)}] {surface!r}")

            result = query_surface(page, surface)

            # ── append to CSV immediately (crash-safe) ─────────────────
            out_row = pd.DataFrame([{
                "surface":  surface,
                "ol_keys":  "|".join(result["ol_keys"]),
                "n_hits":   len(result["ol_keys"]),
                "error":    result["error"] or "",
                "notes":    notes,
            }])
            out_row.to_csv(
                args.output,
                mode="a",
                header=not out_path.exists(),
                index=False,
            )
            out_path = Path(args.output)   # ensure exists flag updates

            if i < len(rows):
                time.sleep(random.uniform(args.delay_min, args.delay_max))

        browser.close()

    # ── final summary ──────────────────────────────────────────────────────
    final   = pd.read_csv(args.output)
    ok      = (final["error"] == "").sum()
    no_hits = (final["n_hits"] == 0).sum()
    errors  = (final["error"] != "").sum()
    print(f"\n── Done ────────────────────────────────────────────────────")
    print(f"  total rows written : {len(final)}")
    print(f"  with OL keys       : {ok - no_hits}")
    print(f"  no results (ok)    : {no_hits}")
    print(f"  errors             : {errors}")
    print(f"  output             : {args.output}")
    print(f"────────────────────────────────────────────────────────────")


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