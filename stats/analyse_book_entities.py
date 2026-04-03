"""
analyse_book_entities.py

Exploratory analysis of book_entity_links.csv.
Produces:
  - Console summary stats
  - output/top100_raw.csv          — top 100 by raw mention count
  - output/top100_thread_norm.csv  — top 100 by unique thread appearances
  - output/top100_comparison.csv   — both rankings side by side
"""

import pandas as pd # type: ignore
from pathlib import Path

# output2 contains fully disambiguated data
Path("output2").mkdir(exist_ok=True)

df = pd.read_csv("output2/entity_links.csv", low_memory=False)

# input -> Target, we keep writer of target.
# we just swap these in entity_links directly, before aggregating statistics.
book_map = {
    "À rebours": "Against Nature",
    "Absalom": "Absalom, Absalom!",
    "Anéantir":  "Annihilation",
    "Bartleby": "Bartleby the Scrivener",
    "Beautiful World": "Beautiful World, Where Are You",
    "Berlin": "Berlin Alexanderplatz",
    "Capital": "Das Kapital",
    "Capital Vol-1": "Das Kapital",
    "Crime & Punishment": "Crime and Punishment",
    "Dorian Gray": "The Picture of Dorian Gray",
    "Ecclesiastes": "The Bible",
    "Exodus": "The Bible",
    "Genesis": "The Bible",
    "Bible": "The Bible",
    "Finnegans Wake": "Finnegan's wake",
    "Franny y Zooey": "Franny and Zooey",
    "Gatsby": "The Great Gatsby", 
    "Goon Squad": "A Visit from the Goon Squad", 
    "Huck Finn": "Huckleberry Finn", 
    "If on a Winter's Night a Traveller": "If on a Winter's Night a Traveler",
    "Junkie": "Junky",
    "Lot 49": "The Crying of Lot 49", 
    "Mason and Dixon's line": "Mason & Dixon", 
    "Metamorphoses": "Metamorphosis",
    "Notes from Underground": "Notes from the Underground",
    "On the shore": "Kafka on the Shore",
    "Persepolis": "Persepolis: The Story of a Childhood",
    "Pride & Prejudice": "Pride and Prejudice",
    "Siddartha": "Siddhartha",
    "Swanns Way": "Swann's Way",
    "Tbk": "The Brothers Karamazov",
    "Thousand Plateaus, a": "A Thousand Plateaus",
    "Tre cantos": "The Cantos",
    "War & Peace": "War and Peace",
    "Zarathustra": "Thus Spoke Zarathustra",
    "Time Shards": "The Shards",
    "100 years of solitude": "One Hundred Years of Solitude",
    "Range Detectives": "The Savage Detectives",
    "Story of the 57": "Story of the Eye",
    "The White Rail": "The White Album",
}

df["canonical_title"] = df["canonical_title"].replace(book_map)

print(f"Loaded {len(df):,} rows, {df['thread_id'].nunique():,} unique threads.\n")
print(f"  Total BOOK mentions          : {len(df):,}")
print(f"  Unique threads w/ Books      : {df['thread_id'].nunique():,}")
print(f"  Unique ol_keys (works)       : {df['ol_key'].nunique():,}")
print(f"  Unique canonical titles      : {df['canonical_title'].nunique():,}")

resolved   = df[df["method"] != "failed"]
unresolved = df[df["method"] == "failed"]

res = resolved[
    resolved["ol_key"].notna() & (resolved["ol_key"] != "") |
    resolved["canonical_title"].notna() & (resolved["canonical_title"] != "")
].copy()


top_raw = (
    res.groupby(["canonical_title", "author_name"], dropna=False)
    .agg(
        ol_key          = ("ol_key", lambda x: x[x != ""].iloc[0] if (x != "").any() else ""),
        mention_count   = ("text", "count"),
        thread_count    = ("thread_id", "nunique"),
        avg_confidence  = ("confidence", "mean"),
        ambiguous_pct   = ("ambiguous", lambda x: round(100 * x.eq(True).sum() / len(x), 1)),
    )
    .reset_index()
    .sort_values("mention_count", ascending=False)
    .reset_index(drop=True)
)[["canonical_title", "author_name", "mention_count", "thread_count"]]

top_raw[top_raw["author_name"] == "Thomas Pynchon"]
top_raw[top_raw["canonical_title"] == "V."]

# top_raw.head(200).to_csv("top200_raw.tsv", sep="\t", index=False)

# Now for a hard cut on the "cleaned" data. 
# We know that disambiguation yielded a lot of new book titles, unfortunately
# the writer attribution via open library was rather stupid.
# so we just look at the top 1000 books by total mention count 
# and then map them to the correct authors.

top_raw.tail(5)
top_1k = top_raw.head(1200)

import numpy as np # type: ignore
import re
import unicodedata
import pandas as pd # type: ignore

LEADING_ARTICLES = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)
PUNCTUATION      = re.compile(r"[^\w\s]")
WHITESPACE       = re.compile(r"\s+")

def normalize(title: str) -> str:
    """Lowercase, NFD-normalize, strip leading articles and punctuation."""
    t = unicodedata.normalize("NFD", title)
    t = t.encode("ascii", "ignore").decode("ascii")   # drop accents
    t = t.lower()
    t = LEADING_ARTICLES.sub("", t)
    t = PUNCTUATION.sub(" ", t)
    t = WHITESPACE.sub(" ", t).strip()
    return t
 
 
def best_author(group: pd.DataFrame) -> str:
    """Filler, we will have to set writers manually anyway."""
    named = group.dropna(subset=["author_name"])
    if named.empty:
        return np.nan
    return named.groupby("author_name")["thread_count"].sum().idxmax()
 
 
def merge_group(group: pd.DataFrame) -> dict:
    """Collapse a group to a single representative row."""
    return {
        "canonical_title": group["canonical_title"].iloc[0],  # first = representative
        "author_name":     best_author(group),
        "mention_count":   group["mention_count"].sum(),
        "thread_count":    group["thread_count"].sum(),
        "n_merged":        len(group),
    }


def group_exact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group rows whose normalized titles are identical.
    Within each group the canonical_title chosen is the one that appeared
    with the highest thread_count (most prominent form).
    """
    df = df.copy()
    df["_norm"] = df["canonical_title"].map(normalize)
    # pick the representative title = row with highest thread_count per norm key
    title_rep = (
        df.sort_values("thread_count", ascending=False)
          .drop_duplicates(subset="_norm")
          .set_index("_norm")["canonical_title"]
    )
    df["canonical_title"] = df["_norm"].map(title_rep)
    rows = []
    for _, group in df.groupby("_norm"):
        row = merge_group(group)
        rows.append(row)
    result = pd.DataFrame(rows).sort_values("thread_count", ascending=False)
    result = result.drop(columns=["n_merged"], errors="ignore")
    return result.reset_index(drop=True)

exact_grouped = group_exact(top_1k) 
exact_grouped.to_csv("top1000_raw.csv", sep=",", index=False)





temp = top_raw.sort_values(by="canonical_title", key=lambda x: x.str.len()).reset_index(drop=True)






































top_raw.index += 1
top_raw.index.name = "rank_raw"

top_raw.to_csv("output/top100_raw.csv")

print("\n" + "═" * 55)
print("TOP 20 — RAW MENTION COUNT")
print("═" * 55)
print(
    top_raw.head(20)[["canonical_title", "author_name", "mention_count", "thread_count"]]
    .to_string()
)

# ── Top 100 — thread-normalised (unique threads) ──────────────────────────────

top_threads = (
    res.groupby(["ol_key", "canonical_title", "author_name"])
    .agg(
        thread_count    = ("thread_id", "nunique"),
        mention_count   = ("text", "count"),
        avg_confidence  = ("confidence", "mean"),
        ambiguous_pct   = ("ambiguous", lambda x: round(100 * x.eq(True).sum() / len(x), 1)),
    )
    .reset_index()
    .sort_values("thread_count", ascending=False)
    .head(100)
    .reset_index(drop=True)
)
top_threads.index += 1
top_threads.index.name = "rank_threads"

top_threads.to_csv("output/top100_thread_norm.csv")

print("\n" + "═" * 55)
print("TOP 20 — UNIQUE THREAD COUNT (NORMALISED)")
print("═" * 55)
print(
    top_threads.head(20)[["canonical_title", "author_name", "thread_count", "mention_count"]]
    .to_string()
)

# ── Comparison: rank shift between the two approaches ────────────────────────

raw_ranks    = top_raw.reset_index()[["rank_raw",     "ol_key", "canonical_title", "author_name", "mention_count", "thread_count"]]
thread_ranks = top_threads.reset_index()[["rank_threads", "ol_key", "thread_count", "mention_count"]]
thread_ranks = thread_ranks.rename(columns={"thread_count": "thread_count_t", "mention_count": "mention_count_t"})

# Union of both top-100s
all_keys = pd.concat([
    top_raw[["ol_key", "canonical_title", "author_name"]],
    top_threads[["ol_key", "canonical_title", "author_name"]],
]).drop_duplicates("ol_key")

cmp = all_keys.merge(raw_ranks,    on="ol_key", how="left", suffixes=("", "_r"))
cmp = cmp.merge(thread_ranks,      on="ol_key", how="left")
cmp["rank_raw"]     = cmp["rank_raw"].fillna(">100").astype(str)
cmp["rank_threads"] = cmp["rank_threads"].fillna(">100").astype(str)

# Rank shift (numeric only)
cmp["rank_shift"] = pd.to_numeric(cmp["rank_raw"], errors="coerce") - \
                    pd.to_numeric(cmp["rank_threads"], errors="coerce")

cmp = cmp.rename(columns={
    "canonical_title_r": "canonical_title",
    "author_name_r":     "author_name",
    "mention_count":     "mention_count",
    "thread_count":      "thread_count",
}).sort_values(
    by=["rank_raw"],
    key=lambda col: pd.to_numeric(col, errors="coerce").fillna(999),
)[["ol_key", "canonical_title", "author_name",
   "mention_count", "thread_count",
   "rank_raw", "rank_threads", "rank_shift"]]

cmp.to_csv("output/top100_comparison.csv", index=False)

print("\n" + "═" * 55)
print("TOP 20 — RANK COMPARISON (raw vs. thread-normalised)")
print("═" * 55)
print(
    cmp.head(20)[["canonical_title", "mention_count", "thread_count", "rank_raw", "rank_threads", "rank_shift"]]
    .to_string(index=False)
)

# ── Extra: mentions-per-thread ratio (hubness) ────────────────────────────────

hub = (
    res.groupby(["ol_key", "canonical_title", "author_name"])
    .agg(mention_count=("text", "count"), thread_count=("thread_id", "nunique"))
    .reset_index()
)
hub["mentions_per_thread"] = (hub["mention_count"] / hub["thread_count"]).round(2)
hub_top = hub[hub["thread_count"] >= 5].sort_values("mentions_per_thread", ascending=False).head(20)

print("\n" + "═" * 55)
print("TOP 20 — MOST OBSESSIVELY DISCUSSED (mentions/thread, min 5 threads)")
print("═" * 55)
print(hub_top[["canonical_title", "author_name", "mention_count", "thread_count", "mentions_per_thread"]].to_string(index=False))

print("\n✓ Outputs written to output/top100_raw.csv, top100_thread_norm.csv, top100_comparison.csv")