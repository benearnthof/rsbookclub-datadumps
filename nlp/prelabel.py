#!/usr/bin/env python3
"""
NER pre-annotation pipeline for Label Studio.

1. Call Claude API per thread, save raw entity strings to extractions.jsonl so the run is resumable.
2. Load tasks.json + checkpoint, resolve exact character spans, write a Label Studio pre-annotation JSON.

Typical workflow
# First time (or after a crash when we skip threads that have already been processed):
python prelabel.py extract tasks.json extractions.jsonl

# Build the final pre-annotation file:
python prelabel.py annotate tasks.json extractions.jsonl preannotated.json

# Or run both phases in one shot:
python prelabel.py run tasks.json extractions.jsonl preannotated.json

Label Studio import
Import preannotated.json via  Project → Import  and choose
"Import as pre-annotations" so the spans appear as draft annotations
for human review.
"""

import argparse
import json
import re
import sys
import time
import unicodedata
import uuid
from pathlib import Path

try:
    import anthropic
except ImportError:
    anthropic = None   # checked at runtime only when the extract phase runs


MODEL          = "claude-haiku-4-5-20251001"
MAX_TOKENS     = 4096
REQUESTS_PER_MINUTE = 50
SLEEP_BETWEEN_CALLS = 60 / REQUESTS_PER_MINUTE

LABEL_BOOK   = "BOOK"
LABEL_WRITER = "WRITER"

SYSTEM_PROMPT = """\
You are a precise literary named-entity recogniser.

Given a Reddit thread about books, extract every mention of a BOOK or WRITER.

Label definitions
-----------------
BOOK   — any book title: novels, short-story collections, poetry collections,
         non-fiction books, graphic novels, plays, essay collections.
         Include series titles used as a standalone reference.
         Include foreign-language titles and alternate titles.
         Do NOT label: podcast names, films, TV shows, albums.

WRITER — any author name, even surname-only ("Nabokov", "Houellebecq"),
         even informal ("Bolaño guy").
         Do NOT label: film directors, musicians, or historical figures
         mentioned purely outside a literary context.

Output format
-------------
Return ONLY a JSON array — no prose, no markdown fences, no explanation.
Each element must have exactly two keys:
  "text"  : the entity string as it appears in the input (verbatim)
  "label" : "BOOK" or "WRITER"

If no entities are found return an empty array: []

Example output:
[
  {"text": "Blood Meridian", "label": "BOOK"},
  {"text": "Cormac McCarthy",  "label": "WRITER"},
  {"text": "Houellebecq",      "label": "WRITER"}
]
"""

def call_api(client, text: str) -> list[dict]:
    """
    Call the Claude API and return a list of {"text": ..., "label": ...} dicts.
    Returns an empty list on parse failure (logged to stderr).
    """
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text}],
    )

    # Catch output truncation before attempting to parse a truncated JSON
    # array is the most common cause of parse failures on long threads.
    if response.stop_reason == "max_tokens":
        print(
            f"\n  WARNING: response hit max_tokens ({MAX_TOKENS}) and was truncated. "
            f"Increase MAX_TOKENS or chunk the input. Returning 0 entities.",
            file=sys.stderr,
        )
        return []

    raw = response.content[0].text.strip()

    # Strip accidental markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        entities = json.loads(raw)
        if not isinstance(entities, list):
            raise ValueError("top-level value is not a list")
        # Normalise and validate each entry
        clean = []
        for item in entities:
            label = str(item.get("label", "")).upper().strip()
            text  = str(item.get("text",  "")).strip()
            if label in (LABEL_BOOK, LABEL_WRITER) and text:
                clean.append({"text": text, "label": label})
        return clean
    except Exception as exc:
        print(f"  WARNING: could not parse API response: {exc}\n  Raw: {raw[:200]}",
              file=sys.stderr)
        return []


def load_checkpoint(checkpoint_path: Path) -> dict[str, list[dict]]:
    """Return {thread_id: [entities]} for all already-processed tasks."""
    done: dict[str, list[dict]] = {}
    if not checkpoint_path.exists():
        return done
    with checkpoint_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                done[record["thread_id"]] = record["entities"]
            except Exception:
                pass
    return done


def phase_extract(tasks_path: Path, checkpoint_path: Path, n: int | None = None) -> None:
    if anthropic is None:
        sys.exit("ERROR: 'anthropic' package not installed. Run: pip install anthropic")

    client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from env

    with tasks_path.open(encoding="utf-8") as fh:
        tasks: list[dict] = json.load(fh)

    if n is not None:
        tasks = tasks[:n]
        print(f"[--n {n}] limiting to first {len(tasks)} tasks", file=sys.stderr)

    done = load_checkpoint(checkpoint_path)
    remaining = [t for t in tasks if t["data"].get("thread_id") not in done]

    print(f"Tasks total   : {len(tasks)}", file=sys.stderr)
    print(f"Already done  : {len(done)}",  file=sys.stderr)
    print(f"To process    : {len(remaining)}", file=sys.stderr)

    if not remaining:
        print("Nothing to do.", file=sys.stderr)
        return

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    with checkpoint_path.open("a", encoding="utf-8") as ckpt:
        for idx, task in enumerate(remaining, start=1):
            data      = task["data"]
            thread_id = data.get("thread_id", f"__unknown_{idx}")
            text      = data.get("text", "")

            if not text.strip():
                entities = []
            else:
                print(f"[{idx}/{len(remaining)}] {thread_id} …", file=sys.stderr, end=" ")
                try:
                    entities = call_api(client, text)
                    print(f"{len(entities)} entities", file=sys.stderr)
                except Exception as exc:
                    print(f"API ERROR: {exc}", file=sys.stderr)
                    entities = []
                time.sleep(SLEEP_BETWEEN_CALLS)

            record = {"thread_id": thread_id, "entities": entities}
            ckpt.write(json.dumps(record, ensure_ascii=False) + "\n")
            ckpt.flush()

    print("Extraction complete.", file=sys.stderr)



def normalise(s: str) -> str:
    """Lowercase, collapse whitespace, normalise unicode for fuzzy matching."""
    s = unicodedata.normalize("NFKD", s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def find_spans(text: str, entity_text: str) -> list[tuple[int, int]]:
    """
    Return a list of (start, end) character offsets for all occurrences of
    entity_text in text.  Three passes, stopping at the first that yields hits:

      1. Verbatim regex (preserves original capitalisation)
      2. Case-insensitive regex
      3. Normalised match on a lowercased copy of the text
         (catches unicode/whitespace drift between LLM output and source)
    """
    spans: list[tuple[int, int]] = []

    def _regex_spans(pattern: str, flags: int = 0) -> list[tuple[int, int]]:
        try:
            return [(m.start(), m.end()) for m in re.finditer(pattern, text, flags)]
        except re.error:
            return []

    # Pass 1 – verbatim
    spans = _regex_spans(re.escape(entity_text))
    if spans:
        return spans

    # Pass 2 – case-insensitive
    spans = _regex_spans(re.escape(entity_text), re.IGNORECASE)
    if spans:
        return spans

    # Pass 3 – normalised search on normalised text
    norm_text   = normalise(text)
    norm_entity = normalise(entity_text)
    if not norm_entity:
        return []

    for m in re.finditer(re.escape(norm_entity), norm_text):
        norm_start, norm_end = m.start(), m.end()
        # Map normalised offsets back to original text offsets.
        # This works perfectly when the only difference is case/unicode;
        # it breaks when whitespace was collapsed – we accept that rare miss.
        orig_start = _norm_to_orig_offset(text, norm_text, norm_start)
        orig_end   = _norm_to_orig_offset(text, norm_text, norm_end)
        if orig_start is not None and orig_end is not None:
            spans.append((orig_start, orig_end))

    return spans


def _norm_to_orig_offset(orig: str, norm: str, norm_offset: int) -> int | None:
    """
    Map a character offset in the normalised string back to the original string.
    We build a cumulative map lazily: normalise char-by-char and track correspondence.
    """
    if norm_offset == 0:
        return 0
    rebuilt = ""
    for i, ch in enumerate(orig):
        rebuilt += normalise(ch)
        if len(rebuilt) >= norm_offset:
            return i + 1
    return None


def dedup_spans(annotations: list[dict]) -> list[dict]:
    """
    Remove exact duplicate (start, end, label) triples that can arise when
    the LLM returns the same entity string multiple times.
    """
    seen = set()
    out  = []
    for ann in annotations:
        key = (ann["value"]["start"], ann["value"]["end"],
               ann["value"]["labels"][0])
        if key not in seen:
            seen.add(key)
            out.append(ann)
    return out


def entities_to_ls_result(text: str, entities: list[dict]) -> list[dict]:
    """
    Convert a list of {"text": ..., "label": ...} entity dicts into the
    Label Studio 'result' format, finding all spans in the source text.
    """
    result: list[dict] = []
    unmatched: list[str] = []

    for ent in entities:
        ent_text  = ent["text"]
        ent_label = ent["label"]
        spans     = find_spans(text, ent_text)

        if not spans:
            unmatched.append(f"{ent_label}:{ent_text!r}")
            continue

        for start, end in spans:
            result.append({
                "id":        str(uuid.uuid4())[:8],
                "from_name": "label",
                "to_name":   "text",
                "type":      "labels",
                "value": {
                    "start":  start,
                    "end":    end,
                    "text":   text[start:end],
                    "labels": [ent_label],
                },
            })

    return dedup_spans(result), unmatched


def phase_annotate(
    tasks_path:      Path,
    checkpoint_path: Path,
    output_path:     Path,
    pretty:          bool = False,
    n:               int | None = None,
) -> None:
    with tasks_path.open(encoding="utf-8") as fh:
        tasks: list[dict] = json.load(fh)

    if n is not None:
        tasks = tasks[:n]
        print(f"[--n {n}] limiting to first {len(tasks)} tasks", file=sys.stderr)

    extractions = load_checkpoint(checkpoint_path)
    print(f"Tasks          : {len(tasks)}",       file=sys.stderr)
    print(f"Extractions    : {len(extractions)}", file=sys.stderr)

    output_tasks: list[dict] = []
    total_spans   = 0
    total_unmatched = 0
    skipped_no_extraction = 0

    for task in tasks:
        data      = task["data"]
        thread_id = data.get("thread_id", "")
        text      = data.get("text", "")

        if thread_id not in extractions:
            skipped_no_extraction += 1
            # Still emit the task so humans can annotate it from scratch
            output_tasks.append({**task, "predictions": []})
            continue

        entities = extractions[thread_id]

        if not entities:
            output_tasks.append({**task, "predictions": []})
            continue

        result, unmatched = entities_to_ls_result(text, entities)
        total_spans     += len(result)
        total_unmatched += len(unmatched)

        if unmatched:
            print(f"  UNMATCHED in {thread_id}: {unmatched}", file=sys.stderr)

        prediction = {
            "model_version": f"{MODEL}-ner",
            "score":         0.8,
            "result":        result,
        }
        output_tasks.append({**task, "predictions": [prediction]})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        if pretty:
            json.dump(output_tasks, fh, ensure_ascii=False, indent=2)
        else:
            json.dump(output_tasks, fh, ensure_ascii=False)

    print(f"\nOutput tasks   : {len(output_tasks)}", file=sys.stderr)
    print(f"Total spans    : {total_spans}",         file=sys.stderr)
    print(f"Unmatched ents : {total_unmatched}",     file=sys.stderr)
    if skipped_no_extraction:
        print(f"No extraction  : {skipped_no_extraction} (emitted with empty predictions)",
              file=sys.stderr)
    print(f"Written to     : {output_path}",         file=sys.stderr)


# cli
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="phase", required=True)

    # extract
    pe = sub.add_parser("extract",
        help="Phase 1: call Claude API, write entity strings to checkpoint JSONL.")
    pe.add_argument("tasks",      type=Path, help="tasks.json from threads_to_labelstudio.py")
    pe.add_argument("checkpoint", type=Path, help="Output checkpoint JSONL (append-safe)")
    pe.add_argument("--n", type=int, default=None, metavar="N",
                    help="Only process the first N tasks (useful for testing)")

    # annotate
    pa = sub.add_parser("annotate",
        help="Phase 2: resolve spans from checkpoint, write LS pre-annotation JSON.")
    pa.add_argument("tasks",      type=Path, help="tasks.json")
    pa.add_argument("checkpoint", type=Path, help="Checkpoint JSONL from extract phase")
    pa.add_argument("output",     type=Path, help="Output pre-annotated JSON for Label Studio")
    pa.add_argument("--pretty",   action="store_true", help="Pretty-print output JSON")
    pa.add_argument("--n", type=int, default=None, metavar="N",
                    help="Only annotate the first N tasks")

    # both
    pr = sub.add_parser("run",
        help="Run both phases end-to-end.")
    pr.add_argument("tasks",      type=Path)
    pr.add_argument("checkpoint", type=Path)
    pr.add_argument("output",     type=Path)
    pr.add_argument("--pretty",   action="store_true")
    pr.add_argument("--n", type=int, default=None, metavar="N",
                    help="Only process the first N tasks (useful for testing)")

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.phase in ("extract", "run"):
        phase_extract(args.tasks, args.checkpoint, n=args.n)

    if args.phase in ("annotate", "run"):
        phase_annotate(
            args.tasks,
            args.checkpoint,
            args.output,
            pretty=getattr(args, "pretty", False),
            n=args.n,
        )


if __name__ == "__main__":
    main()