#!/usr/bin/env bash
# monolith_dl.sh
# Phase 1: fetch torrent metadata only & inspect file list
# Phase 2: download only files within MIN_YM..MAX_YM inclusive
#
# Usage:
#   ./monolith_dl.sh "<magnet_link>"
#
# Requirements:
#   pip install torf

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAGNET="${1:?Usage: $0 '<magnet_link>'}"
DOWNLOAD_DIR="/workspace/downloads"
FILTERED_SUBS="/workspace/filtered/submissions"
FILTERED_COMS="/workspace/filtered/comments"
META_DIR="/workspace/torrent_meta"
LOG_FILE="${SCRIPT_DIR}/pipeline_monolith.log"
WORKERS=$(nproc)
MIN_YM="2021-01"   # inclusive lower bound (YYYY-MM)
MAX_YM="2024-12"   # inclusive upper bound (YYYY-MM)

mkdir -p "$META_DIR" "$FILTERED_SUBS" "$FILTERED_COMS"

log() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG_FILE"
}

# ── Phase 1: fetch metadata ───────────────────────────────────────────────────
log "Phase 1: fetching torrent metadata (no data downloaded)..."

aria2c \
    "${MAGNET}" \
    --bt-metadata-only=true \
    --bt-save-metadata=true \
    --dir="${META_DIR}" \
    --seed-time=0 \
    --console-log-level=notice \
    2>&1 | tee -a "$LOG_FILE"

# aria2c saves the .torrent as <info-hash>.torrent
TORRENT_FILE=$(find "${META_DIR}" -name "*.torrent" | head -1)
if [ -z "$TORRENT_FILE" ]; then
    log "ERROR: no .torrent file found in ${META_DIR}"
    exit 1
fi
log "Metadata saved: ${TORRENT_FILE}"

# ── Phase 2: parse file list and build --select-file indices ─────────────────
log "Phase 2: parsing file list and selecting files ${MIN_YM}..${MAX_YM}..."

SELECT_INDICES=$(python3 - <<EOF
import sys, re
try:
    import torf
except ImportError:
    print("ERROR: run: pip install torf", file=sys.stderr)
    sys.exit(1)

t      = torf.Torrent.read("${TORRENT_FILE}")
min_ym = "${MIN_YM}"
max_ym = "${MAX_YM}"
selected = []

# torf uses 1-based file indices (matching aria2c's --select-file)
for i, f in enumerate(t.files, start=1):
    path = str(f)
    m = re.search(r'[RC][CS]_(\d{4}-\d{2})\.zst', path)
    if m:
        ym = m.group(1)          # "YYYY-MM" — string comparison works correctly
        if min_ym <= ym <= max_ym:
            selected.append(i)
            print(f"  SELECTED [{i:4d}] {path}", file=sys.stderr)
        else:
            print(f"  skipped  [{i:4d}] {path}", file=sys.stderr)
    else:
        print(f"  skipped  [{i:4d}] {path}  (no date match)", file=sys.stderr)

print(",".join(str(i) for i in selected))
EOF
)

if [ -z "$SELECT_INDICES" ]; then
    log "ERROR: no files matched range ${MIN_YM}..${MAX_YM}"
    exit 1
fi

COUNT=$(echo "$SELECT_INDICES" | tr ',' '\n' | wc -l)
log "Selected ${COUNT} files — indices: ${SELECT_INDICES}"

# ── Phase 3: download only selected files ────────────────────────────────────
log "Phase 3: downloading ${COUNT} selected files..."

aria2c \
    "${MAGNET}" \
    --select-file="${SELECT_INDICES}" \
    --seed-time=0 \
    --dir="${DOWNLOAD_DIR}" \
    --console-log-level=notice \
    --summary-interval=60 \
    2>&1 | tee -a "$LOG_FILE"

log "Download complete."

# ── Phase 4 & 5: filter submissions and comments in parallel ─────────────────
# Both dirs are independent — run concurrently so the faster one
# (submissions) doesn't block behind comments.
SUBS_DIR="${DOWNLOAD_DIR}/reddit/submissions"
COMS_DIR="${DOWNLOAD_DIR}/reddit/comments"

if [ -d "$SUBS_DIR" ] && compgen -G "${SUBS_DIR}/*.zst" > /dev/null; then
    log "Filtering submissions with ${WORKERS} workers (background)..."
    python3 "${SCRIPT_DIR}/worker.py" \
        "$SUBS_DIR" "$FILTERED_SUBS" \
        --workers "$WORKERS" --delete-source \
        2>&1 | tee -a "$LOG_FILE" &
fi

if [ -d "$COMS_DIR" ] && compgen -G "${COMS_DIR}/*.zst" > /dev/null; then
    log "Filtering comments with ${WORKERS} workers (background)..."
    python3 "${SCRIPT_DIR}/worker.py" \
        "$COMS_DIR" "$FILTERED_COMS" \
        --workers "$WORKERS" --delete-source \
        2>&1 | tee -a "$LOG_FILE" &
fi

log "Waiting for both filter jobs to complete..."
wait
log "Filtering complete."

# ── Phase 6: cleanup ──────────────────────────────────────────────────────────
log "Cleaning up..."
find "${DOWNLOAD_DIR}/reddit" -type f \( -name "*.aria2" -o -name "*.torrent" \) -delete
find "${DOWNLOAD_DIR}/reddit" -mindepth 1 -type d -empty -delete
find "${DOWNLOAD_DIR}" -mindepth 1 -maxdepth 1 -type d -empty -delete

log "All done."
log "Submissions: $(ls "${FILTERED_SUBS}"/*.jsonl 2>/dev/null | wc -l) file(s)"
log "Comments:    $(ls "${FILTERED_COMS}"/*.jsonl 2>/dev/null | wc -l) file(s)"
