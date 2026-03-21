#!/usr/bin/env bash
# filter_data.sh
# Filter all .zst files in the download directory in parallel.
# Run this on the high-core VM once download_torrent.sh has finished.
#
# Usage:
#   ./filter_data.sh [--download-dir DIR] [--workers N]
#
# Defaults:
#   --download-dir  /workspace/downloads
#   --workers       $(nproc)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOAD_DIR="/workspace/downloads"
FILTERED_SUBS="/workspace/filtered/submissions"
FILTERED_COMS="/workspace/filtered/comments"
LOG_FILE="${SCRIPT_DIR}/filter_data.log"
WORKERS=$(nproc)

# parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --download-dir) DOWNLOAD_DIR="$2"; shift 2 ;;
        --workers)      WORKERS="$2";      shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "$FILTERED_SUBS" "$FILTERED_COMS"

log() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG_FILE"
}

SUBS_DIR="${DOWNLOAD_DIR}/reddit/submissions"
COMS_DIR="${DOWNLOAD_DIR}/reddit/comments"

# Verify there's actually something to filter
ZST_COUNT=$(find "$SUBS_DIR" "$COMS_DIR" -name "*.zst" 2>/dev/null | wc -l)
if [ "$ZST_COUNT" -eq 0 ]; then
    log "ERROR: no .zst files found under ${DOWNLOAD_DIR}/reddit/"
    log "  Expected: ${SUBS_DIR}/*.zst and/or ${COMS_DIR}/*.zst"
    exit 1
fi
log "Found ${ZST_COUNT} .zst file(s) — filtering with ${WORKERS} workers"

# filter submissions & comments in parallel
if compgen -G "${SUBS_DIR}/*.zst" > /dev/null 2>&1; then
    SUB_COUNT=$(ls "${SUBS_DIR}"/*.zst | wc -l)
    log "Filtering ${SUB_COUNT} submission file(s) → ${FILTERED_SUBS}"
    python3 "${SCRIPT_DIR}/worker.py" \
        "$SUBS_DIR" "$FILTERED_SUBS" \
        --workers "$WORKERS" \
        --delete-source \
        2>&1 | tee -a "$LOG_FILE" &
fi

if compgen -G "${COMS_DIR}/*.zst" > /dev/null 2>&1; then
    COM_COUNT=$(ls "${COMS_DIR}"/*.zst | wc -l)
    log "Filtering ${COM_COUNT} comment file(s) → ${FILTERED_COMS}"
    python3 "${SCRIPT_DIR}/worker.py" \
        "$COMS_DIR" "$FILTERED_COMS" \
        --workers "$WORKERS" \
        --delete-source \
        2>&1 | tee -a "$LOG_FILE" &
fi

log "Both filter jobs running — waiting..."
wait

# cleanup empty dirs left by --delete-source 
find "${DOWNLOAD_DIR}" -type f -name "*.aria2" -delete 2>/dev/null || true
find "${DOWNLOAD_DIR}" -mindepth 2 -type d -empty -delete 2>/dev/null || true

log "================================================================"
log "Done."
log "Submissions: $(ls "${FILTERED_SUBS}"/*.jsonl 2>/dev/null | wc -l) file(s)"
log "Comments:    $(ls "${FILTERED_COMS}"/*.jsonl 2>/dev/null | wc -l) file(s)"
