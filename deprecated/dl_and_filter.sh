#!/usr/bin/env bash
# dl_and_filter.sh
# Download -> filter -> delete, one torrent at a time.
# Keeps peak disk usage to max ~65GB.
#
# Prerequisites:
#   - magnets.txt: one magnet link per line (blank lines and # comments ignored)
#   - worker.py in the same directory

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOAD_DIR="/workspace/downloads"
FILTERED_SUBS="/workspace/filtered/submissions"
FILTERED_COMS="/workspace/filtered/comments"
MAGNETS_FILE="${SCRIPT_DIR}/magnets.txt"
LOG_FILE="${SCRIPT_DIR}/pipeline.log"
WORKERS=$(nproc)

mkdir -p "$FILTERED_SUBS" "$FILTERED_COMS"

log() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG_FILE"
}

# Read magnet links
mapfile -t MAGNETS < <(grep -v '^\s*#' "$MAGNETS_FILE" | grep -v '^\s*$')
TOTAL=${#MAGNETS[@]}
log "Found ${TOTAL} magnet link(s) to process"

for i in "${!MAGNETS[@]}"; do
    MAGNET="${MAGNETS[$i]}"
    NUM=$((i + 1))

    log "Torrent ${NUM}/${TOTAL} — starting download"

    # download
    aria2c \
        "${MAGNET}" \
        --max-concurrent-downloads=1 \
        --seed-time=0 \
        --dir="${DOWNLOAD_DIR}" \
        --console-log-level=notice \
        --summary-interval=60 \
        2>&1 | tee -a "$LOG_FILE"

    log "Torrent ${NUM}/${TOTAL} — download complete"

    # filter submissions
    SUBS_DIR="${DOWNLOAD_DIR}/reddit/submissions"
    if [ -d "$SUBS_DIR" ] && compgen -G "${SUBS_DIR}/*.zst" > /dev/null; then
        log "Filtering submissions with ${WORKERS} workers..."
        python3 "${SCRIPT_DIR}/worker.py" \
            "$SUBS_DIR" \
            "$FILTERED_SUBS" \
            --workers "$WORKERS" \
            --delete-source \
            2>&1 | tee -a "$LOG_FILE"
    else
        log "No submission .zst files found in ${SUBS_DIR} — skipping"
    fi

    # filter comments
    COMS_DIR="${DOWNLOAD_DIR}/reddit/comments"
    if [ -d "$COMS_DIR" ] && compgen -G "${COMS_DIR}/*.zst" > /dev/null; then
        log "Filtering comments with ${WORKERS} workers..."
        python3 "${SCRIPT_DIR}/worker.py" \
            "$COMS_DIR" \
            "$FILTERED_COMS" \
            --workers "$WORKERS" \
            --delete-source \
            2>&1 | tee -a "$LOG_FILE"
    else
        log "No comment .zst files found in ${COMS_DIR} — skipping"
    fi

    # cleanup
    log "Cleaning up download directory..."
    find "${DOWNLOAD_DIR}/reddit" -type f \( -name "*.aria2" -o -name "*.torrent" \) -delete
    find "${DOWNLOAD_DIR}/reddit" -mindepth 1 -type d -empty -delete
    find "${DOWNLOAD_DIR}" -mindepth 1 -maxdepth 1 -type d -empty -delete

    log "Torrent ${NUM}/${TOTAL} — complete. Filtered output in:"
    log "  ${FILTERED_SUBS}"
    log "  ${FILTERED_COMS}"
done

log "================================================================"
log "All ${TOTAL} torrents processed."
log "Submissions: $(ls "${FILTERED_SUBS}"/*.jsonl 2>/dev/null | wc -l) file(s)"
log "Comments:    $(ls "${FILTERED_COMS}"/*.jsonl 2>/dev/null | wc -l) file(s)"
