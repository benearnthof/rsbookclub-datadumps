#!/usr/bin/env bash
# download_torrent.sh
# Selectively download pushshift monolith files by date range.
# Run this on the cheap download VM, writing to a network volume.
# When complete, run filter_data.sh on the high-core VM.
#
# Usage:
#   ./download_torrent.sh <file.torrent | magnet_link>
#
# Requirements:
#   pip install torf

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT="${1:?Usage: $0 <file.torrent | magnet_link>}"
DOWNLOAD_DIR="/workspace/downloads"
META_DIR="/workspace/torrent_meta"
LOG_FILE="${SCRIPT_DIR}/download_monolith.log"

# date range 
MIN_YM="2021-01"   # inclusive lower bound (YYYY-MM)
MAX_YM="2024-12"   # inclusive upper bound (YYYY-MM)

mkdir -p "$META_DIR" "$DOWNLOAD_DIR"

log() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG_FILE"
}

# Step 1: resolve torrent source
if [[ "$INPUT" == magnet:* ]]; then
    log "Step 1: magnet detected — fetching metadata from peers..."
    aria2c \
        "${INPUT}" \
        --bt-metadata-only=true \
        --bt-save-metadata=true \
        --dir="${META_DIR}" \
        --seed-time=0 \
        --console-log-level=notice \
        2>&1 | tee -a "$LOG_FILE"
    TORRENT_FILE=$(find "${META_DIR}" -name "*.torrent" | head -1)
    if [ -z "$TORRENT_FILE" ]; then
        log "ERROR: no .torrent file found in ${META_DIR}"
        exit 1
    fi
    log "Metadata saved: ${TORRENT_FILE}"
else
    TORRENT_FILE="$(realpath "$INPUT")"
    [ -f "$TORRENT_FILE" ] || { log "ERROR: not found: ${TORRENT_FILE}"; exit 1; }
    log "Step 1: using local torrent: ${TORRENT_FILE}"
fi

log "Step 2: selecting files ${MIN_YM}..${MAX_YM}..."

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

for i, f in enumerate(t.files, start=1):
    path = str(f)
    m = re.search(r'[RC][CS]_(\d{4}-\d{2})\.zst', path)
    if m:
        ym = m.group(1)
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
log "Selected ${COUNT} files — starting download..."

# ── Step 3: download ─────────────────────────────────────────────────────────
aria2c \
    "${TORRENT_FILE}" \
    --select-file="${SELECT_INDICES}" \
    --file-allocation=none \
    --seed-time=0 \
    --dir="${DOWNLOAD_DIR}" \
    --console-log-level=notice \
    --summary-interval=60 \
    2>&1 | tee -a "$LOG_FILE"

log "================================================================"
log "Download complete. ${COUNT} files in ${DOWNLOAD_DIR}"
