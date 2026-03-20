#!/usr/bin/env bash
# monolith_dl.sh
# Selectively download, filter, and delete a pushshift monolith torrent
# in configurable batches to cap peak disk usage.
#
# Usage:
#   ./monolith_dl.sh <file.torrent | magnet_link> [--batch-size N]
#
#   --batch-size N   files per download-filter-delete cycle (default: 48)
#
# Disk usage per batch (approximate):
#   --batch-size 48  →  ~2 TB   (full run, no cycling)
#   --batch-size 24  →  ~1 TB   (good for older/smaller months)
#   --batch-size 12  →  ~500 GB (safe for recent large months)
#   --batch-size  8  →  ~300 GB (fits most included local NVMe)
#
# Example — two-phase strategy matching archive size distribution:
#   ./monolith_dl.sh file.torrent --batch-size 24   # 2021-01..2022-12
#   edit MIN_YM/MAX_YM, then:
#   ./monolith_dl.sh file.torrent --batch-size 12   # 2023-01..2024-12
#
# Requirements:
#   pip install torf

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOAD_DIR="/workspace/downloads"
FILTERED_SUBS="/workspace/filtered/submissions"
FILTERED_COMS="/workspace/filtered/comments"
META_DIR="/workspace/torrent_meta"
LOG_FILE="${SCRIPT_DIR}/pipeline_monolith.log"
WORKERS=$(nproc)

# ── date range — adjust before each run ──────────────────────────────────────
MIN_YM="2021-01"   # inclusive lower bound (YYYY-MM)
MAX_YM="2024-12"   # inclusive upper bound (YYYY-MM)

# ── parse arguments ───────────────────────────────────────────────────────────
INPUT=""
BATCH_SIZE=48   # default: all at once (original behaviour)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --batch-size) BATCH_SIZE="${2:?--batch-size requires a value}"; shift 2 ;;
        --*)          echo "Unknown option: $1"; exit 1 ;;
        *)            INPUT="$1"; shift ;;
    esac
done

if [ -z "$INPUT" ]; then
    echo "Usage: $0 <file.torrent | magnet_link> [--batch-size N]"
    exit 1
fi

mkdir -p "$META_DIR" "$FILTERED_SUBS" "$FILTERED_COMS"

log() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG_FILE"
}

# ── Phase 1: resolve torrent source ──────────────────────────────────────────
if [[ "$INPUT" == magnet:* ]]; then
    log "Phase 1: magnet detected — fetching metadata from peers..."
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
    if [ ! -f "$TORRENT_FILE" ]; then
        log "ERROR: file not found: ${TORRENT_FILE}"
        exit 1
    fi
    log "Phase 1: using local torrent file: ${TORRENT_FILE}"
fi

# ── Phase 2: build ordered index list ────────────────────────────────────────
log "Phase 2: selecting files ${MIN_YM}..${MAX_YM}..."

# Outputs one "INDEX PATH" line per selected file, sorted chronologically.
# Both RC and RS for a given month land in the same batch where possible.
INDEX_PATH_LIST=$(python3 - <<EOF
import sys, re
try:
    import torf
except ImportError:
    print("ERROR: run: pip install torf", file=sys.stderr)
    sys.exit(1)

t      = torf.Torrent.read("${TORRENT_FILE}")
min_ym = "${MIN_YM}"
max_ym = "${MAX_YM}"
rows   = []

for i, f in enumerate(t.files, start=1):
    path = str(f)
    m = re.search(r'[RC][CS]_(\d{4}-\d{2})\.zst', path)
    if m:
        ym = m.group(1)
        if min_ym <= ym <= max_ym:
            rows.append((ym, path, i))
            print(f"  SELECTED [{i:4d}] {path}", file=sys.stderr)
        else:
            print(f"  skipped  [{i:4d}] {path}", file=sys.stderr)
    else:
        print(f"  skipped  [{i:4d}] {path}  (no date match)", file=sys.stderr)

rows.sort()   # chronological; submissions sort before comments within same month
for ym, path, idx in rows:
    print(f"{idx} {path}")
EOF
)

if [ -z "$INDEX_PATH_LIST" ]; then
    log "ERROR: no files matched range ${MIN_YM}..${MAX_YM}"
    exit 1
fi

TOTAL_FILES=$(echo "$INDEX_PATH_LIST" | wc -l)
TOTAL_BATCHES=$(( (TOTAL_FILES + BATCH_SIZE - 1) / BATCH_SIZE ))
log "Selected ${TOTAL_FILES} files → ${TOTAL_BATCHES} batch(es) of up to ${BATCH_SIZE}"

# ── helper: filter both dirs in parallel then clean up ───────────────────────
filter_and_clean() {
    local subs_dir="${DOWNLOAD_DIR}/reddit/submissions"
    local coms_dir="${DOWNLOAD_DIR}/reddit/comments"

    if [ -d "$subs_dir" ] && compgen -G "${subs_dir}/*.zst" > /dev/null 2>&1; then
        log "  Filtering submissions (${WORKERS} workers)..."
        python3 "${SCRIPT_DIR}/worker.py" \
            "$subs_dir" "$FILTERED_SUBS" \
            --workers "$WORKERS" --delete-source \
            2>&1 | tee -a "$LOG_FILE" &
    fi

    if [ -d "$coms_dir" ] && compgen -G "${coms_dir}/*.zst" > /dev/null 2>&1; then
        log "  Filtering comments (${WORKERS} workers)..."
        python3 "${SCRIPT_DIR}/worker.py" \
            "$coms_dir" "$FILTERED_COMS" \
            --workers "$WORKERS" --delete-source \
            2>&1 | tee -a "$LOG_FILE" &
    fi

    wait   # submissions + comments run in parallel

    find "${DOWNLOAD_DIR}" -type f -name "*.aria2" -delete
    find "${DOWNLOAD_DIR}" -mindepth 2 -type d -empty -delete
}

# ── Phase 3: batched download → filter → delete loop ─────────────────────────
BATCH_NUM=0
BATCH_INDICES=()

process_batch() {
    if [ ${#BATCH_INDICES[@]} -eq 0 ]; then return; fi

    BATCH_NUM=$(( BATCH_NUM + 1 ))
    local select_str
    select_str=$(IFS=,; echo "${BATCH_INDICES[*]}")
    local count=${#BATCH_INDICES[@]}

    log "================================================================"
    log "Batch ${BATCH_NUM}/${TOTAL_BATCHES} — downloading ${count} file(s)"
    log "  Indices: ${select_str}"

    aria2c \
        "${TORRENT_FILE}" \
        --select-file="${select_str}" \
        --file-allocation=none \
        --seed-time=0 \
        --dir="${DOWNLOAD_DIR}" \
        --console-log-level=notice \
        --summary-interval=60 \
        2>&1 | tee -a "$LOG_FILE"

    log "Batch ${BATCH_NUM}/${TOTAL_BATCHES} — download complete, filtering..."
    filter_and_clean

    log "Batch ${BATCH_NUM}/${TOTAL_BATCHES} — done."
    log "  Submissions so far: $(ls "${FILTERED_SUBS}"/*.jsonl 2>/dev/null | wc -l) file(s)"
    log "  Comments so far:    $(ls "${FILTERED_COMS}"/*.jsonl 2>/dev/null | wc -l) file(s)"

    BATCH_INDICES=()
}

while IFS=" " read -r idx path; do
    BATCH_INDICES+=("$idx")
    if [ ${#BATCH_INDICES[@]} -ge "$BATCH_SIZE" ]; then
        process_batch
    fi
done <<< "$INDEX_PATH_LIST"

# flush any remaining files in a partial last batch
process_batch

# ── Done ──────────────────────────────────────────────────────────────────────
log "================================================================"
log "All ${TOTAL_BATCHES} batch(es) complete."
log "Submissions: $(ls "${FILTERED_SUBS}"/*.jsonl 2>/dev/null | wc -l) file(s)"
log "Comments:    $(ls "${FILTERED_COMS}"/*.jsonl 2>/dev/null | wc -l) file(s)"
