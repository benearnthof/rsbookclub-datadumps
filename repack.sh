#!/usr/bin/env bash
# repack.sh
# Recompress filtered JSONL files back into .zst archives for sharing.
#
# Output structure mirrors the original pushshift layout:
#   release/
#     comments/RC_YYYY-MM.jsonl.zst
#     submissions/RS_YYYY-MM.jsonl.zst
#     SHA256SUMS.txt
#
# Usage:
#   ./repack.sh [--level N]   compression level 1-19, default 15
#               [--threads N] zstd threads per file, default $(nproc)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FILTERED_SUBS="/workspace/filtered/submissions"
FILTERED_COMS="/workspace/filtered/comments"
RELEASE_DIR="/workspace/release"
LOG_FILE="${SCRIPT_DIR}/repack.log"

LEVEL=15 
THREADS=$(nproc)

# parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --level)   LEVEL="$2";   shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

log() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG_FILE"
}

human_bytes() {
    numfmt --to=iec-i --suffix=B "$1" 2>/dev/null || echo "${1}B"
}

mkdir -p "${RELEASE_DIR}/submissions" "${RELEASE_DIR}/comments"

log "Repacking filtered JSONL → zstd (level=${LEVEL}, threads=${THREADS})"

TOTAL_IN=0
TOTAL_OUT=0
FILE_COUNT=0

compress_dir() {
    local src_dir="$1"
    local dst_dir="$2"

    for src in "${src_dir}"/*.jsonl; do
        [ -f "$src" ] || continue
        local fname
        fname=$(basename "$src")
        local dst="${dst_dir}/${fname}.zst"

        local size_in
        size_in=$(stat -c%s "$src")

        log "Compressing ${fname} ($(human_bytes ${size_in}))..."
        zstd \
            -"${LEVEL}" \
            -T"${THREADS}" \
            --no-progress \
            --force \
            "$src" \
            -o "$dst"

        local size_out
        size_out=$(stat -c%s "$dst")
        local ratio
        ratio=$(awk "BEGIN {printf \"%.1f\", ${size_in}/${size_out}}")

        log "  → $(human_bytes ${size_out})  (${ratio}x ratio)"

        TOTAL_IN=$((TOTAL_IN + size_in))
        TOTAL_OUT=$((TOTAL_OUT + size_out))
        FILE_COUNT=$((FILE_COUNT + 1))
    done
}

compress_dir "$FILTERED_SUBS" "${RELEASE_DIR}/submissions"
compress_dir "$FILTERED_COMS" "${RELEASE_DIR}/comments"

log "Generating SHA256SUMS.txt..."
(
    cd "${RELEASE_DIR}"
    find submissions comments -name "*.zst" | sort | xargs sha256sum
) > "${RELEASE_DIR}/SHA256SUMS.txt"

RATIO_TOTAL=$(awk "BEGIN {printf \"%.1f\", ${TOTAL_IN}/${TOTAL_OUT}}")

log "Done. ${FILE_COUNT} file(s) repacked."
log "Uncompressed : $(human_bytes ${TOTAL_IN})"
log "Compressed   : $(human_bytes ${TOTAL_OUT})"
log "Overall ratio: ${RATIO_TOTAL}x"
log "Output       : ${RELEASE_DIR}/"
