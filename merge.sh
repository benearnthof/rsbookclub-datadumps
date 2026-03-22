#!/usr/bin/env bash
# merge.sh
# Decompress and merge all per-month JSONL files into two reference files.
#
# Output:
#   releases/rsbookclub_submissions.jsonl
#   releases/rsbookclub_comments.jsonl
#
# Usage:
#   ./merge.sh [--releases-dir DIR] [--compress]
#
#   --releases-dir  root of the releases tree (default: ./releases)
#   --compress      also write .zst versions of the merged files

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASES_DIR="${SCRIPT_DIR}/releases"
COMPRESS=false
LOG_FILE="${SCRIPT_DIR}/merge.log"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --releases-dir) RELEASES_DIR="$2"; shift 2 ;;
        --compress)     COMPRESS=true;     shift   ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

log() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG_FILE"
}

human_bytes() {
    numfmt --to=iec-i --suffix=B "$1" 2>/dev/null || echo "${1}B"
}

OUT_SUBS="${RELEASES_DIR}/rsbookclub_submissions.jsonl"
OUT_COMS="${RELEASES_DIR}/rsbookclub_comments.jsonl"

merge_dir() {
    local src_dir="$1"
    local out_file="$2"
    local label="$3"

    # Glob in sorted (chronological) order
    local files=( $(ls "${src_dir}"/*.jsonl.zst 2>/dev/null | sort) )

    if [ ${#files[@]} -eq 0 ]; then
        log "WARNING: no .jsonl.zst files found in ${src_dir} — skipping ${label}"
        return
    fi

    log "Merging ${#files[@]} ${label} file(s) → $(basename "$out_file")"
    : > "$out_file"   # truncate / create

    local count=0
    for f in "${files[@]}"; do
        zstd -dc --no-progress "$f" >> "$out_file"
        count=$(( count + 1 ))
        log "  [${count}/${#files[@]}] $(basename "$f")"
    done

    local lines
    lines=$(wc -l < "$out_file")
    local size
    size=$(stat -c%s "$out_file")
    log "${label} merged: ${lines} lines  $(human_bytes ${size})"

    if $COMPRESS; then
        log "Compressing → $(basename "${out_file}").zst ..."
        zstd -15 -T0 --no-progress --force "$out_file" -o "${out_file}.zst"
        local csize
        csize=$(stat -c%s "${out_file}.zst")
        log "  compressed: $(human_bytes ${csize})"
    fi
}

log "================================================================"
log "Merging releases from ${RELEASES_DIR}"

merge_dir "${RELEASES_DIR}/submissions" "$OUT_SUBS" "submissions"
merge_dir "${RELEASES_DIR}/comments"    "$OUT_COMS" "comments"

# Checksums for the merged files
log "Writing checksums..."
(
    cd "${RELEASES_DIR}"
    sha256sum \
        "$(basename "$OUT_SUBS")" \
        "$(basename "$OUT_COMS")" \
        $( $COMPRESS && echo "$(basename "$OUT_SUBS").zst $(basename "$OUT_COMS").zst" || true ) \
        >> SHA256SUMS.txt
)

log "================================================================"
log "Done."
log "  ${OUT_SUBS}"
log "  ${OUT_COMS}"
