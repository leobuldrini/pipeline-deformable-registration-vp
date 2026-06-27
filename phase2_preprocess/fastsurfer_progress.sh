#!/usr/bin/env bash
#
# Monitor FastSurfer batch v2 progress by counting completed outputs.
# Estimates remaining time based on recent completion rate.
#
# Usage:
#   ./code/fastsurfer_progress_v2.sh              # one-shot
#   watch -n 30 ./code/fastsurfer_progress_v2.sh  # auto-refresh every 30s

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOBS_FILE="$SCRIPT_DIR/fastsurfer_jobs_v2.tsv"
OUTPUT_BASE="$(dirname "$SCRIPT_DIR")/fastsurfer_output/phase2"   # in-repo

if [[ ! -f "$JOBS_FILE" ]]; then
    echo "Error: $JOBS_FILE not found"
    exit 1
fi

TOTAL=$(tail -n +2 "$JOBS_FILE" | wc -l)

# Count completed (follow symlinks with -L)
mapfile -t COMPLETED < <(find -L "$OUTPUT_BASE" -name "aparc.DKTatlas+aseg.deep.mgz" -printf '%T@\n' 2>/dev/null | sort -n)
DONE=${#COMPLETED[@]}
REMAINING=$((TOTAL - DONE))
PCT=0
if [[ "$TOTAL" -gt 0 ]]; then
    PCT=$((DONE * 100 / TOTAL))
fi

# Progress bar
BAR_WIDTH=40
FILLED=$((PCT * BAR_WIDTH / 100))
EMPTY=$((BAR_WIDTH - FILLED))
BAR=$(printf '%0.s#' $(seq 1 "$FILLED" 2>/dev/null))
BAR+=$(printf '%0.s-' $(seq 1 "$EMPTY" 2>/dev/null))

# ETA from last N completions
ETA_STR="N/A"
RATE_STR=""
if [[ "$DONE" -ge 2 ]]; then
    SAMPLE_SIZE=20
    if [[ "$DONE" -lt "$SAMPLE_SIZE" ]]; then
        SAMPLE_SIZE="$DONE"
    fi

    NEWEST="${COMPLETED[$((DONE - 1))]}"
    OLDEST="${COMPLETED[$((DONE - SAMPLE_SIZE))]}"

    SPAN=$(awk "BEGIN { printf \"%.0f\", $NEWEST - $OLDEST }")

    if [[ "$SPAN" -gt 0 ]]; then
        RATE=$(awk "BEGIN { printf \"%.6f\", ($SAMPLE_SIZE - 1) / $SPAN }")
        ETA_SECS=$(awk "BEGIN { printf \"%.0f\", $REMAINING / $RATE }")

        HOURS=$((ETA_SECS / 3600))
        MINS=$(( (ETA_SECS % 3600) / 60 ))
        if [[ "$HOURS" -gt 0 ]]; then
            ETA_STR="${HOURS}h ${MINS}m"
        else
            ETA_STR="${MINS}m"
        fi

        RATE_MIN=$(awk "BEGIN { printf \"%.1f\", $RATE * 60 }")
        RATE_STR="${RATE_MIN} subjects/min"
    fi
fi

echo "============================================================"
echo "FastSurfer v2 Progress: ${DONE}/${TOTAL} (${PCT}%)"
echo "[${BAR}]"
echo "Remaining: ${REMAINING}    ETA: ${ETA_STR}"
[[ -n "$RATE_STR" ]] && echo "Rate: ${RATE_STR}"
echo "============================================================"
