#!/usr/bin/env bash
#
# Batch-run FastSurfer segmentation on top-1k ranked scans.
#
# Usage:
#   bash code/run_fastsurfer_batch_v2.sh              # full run
#   bash code/run_fastsurfer_batch_v2.sh --dry-run     # print commands only
#   bash code/run_fastsurfer_batch_v2.sh --limit 5     # first 5 jobs
#   bash code/run_fastsurfer_batch_v2.sh --parallel 4  # 4 concurrent workers
#   bash code/run_fastsurfer_batch_v2.sh --chunk 1/4   # manual chunk
#
# Requires: code/fastsurfer_jobs_v2.tsv (from generate_fastsurfer_jobs_v2.py)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
FASTSURFER="$PROJECT_DIR/FastSurfer/run_fastsurfer.sh"
JOBS_FILE="$SCRIPT_DIR/fastsurfer_jobs_v2.tsv"
OUTPUT_BASE="$PROJECT_DIR/fastsurfer_output/phase2"   # in-repo (override with --output)

DRY_RUN=0
LIMIT=0
PARALLEL=0
CHUNK_ID=0
CHUNK_TOTAL=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=1; shift ;;
        --limit)     LIMIT="$2"; shift 2 ;;
        --parallel)  PARALLEL="$2"; shift 2 ;;
        --chunk)     IFS='/' read -r CHUNK_ID CHUNK_TOTAL <<< "$2"; shift 2 ;;
        --output)    OUTPUT_BASE="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--limit N] [--parallel N] [--chunk K/N] [--output DIR]"
            exit 1 ;;
    esac
done

if [[ ! -f "$JOBS_FILE" ]]; then
    echo "Error: Job list not found: $JOBS_FILE"
    echo "Run: python3 code/generate_fastsurfer_jobs_v2.py"
    exit 1
fi

if [[ ! -f "$FASTSURFER" ]]; then
    echo "Error: FastSurfer not found: $FASTSURFER"
    exit 1
fi

TOTAL=$(tail -n +2 "$JOBS_FILE" | wc -l)
if [[ "$LIMIT" -gt 0 && "$LIMIT" -lt "$TOTAL" ]]; then
    TOTAL="$LIMIT"
fi

# --- Parallel mode ---
if [[ "$PARALLEL" -gt 1 ]]; then
    echo "============================================================"
    echo "FastSurfer Batch v2 — PARALLEL MODE"
    echo "============================================================"
    echo "Jobs file:  $JOBS_FILE"
    echo "Output dir: $OUTPUT_BASE"
    echo "Total jobs: $TOTAL"
    echo "Workers:    $PARALLEL"
    [[ "$DRY_RUN" -eq 1 ]] && echo "Mode:       DRY RUN"
    echo "============================================================"

    FORWARD_ARGS=()
    [[ "$DRY_RUN" -eq 1 ]] && FORWARD_ARGS+=(--dry-run)
    [[ "$LIMIT" -gt 0 ]] && FORWARD_ARGS+=(--limit "$LIMIT")
    FORWARD_ARGS+=(--output "$OUTPUT_BASE")

    PIDS=()
    cleanup() {
        echo ""; echo "Interrupted — killing workers..."
        for pid in "${PIDS[@]}"; do
            kill -TERM -- -"$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null
        done
        wait 2>/dev/null; exit 1
    }
    trap cleanup INT TERM

    for (( i=1; i<=PARALLEL; i++ )); do
        LOG="$SCRIPT_DIR/fastsurfer_v2_chunk_${i}.log"
        setsid bash "$0" --chunk "$i/$PARALLEL" "${FORWARD_ARGS[@]}" > "$LOG" 2>&1 &
        PIDS+=($!)
        echo "  Worker $i/$PARALLEL → $LOG (pid $!)"
    done

    echo ""
    echo "Monitor: tail -f $SCRIPT_DIR/fastsurfer_v2_chunk_*.log"
    echo "Waiting..."

    FAILURES=0
    for pid in "${PIDS[@]}"; do
        wait "$pid" || FAILURES=$((FAILURES + 1))
    done
    trap - INT TERM

    echo ""
    echo "============================================================"
    echo "All workers finished. Failures: $FAILURES"
    echo "============================================================"
    exit 0
fi

# --- Sequential / single-chunk mode ---
echo "============================================================"
echo "FastSurfer Batch v2"
echo "============================================================"
echo "Jobs file:  $JOBS_FILE"
echo "Output dir: $OUTPUT_BASE"
echo "Total jobs: $TOTAL"
[[ "$CHUNK_TOTAL" -gt 0 ]] && echo "Chunk:      $CHUNK_ID / $CHUNK_TOTAL"
[[ "$DRY_RUN" -eq 1 ]] && echo "Mode:       DRY RUN"
echo "============================================================"

mkdir -p "$OUTPUT_BASE"

COUNT=0
SKIPPED=0
PROCESSED=0

# TSV columns: patient_id, date, pre_path (no cohort)
while IFS=$'\t' read -r PATIENT_ID DATE PRE_PATH; do
    COUNT=$((COUNT + 1))

    [[ "$LIMIT" -gt 0 && "$COUNT" -gt "$LIMIT" ]] && break

    # Chunk round-robin
    if [[ "$CHUNK_TOTAL" -gt 0 ]]; then
        [[ $(( (COUNT - 1) % CHUNK_TOTAL )) -ne $((CHUNK_ID - 1)) ]] && continue
    fi

    SID="${PATIENT_ID}_${DATE}"

    # Resume: skip if already done
    EXPECTED="$OUTPUT_BASE/$SID/mri/aparc.DKTatlas+aseg.deep.mgz"
    if [[ -f "$EXPECTED" ]]; then
        echo "[$COUNT/$TOTAL] SKIP (done): $SID"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    if [[ ! -f "$PRE_PATH" ]]; then
        echo "[$COUNT/$TOTAL] SKIP (missing): $SID — $PRE_PATH"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo "[$COUNT/$TOTAL] Processing $SID..."

    FASTSURFER_ARGS=(
        --sid "$SID"
        --sd "$OUTPUT_BASE"
        --t1 "$PRE_PATH"
        --seg_only
        --vox_size 1
        --no_hypothal
        --no_cc
        --device cuda
        --viewagg_device cuda
        --threads 4
    )

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "  CMD: $FASTSURFER ${FASTSURFER_ARGS[*]}"
    else
        if "$FASTSURFER" "${FASTSURFER_ARGS[@]}"; then
            PROCESSED=$((PROCESSED + 1))
        else
            echo "  ERROR: FastSurfer failed for $SID"
        fi
    fi
done < <(tail -n +2 "$JOBS_FILE")

echo ""
echo "============================================================"
echo "Done. Processed: $PROCESSED  Skipped: $SKIPPED  Total: $COUNT"
echo "============================================================"
