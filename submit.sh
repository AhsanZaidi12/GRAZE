#!/bin/bash
# Usage:
#   export VIDEO_DIR=/path/to/videos
#   export OUTPUT_DIR=/path/to/results
#   export GRAZE_DIR=/path/to/GRAZE
#   export CONDA_ENV=graze       # your conda env name
#   bash submit.sh

# Validate required env vars
: "${VIDEO_DIR:?Set VIDEO_DIR before running submit.sh}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR before running submit.sh}"
: "${GRAZE_DIR:?Set GRAZE_DIR before running submit.sh}"

mkdir -p logs

JOB_ID=$(sbatch --parsable run_array.sh)
echo "Submitted array job: ${JOB_ID}"

sbatch \
  --job-name=graze_merge \
  --partition=RM-shared \       # Bridges-2 CPU partition; change for your HPC
  --time=00:30:00 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --mem=14G \
  --dependency=afterok:${JOB_ID} \
  --output=logs/merge_%j.out \
  --error=logs/merge_%j.err \
  --wrap="source ~/.bashrc && \
          conda activate ${CONDA_ENV:-graze} && \
          cd ${GRAZE_DIR} && \
          python CombineXls.py --results_dir ${OUTPUT_DIR} --output graze_results.xlsx"

echo "Merge job queued — will run after all 4 batches succeed."
