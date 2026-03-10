#!/bin/bash

JOB_ID=$(sbatch --parsable run_array_B1.sh)
echo "Submitted baseline array job: ${JOB_ID}"

sbatch --job-name=merge_baseline --partition=RM-shared --time=00:30:00 --ntasks=1 --cpus-per-task=8 --mem=14G --dependency=afterok:${JOB_ID} --output=logs/merge_baseline_%j.out --error=logs/merge_baseline_%j.err --wrap="source ~/.bashrc && conda activate sam2_seg && cd /ocean/projects/asc180003p/szaidi/Tackle_Sam2 && python merge_batch_results.py --output_dir /ocean/projects/asc180003p/szaidi/Tackle_Sam2/Results_baseline"

echo "Baseline merge queued — runs after all 4 batches succeed."