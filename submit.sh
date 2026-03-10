#!/bin/bash


JOB_ID=$(sbatch --parsable run_array.sh)
echo "Submitted array job: ${JOB_ID}"

sbatch --job-name=merge_results --partition=RM-shared --time=00:30:00 --ntasks=1 --cpus-per-task=8 --mem=14G --dependency=afterok:${JOB_ID} --output=logs/merge_%j.out --error=logs/merge_%j.err --wrap="source ~/.bashrc && conda activate sam2_seg && cd /ocean/projects/asc180003p/szaidi/Tackle_Sam2 && python merge_batch_results.py --output_dir /ocean/projects/asc180003p/szaidi/Tackle_Sam2/Results_v2"

echo "Merge job queued — will run after all 4 batches succeed."