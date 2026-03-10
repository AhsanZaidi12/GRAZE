#!/bin/bash
#SBATCH --job-name=tackle_seg
#SBATCH --partition=GPU-shared
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:h100-80:1              # FIXED: correct Bridges-2 H100 identifier
#SBATCH --output=logs/tackle_seg_%A_%a.out
#SBATCH --error=logs/tackle_seg_%A_%a.err
#SBATCH --array=0-3

VIDEO_DIR="/ocean/projects/asc180003p/szaidi/videos"
OUTPUT_DIR="/ocean/projects/asc180003p/szaidi/Tackle_Sam2/Results_v3_mp4"
CONDA_ENV="sam2_seg"

export CUDA_VISIBLE_DEVICES=0

TOTAL_VIDEOS=$(ls -1 ${VIDEO_DIR}/*.{mp4,MP4,avi,AVI,mov,MOV} 2>/dev/null | wc -l)
BATCH_SIZE=$(( (TOTAL_VIDEOS + 3) / 4 ))
BATCH_ID=${SLURM_ARRAY_TASK_ID}

mkdir -p ${OUTPUT_DIR}/logs

source ~/.bashrc
conda activate ${CONDA_ENV}
cd /ocean/projects/asc180003p/szaidi/Tackle_Sam2

echo "=========================================="
echo "Bridges-2 Array Job — Batch ${BATCH_ID}/3"
echo "=========================================="
echo "Node:          ${SLURM_NODELIST}"
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Total videos:  ${TOTAL_VIDEOS}"
echo "Batch size:    ${BATCH_SIZE}"
echo "GPU:           $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start time:    $(date)"
echo "=========================================="

python segment_tacklesV3.py \
    --video_dir  ${VIDEO_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_mode \
    --batch_id   ${BATCH_ID} \
    --batch_size ${BATCH_SIZE}

echo "Batch ${BATCH_ID} completed at $(date)"