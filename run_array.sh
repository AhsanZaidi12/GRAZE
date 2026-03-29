#!/bin/bash
#SBATCH --job-name=graze_seg
#SBATCH --partition=GPU-shared          # Bridges-2: GPU-shared | Beocat: ksu-gen-gpu.q
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1                    # Generic; on Bridges-2 use: gpu:h100-80:1
#SBATCH --output=logs/graze_%A_%a.out
#SBATCH --error=logs/graze_%A_%a.err
#SBATCH --array=0-3                     # 4 batches; adjust if needed

# ─── USER CONFIGURATION ────────────────────────────────────────────────────────
# Set these before submitting. Do not hardcode paths.
VIDEO_DIR="${VIDEO_DIR:-/path/to/your/videos}"
OUTPUT_DIR="${OUTPUT_DIR:-./Results}"
GRAZE_DIR="${GRAZE_DIR:-$PWD}"
CONDA_ENV="${CONDA_ENV:-graze}"
# ───────────────────────────────────────────────────────────────────────────────

export CUDA_VISIBLE_DEVICES=0

TOTAL_VIDEOS=$(ls -1 ${VIDEO_DIR}/*.{mp4,MP4,avi,AVI,mov,MOV} 2>/dev/null | wc -l)
BATCH_SIZE=$(( (TOTAL_VIDEOS + 3) / 4 ))
BATCH_ID=${SLURM_ARRAY_TASK_ID}

mkdir -p ${OUTPUT_DIR} logs

source ~/.bashrc
conda activate ${CONDA_ENV}
cd ${GRAZE_DIR}

echo "=========================================="
echo "GRAZE Array Job — Batch ${BATCH_ID}/3"
echo "Node: ${SLURM_NODELIST} | Job: ${SLURM_JOB_ID}"
echo "Videos: ${TOTAL_VIDEOS} | Batch size: ${BATCH_SIZE}"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start: $(date)"
echo "=========================================="

python segment_tacklesV3.py \
  --video_dir ${VIDEO_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --batch_mode \
  --batch_id ${BATCH_ID} \
  --batch_size ${BATCH_SIZE}

echo "Batch ${BATCH_ID} completed at $(date)"
