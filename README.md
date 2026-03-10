# GRAZE: Grounded Refinement and Motion-Aware Zero-Shot Event Localization

[![CVSports 2026](https://img.shields.io/badge/CVSports-CVPR%202026-blue)]()
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of **GRAZE**, accepted at **CVSports Workshop @ CVPR 2026**.

GRAZE is a **training-free** pipeline for **First Point of Contact (FPOC) detection** in American football tackle videos. It combines open-vocabulary grounding (GroundingDINO), promptable segmentation (SAM2), and motion-aware temporal reasoning — no task-specific training required.

---

## Method

<p align="center">
  <img src="https://github.com/user-attachments/assets/2c6192b9-a354-46c9-bba2-80f5f000ea5e" width="900"/>
  <br>
  <em>GRAZE pipeline: text-prompted grounding → SAM2 segmentation → multi-prompt temporal search → motion-aware backward refinement → FPOC prediction</em>
</p>

---

## Repository Structure

```
GRAZE/
├── segment_tacklesV3.py       # ★ GRAZE pipeline (full method, matches paper results)
├── CombineXls.py              # Aggregates per-video predictions to Excel
├── verify_setup.py            # Checks weights and environment
├── setupenv.sh                # Conda environment setup
├── submit.sh                  # SLURM single-job submission
├── run_array.sh               # SLURM array job (one job per video)
├── configs/                   # SAM2 model configuration files
│   └── sam2.1/
└── requirements.txt
```

---

## Installation

### 1. Clone
```bash
git clone https://github.com/AhsanZaidi12/GRAZE.git
cd GRAZE
```

### 2. Create environment
```bash
bash setupenv.sh
# OR manually:
conda create -n graze python=3.10 -y
conda activate graze
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 3. Install GroundingDINO
```bash
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
```

### 4. Install SAM2
```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

### 5. Download weights
```bash
mkdir -p weights

# GroundingDINO
wget -P weights/ https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget -P weights/ https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py

# SAM2 Large
wget -P weights/ https://dl.fbaipublicfiles.com/segment_anything_2/sam2_hiera_large.pt
```

### 6. Verify
```bash
python verify_setup.py
```

---

## Usage

### Single video
```bash
conda activate graze

python segment_tacklesV3.py \
  --video_path /path/to/clip.mp4 \
  --output_dir ./Results \
  --weights_dir ./weights
```

### Batch inference — SLURM (HPC)
```bash
# Edit submit.sh: set your video directory and SLURM allocation
sbatch submit.sh

# Large-scale array job (used for 738 clips in paper)
sbatch run_array.sh
```

---

## Qualitative Results

<p align="center">
  <img src="https://github.com/user-attachments/assets/82a8fc38-d37b-47fa-9578-acfe14614130" width="900"/>
  <br>
  <em>FPOC detection examples: GRAZE correctly localizes the first point of contact frame across diverse tackle scenarios in practice footage</em>
</p>

---

## Aggregating Results

```bash
python CombineXls.py --results_dir ./Results --output graze_results.xlsx
```

---

## Dataset

The dataset contains 738 annotated football practice tackle clips with FPOC ground truth labeled using the SATT rubric. Raw video data cannot be released due to athlete privacy and institutional agreements. Contact `ahsanzaidi@ksu.edu` for research access.

---

## Citation

```bibtex
@inproceedings{zaidi2026graze,
  title     = {Grounded Refinement and Motion-Aware Zero-Shot Event Localization},
  author    = {Zaidi, Ahsan and Hsu, William and Dietrich, Scott},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVSports)},
  year      = {2026}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).  
GroundingDINO and SAM2 are Apache 2.0 licensed by their respective authors.
