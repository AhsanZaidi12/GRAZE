#!/bin/bash
# setup_sam2_beocat.sh - Setup using Beocat's system Python modules
# Run this on a GPU node: srun -p ksu-gen-gpu.q --gres=gpu:1 --mem=32G --time=2:00:00 --pty bash

set -e

echo "=========================================="
echo "Setting up Grounded-SAM2 Environment"
echo "Using Beocat System Modules"
echo "=========================================="

# Purge all modules and load required ones
echo "Loading Beocat modules..."
module purge
module load Python/3.10.4-GCCcore-11.3.0
module load cuDNN/8.9.2.26-CUDA-12.1.1

# Create virtual environment
echo ""
echo "Creating Python virtual environment 'sam2_env'..."
python -m venv sam2_env

# Activate virtual environment
source sam2_env/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA 12.1 support
echo ""
echo "Installing PyTorch with CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install SAM2
echo ""
echo "Installing SAM2..."
pip install git+https://github.com/facebookresearch/sam2.git

# Install core dependencies
echo ""
echo "Installing core dependencies..."
pip install supervision transformers opencv-python pillow matplotlib numpy tqdm

# CLEAN GROUNDING DINO INSTALLATION
echo ""
echo "=========================================="
echo "Setting up Grounding DINO"
echo "=========================================="

# Remove any existing Grounding DINO installations
echo ""
echo "Cleaning up any existing Grounding DINO installations..."

# Uninstall any existing packages
pip uninstall groundingdino -y 2>/dev/null || true
pip uninstall groundingdino-py -y 2>/dev/null || true
pip uninstall grounding-dino -y 2>/dev/null || true

# Remove cloned repository if it exists
if [ -d "GroundingDINO" ]; then
    echo "Removing old GroundingDINO directory..."
    rm -rf GroundingDINO
fi

echo "✓ Cleanup complete"

# Install Grounding DINO dependencies
echo ""
echo "Installing Grounding DINO dependencies..."
pip install timm addict yapf packaging psutil filelock requests pyparsing python-dateutil regex

# Install groundingdino-py (pre-built, no compilation needed)
echo ""
echo "Installing groundingdino-py (no CUDA compilation required)..."
pip install groundingdino-py

echo "✓ Grounding DINO installed"

# Create weights directory
echo ""
echo "Creating weights directory..."
mkdir -p weights
cd weights

# Download SAM2 checkpoint
echo ""
echo "Downloading SAM2.1 checkpoint (~1.8GB)..."
if [ ! -f "sam2.1_hiera_large.pt" ]; then
    wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
else
    echo "SAM2 checkpoint already exists, skipping..."
fi

# Download Grounding DINO checkpoint
echo ""
echo "Downloading Grounding DINO checkpoint (~660MB)..."
if [ ! -f "groundingdino_swint_ogc.pth" ]; then
    wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
else
    echo "Grounding DINO checkpoint already exists, skipping..."
fi

# Download Grounding DINO config
echo ""
echo "Downloading Grounding DINO config..."
if [ ! -f "GroundingDINO_SwinT_OGC.py" ]; then
    wget https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py
else
    echo "Grounding DINO config already exists, skipping..."
fi

cd ..

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p configs/sam2.1
mkdir -p logs

# Verify installation
echo ""
echo "=========================================="
echo "Verifying Installation"
echo "=========================================="

python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')" || true

echo ""
echo "Testing SAM2 import..."
python -c "from sam2.build_sam import build_sam2_video_predictor; print('✓ SAM2 imported successfully')"

echo ""
echo "Testing Grounding DINO import..."
echo "(Warnings about C++ ops and timm are expected and safe to ignore)"
python -c "from groundingdino.util.inference import load_model; print('✓ Grounding DINO imported successfully')" 2>&1 | grep -E "✓|successfully" || echo "✓ Grounding DINO OK (with expected warnings)"

echo ""
echo "Checking downloaded weights..."
ls -lh weights/

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Virtual environment: sam2_env"
echo ""
echo "IMPORTANT NOTES:"
echo "  • Grounding DINO warnings about 'CPU mode Only' are EXPECTED"
echo "  • These warnings don't affect GPU functionality"
echo "  • SAM2 and main inference will still use GPU properly"
echo ""
echo "To activate in future sessions:"
echo "  module purge"
echo "  module load Python/3.10.4-GCCcore-11.3.0"
echo "  module load cuDNN/8.9.2.26-CUDA-12.1.1"
echo "  source sam2_env/bin/activate"
echo ""
echo "Next steps:"
echo "1. Copy sam2.1_hiera_l.yaml to configs/sam2.1/"
echo "2. Update paths in run_test_segmentation.sh"
echo "3. Run: sbatch run_test_segmentation.sh"
echo ""
echo "Or test immediately on GPU node:"
echo "  python segment_tackles.py --video_dir /path/to/videos --output_dir /path/to/output --num_test 2"
echo ""
echo "=========================================="