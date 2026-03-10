#!/usr/bin/env python3
"""
Quick verification script to test Grounded-SAM2 setup
Run this before submitting SLURM jobs
"""

import sys
import subprocess

def print_header(text):
    print("\n" + "="*60)
    print(text)
    print("="*60)

def test_import(module_name, import_statement):
    """Test if a module can be imported"""
    try:
        exec(import_statement)
        print(f"✓ {module_name} import successful")
        return True
    except Exception as e:
        print(f"✗ {module_name} import failed: {e}")
        return False

def test_file_exists(filepath, description):
    """Test if a file exists"""
    import os
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        size_mb = size / (1024 * 1024)
        print(f"✓ {description} exists ({size_mb:.1f} MB)")
        return True
    else:
        print(f"✗ {description} not found: {filepath}")
        return False

def main():
    print_header("GROUNDED-SAM2 SETUP VERIFICATION")
    
    all_passed = True
    
    # Test 1: Python version
    print_header("1. Python Environment")
    print(f"Python version: {sys.version}")
    if sys.version_info >= (3, 10):
        print("✓ Python version >= 3.10")
    else:
        print("✗ Python version should be >= 3.10")
        all_passed = False
    
    # Test 2: Core imports
    print_header("2. Core Dependencies")
    imports = [
        ("PyTorch", "import torch"),
        ("TorchVision", "import torchvision"),
        ("NumPy", "import numpy"),
        ("OpenCV", "import cv2"),
        ("PIL", "from PIL import Image"),
        ("Transformers", "import transformers"),
        ("Supervision", "import supervision"),
        ("tqdm", "from tqdm import tqdm"),
    ]
    
    for name, stmt in imports:
        if not test_import(name, stmt):
            all_passed = False
    
    # Test 3: CUDA
    print_header("3. CUDA Support")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            print(f"GPU name: {torch.cuda.get_device_name(0)}")
            print("✓ CUDA is ready")
        else:
            print("✗ CUDA not available - you need a GPU node!")
            print("  Run: srun -p ksu-gen-gpu.q --gres=gpu:1 --mem=32G --time=1:00:00 --pty bash")
            all_passed = False
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
        all_passed = False
    
    # Test 4: SAM2
    print_header("4. SAM2")
    if test_import("SAM2 video predictor", "from sam2.build_sam import build_sam2_video_predictor"):
        if test_import("SAM2 image predictor", "from sam2.sam2_image_predictor import SAM2ImagePredictor"):
            print("✓ SAM2 is ready")
        else:
            all_passed = False
    else:
        all_passed = False
    
    # Test 5: Grounding DINO
    print_header("5. Grounding DINO")
    if test_import("Grounding DINO", "from groundingdino.util.inference import load_model, predict"):
        print("✓ Grounding DINO is ready")
    else:
        print("✗ Grounding DINO not available")
        print("  Install: pip install git+https://github.com/IDEA-Research/GroundingDINO.git")
        all_passed = False
    
    # Test 6: Model weights
    print_header("6. Model Weights")
    weights = [
        ("weights/sam2.1_hiera_large.pt", "SAM2.1 checkpoint"),
        ("weights/groundingdino_swint_ogc.pth", "Grounding DINO checkpoint"),
        ("weights/GroundingDINO_SwinT_OGC.py", "Grounding DINO config"),
    ]
    
    for filepath, desc in weights:
        if not test_file_exists(filepath, desc):
            all_passed = False
    
    # Test 7: Config files
    print_header("7. Configuration Files")
    if test_file_exists("configs/sam2.1/sam2.1_hiera_l.yaml", "SAM2 config"):
        print("✓ SAM2 config is ready")
    else:
        print("✗ SAM2 config missing - create configs/sam2.1/sam2.1_hiera_l.yaml")
        all_passed = False
    
    # Test 8: Try loading models (if on GPU)
    print_header("8. Model Loading Test")
    try:
        import torch
        if torch.cuda.is_available():
            print("Testing model loading (this may take a minute)...")
            
            from groundingdino.util.inference import load_model
            grounding_model = load_model(
                "weights/GroundingDINO_SwinT_OGC.py",
                "weights/groundingdino_swint_ogc.pth",
                device="cuda"
            )
            print("✓ Grounding DINO model loaded")
            
            from sam2.build_sam import build_sam2_video_predictor
            video_predictor = build_sam2_video_predictor(
                "configs/sam2.1/sam2.1_hiera_l.yaml",
                "weights/sam2.1_hiera_large.pt",
                device="cuda"
            )
            print("✓ SAM2 model loaded")
            
            print("✓ All models load successfully!")
        else:
            print("⚠ Skipping model loading test (no GPU available)")
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        all_passed = False
    
    # Test 9: Directory structure
    print_header("9. Project Structure")
    import os
    required_files = [
        "segment_tacklesV2.py",
        "batch_process_all.py",
        "visualize_results.py",
        "run_test_segmentation.sh",
        "run_batch_all.sh",
    ]
    
    for filename in required_files:
        if not test_file_exists(filename, filename):
            all_passed = False
    
    if not os.path.exists("logs"):
        os.makedirs("logs")
        print("✓ Created logs/ directory")
    
    # Final summary
    print_header("VERIFICATION SUMMARY")
    if all_passed:
        print("✓ ✓ ✓ ALL TESTS PASSED! ✓ ✓ ✓")
        print("\nYou're ready to run segmentation!")
        print("\nNext steps:")
        print("1. Update paths in run_test_segmentation.sh")
        print("2. Submit job: sbatch run_test_segmentation.sh")
    else:
        print("✗ ✗ ✗ SOME TESTS FAILED ✗ ✗ ✗")
        print("\nPlease fix the issues above before proceeding.")
        print("Refer to README.md for troubleshooting.")
    
    print("="*60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
