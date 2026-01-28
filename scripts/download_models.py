"""Script to download and convert ML models for lane detection."""

import os
import sys
import logging
from pathlib import Path
import urllib.request
import hashlib

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"
FAST_SCNN_DIR = MODELS_DIR / "fast_scnn"
YOLO_DIR = MODELS_DIR / "yolo"


def download_file(url: str, dest_path: Path, expected_hash: str = None):
    """Download a file from URL to destination path."""
    logger.info(f"Downloading {url} to {dest_path}")
    
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        urllib.request.urlretrieve(url, dest_path)
        logger.info(f"Downloaded: {dest_path}")
        
        # Verify hash if provided
        if expected_hash:
            with open(dest_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            if file_hash != expected_hash:
                logger.warning(f"Hash mismatch for {dest_path}. Expected: {expected_hash}, Got: {file_hash}")
            else:
                logger.info(f"Hash verified for {dest_path}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def check_onnx_available():
    """Check if ONNX conversion tools are available."""
    try:
        import onnx
        import torch
        return True
    except ImportError:
        return False


def convert_pytorch_to_onnx(model_path: Path, output_path: Path, input_size: tuple):
    """
    Convert PyTorch model to ONNX format.
    
    Note: This requires PyTorch and the original model architecture.
    For production, pre-converted ONNX models should be provided.
    """
    try:
        import torch
        import onnx
        
        logger.info(f"Converting {model_path} to ONNX...")
        logger.warning("ONNX conversion requires the original model architecture code.")
        logger.warning("For production use, download pre-converted ONNX models.")
        logger.warning("This is a placeholder - actual conversion requires model-specific code.")
        
        # Placeholder - actual conversion would require:
        # 1. Loading the PyTorch model with its architecture
        # 2. Creating dummy input
        # 3. Exporting to ONNX
        # 4. Quantizing to INT8
        
        return False
    except ImportError:
        logger.error("PyTorch or ONNX not available. Install with: pip install torch onnx")
        return False
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        return False


def download_fast_scnn():
    """Download Fast SCNN model."""
    logger.info("Fast SCNN model download")
    logger.info("=" * 50)
    
    # Fast SCNN models are typically available from:
    # - GitHub repositories (antoniojkim/Fast-SCNN, zacario-li/Fast-SCNN_pytorch)
    # - Pre-trained on Cityscapes dataset
    
    model_path = FAST_SCNN_DIR / "fast_scnn.pth"
    onnx_path = FAST_SCNN_DIR / "fast_scnn_quantized.onnx"
    
    if onnx_path.exists():
        logger.info(f"ONNX model already exists: {onnx_path}")
        return True
    
    # Placeholder URLs - these would need to be actual model download links
    logger.warning("Fast SCNN model download URLs need to be configured.")
    logger.warning("Options:")
    logger.warning("1. Download from GitHub: https://github.com/antoniojkim/Fast-SCNN")
    logger.warning("2. Train your own model on lane detection dataset")
    logger.warning("3. Use pre-trained Cityscapes model and fine-tune")
    
    # Example structure (would need actual URLs):
    # pytorch_url = "https://github.com/user/repo/releases/download/v1.0/fast_scnn.pth"
    # if download_file(pytorch_url, model_path):
    #     if check_onnx_available():
    #         convert_pytorch_to_onnx(model_path, onnx_path, (320, 240))
    
    logger.info("To use Fast SCNN:")
    logger.info("1. Download a pre-trained Fast SCNN model")
    logger.info("2. Convert to ONNX format (use provided conversion script)")
    logger.info("3. Quantize to INT8 (use onnxruntime quantization tools)")
    logger.info("4. Place the quantized ONNX model at: models/fast_scnn/fast_scnn_quantized.onnx")
    
    return False


def download_yolo():
    """Download YOLOv8 segmentation model."""
    logger.info("YOLOv8 model download")
    logger.info("=" * 50)
    
    model_path = YOLO_DIR / "yolov8n_lane_seg.pt"
    onnx_path = YOLO_DIR / "yolov8n_lane_seg_quantized.onnx"
    onnx_unquantized_path = YOLO_DIR / "yolov8n_lane_seg.onnx"
    
    if onnx_path.exists():
        logger.info(f"Quantized ONNX model already exists: {onnx_path}")
        return True
    
    # YOLOv8 can be downloaded using ultralytics
    try:
        from ultralytics import YOLO
        
        logger.info("Downloading YOLOv8n segmentation model...")
        model = YOLO('yolov8n-seg.pt')  # Downloads automatically
        
        # Check if ONNX is available
        try:
            import onnx
            logger.info("ONNX package found, exporting to ONNX...")
        except ImportError:
            logger.error("ONNX package not found!")
            logger.error("Install it using: uv pip install onnx")
            logger.error("Or: pip install onnx (if not using uv)")
            logger.info("")
            logger.info("For now, saving PyTorch model. You can export to ONNX later:")
            logger.info(f"  python -c \"from ultralytics import YOLO; YOLO('{model_path}').export(format='onnx', imgsz=416)\"")
            # Save PyTorch model
            import shutil
            if not model_path.exists():
                shutil.move('yolov8n-seg.pt', model_path)
            return False
        
        # Export to ONNX
        logger.info("Exporting to ONNX...")
        onnx_model_path = model.export(format='onnx', imgsz=416, simplify=True)
        
        # Move to our models directory
        import shutil
        if onnx_unquantized_path.exists():
            logger.info(f"Unquantized ONNX model already exists: {onnx_unquantized_path}")
        else:
            shutil.move(onnx_model_path, onnx_unquantized_path)
            logger.info(f"ONNX model saved to: {onnx_unquantized_path}")
        
        # Try to quantize
        logger.info("Attempting to quantize model...")
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            quantize_dynamic(
                model_input=str(onnx_unquantized_path),
                model_output=str(onnx_path),
                weight_type=QuantType.QUInt8
            )
            logger.info(f"✓ Quantized model saved to: {onnx_path}")
        except Exception as quantize_error:
            logger.warning(f"Quantization failed: {quantize_error}")
            logger.warning("You can quantize manually using: python scripts/quantize_model.py")
            logger.info(f"Using unquantized model: {onnx_unquantized_path}")
            # Update config to use unquantized model if quantized doesn't exist
            onnx_path = onnx_unquantized_path
        
        return True
        
    except ImportError as e:
        logger.error(f"ultralytics not available: {e}")
        logger.error("Install dependencies using: uv pip install ultralytics onnx")
        logger.info("Alternative: Download pre-converted ONNX model manually")
        return False
    except Exception as e:
        logger.error(f"YOLOv8 download/export failed: {e}")
        logger.error("Make sure you have installed: uv pip install ultralytics onnx onnxruntime")
        return False


def create_model_config():
    """Create model configuration files."""
    fast_scnn_config = FAST_SCNN_DIR / "config.json"
    if not fast_scnn_config.exists():
        import json
        config = {
            "model_type": "fast_scnn",
            "input_size": [320, 240],
            "num_classes": 19,  # Cityscapes
            "lane_class": 6,  # Road class
            "description": "Fast SCNN for lane detection"
        }
        with open(fast_scnn_config, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Created config: {fast_scnn_config}")
    
    yolo_config = YOLO_DIR / "config.json"
    if not yolo_config.exists():
        import json
        config = {
            "model_type": "yolov8n_seg",
            "input_size": [416, 416],
            "confidence_threshold": 0.25,
            "description": "YOLOv8n segmentation for lane detection"
        }
        with open(yolo_config, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Created config: {yolo_config}")


def main():
    """Main function to download and prepare models."""
    logger.info("ML Model Download and Setup")
    logger.info("=" * 50)
    
    # Check dependencies
    logger.info("\nChecking dependencies...")
    missing_deps = []
    try:
        import onnx
    except ImportError:
        missing_deps.append("onnx")
    
    try:
        from ultralytics import YOLO
    except ImportError:
        missing_deps.append("ultralytics")
    
    if missing_deps:
        logger.warning(f"Missing dependencies: {', '.join(missing_deps)}")
        logger.info("Install them using: uv pip install " + " ".join(missing_deps))
        logger.info("Or if not using uv: pip install " + " ".join(missing_deps))
        logger.info("")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            logger.info("Exiting. Install dependencies and run again.")
            return
    
    # Create directories
    MODELS_DIR.mkdir(exist_ok=True)
    FAST_SCNN_DIR.mkdir(exist_ok=True)
    YOLO_DIR.mkdir(exist_ok=True)
    
    # Create config files
    create_model_config()
    
    # Download models
    logger.info("\n")
    fast_scnn_ok = download_fast_scnn()
    
    logger.info("\n")
    yolo_ok = download_yolo()
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("Download Summary:")
    logger.info(f"Fast SCNN: {'✓' if fast_scnn_ok else '✗ (manual setup required)'}")
    logger.info(f"YOLOv8: {'✓' if yolo_ok else '✗ (manual setup required)'}")
    
    if not (fast_scnn_ok and yolo_ok):
        logger.info("\nNote: Some models require manual setup.")
        logger.info("See README.md for detailed instructions.")
    
    if not yolo_ok and missing_deps:
        logger.info("\nTo fix YOLOv8 download:")
        logger.info("  1. Install dependencies: uv pip install ultralytics onnx onnxruntime")
        logger.info("  2. Run this script again: python scripts/download_models.py")


if __name__ == "__main__":
    main()
