"""Helper script to quantize ONNX models to INT8 for Raspberry Pi optimization."""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType
except ImportError:
    logger.error("onnx and onnxruntime required for quantization")
    logger.error("Install with: pip install onnx onnxruntime")
    sys.exit(1)


def quantize_onnx_model(input_path: Path, output_path: Path):
    """
    Quantize an ONNX model to INT8 for faster inference.
    
    Args:
        input_path: Path to input FP32 ONNX model
        output_path: Path to save quantized INT8 model
    """
    if not input_path.exists():
        logger.error(f"Input model not found: {input_path}")
        return False
    
    try:
        logger.info(f"Quantizing model: {input_path}")
        logger.info(f"Output will be saved to: {output_path}")
        
        # Dynamic quantization (simpler, no calibration data needed)
        quantize_dynamic(
            model_input=str(input_path),
            model_output=str(output_path),
            weight_type=QuantType.QUInt8  # Use UINT8 for weights
        )
        
        logger.info(f"✓ Quantization complete: {output_path}")
        
        # Compare file sizes
        input_size = input_path.stat().st_size / (1024 * 1024)  # MB
        output_size = output_path.stat().st_size / (1024 * 1024)  # MB
        reduction = (1 - output_size / input_size) * 100
        
        logger.info(f"Input size: {input_size:.2f} MB")
        logger.info(f"Output size: {output_size:.2f} MB")
        logger.info(f"Size reduction: {reduction:.1f}%")
        
        return True
        
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        return False


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quantize ONNX models to INT8")
    parser.add_argument("input", type=Path, help="Input ONNX model path")
    parser.add_argument("output", type=Path, nargs="?", default=None, 
                       help="Output quantized model path (default: input_quantized.onnx)")
    
    args = parser.parse_args()
    
    if args.output is None:
        # Default: add _quantized before .onnx extension
        output_path = args.input.parent / f"{args.input.stem}_quantized.onnx"
    else:
        output_path = args.output
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    success = quantize_onnx_model(args.input, output_path)
    
    if success:
        logger.info("\n✓ Quantization successful!")
        logger.info(f"Use the quantized model: {output_path}")
        logger.info("\nNote: Quantized models are ~4× smaller and 2-3× faster")
        logger.info("      with minimal accuracy loss (typically <3%)")
    else:
        logger.error("\n✗ Quantization failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
