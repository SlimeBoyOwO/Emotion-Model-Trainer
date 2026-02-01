#!/usr/bin/env python3
"""
Quantize ONNX models in the root directory and place quantized models
and inference files in ./onnxmodel directory.
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path

try:
    import onnx
    import onnxruntime as ort
except ImportError as e:
    print(f"Missing required packages: {e}")
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "onnx", "onnxruntime"])
    import onnx
    import onnxruntime as ort


def find_onnx_files(root_dir="."):
    """Find all .onnx files in the root directory (not in subdirectories)."""
    root_path = Path(root_dir)
    onnx_files = list(root_path.glob("*.onnx"))
    # Also check for emotion_model_tokenizer directory if it exists
    tokenizer_dir = root_path / "emotion_model_tokenizer"
    if tokenizer_dir.exists():
        onnx_files.extend(tokenizer_dir.glob("*.onnx"))
    return onnx_files


def copy_inference_files(src_dir, dst_dir):
    """Copy tokenizer and config files needed for inference."""
    dst_path = Path(dst_dir)
    dst_path.mkdir(parents=True, exist_ok=True)

    # Files to copy
    patterns = ["*.json", "*.txt", "*.vocab", "*.model", "config.json", "tokenizer.json",
                "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "vocab.json"]

    # Look in current directory and emotion_model_tokenizer
    search_dirs = [Path("."), Path("emotion_model_tokenizer")]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for pattern in patterns:
            for file in search_dir.glob(pattern):
                if file.is_file():
                    dst_file = dst_path / file.name
                    # Avoid overwriting if already copied from another location
                    if not dst_file.exists():
                        shutil.copy2(file, dst_file)
                        print(f"Copied inference file: {file} -> {dst_file}")


def quantize_model(model_path, output_dir):
    """Quantize an ONNX model using dynamic quantization."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output filename
    quantized_name = model_path.stem + "_quantized" + model_path.suffix
    quantized_path = output_dir / quantized_name

    print(f"Quantizing {model_path}...")

    try:
        # Dynamic quantization
        quantize_dynamic(
            model_input=model_path,
            model_output=quantized_path,
            weight_type=QuantType.QUInt8  # Use uint8 for weights
        )
        print(f"Quantized model saved to: {quantized_path}")
        return quantized_path
    except Exception as e:
        print(f"Failed to quantize {model_path}: {e}")
        return None


def test_model(model_path):
    """Test if a model can be loaded and run inference."""
    try:
        # Load model
        sess = ort.InferenceSession(str(model_path))

        # Get input/output info
        inputs = sess.get_inputs()
        outputs = sess.get_outputs()

        print(f"Model loaded successfully: {model_path}")
        print(f"  Inputs: {[i.name for i in inputs]}")
        print(f"  Outputs: {[o.name for o in outputs]}")

        # Create dummy input based on input shape
        # Use small random data
        import numpy as np
        input_data = {}
        for inp in inputs:
            # Replace any dynamic dimensions with 1
            shape = []
            for dim in inp.shape:
                if isinstance(dim, str) or dim < 0:
                    shape.append(1)
                else:
                    shape.append(dim)
            # Use appropriate data type
            if inp.type == 'tensor(float)':
                dtype = np.float32
            elif inp.type == 'tensor(int64)':
                dtype = np.int64
            elif inp.type == 'tensor(int32)':
                dtype = np.int32
            else:
                dtype = np.float32  # default

            input_data[inp.name] = np.random.randn(*shape).astype(dtype)

        # Run inference
        output = sess.run(None, input_data)
        print(f"  Inference successful, output shape: {[o.shape for o in output]}")
        return True

    except Exception as e:
        print(f"Failed to test model {model_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Quantize ONNX models")
    parser.add_argument("--input-dir", default=".", help="Directory containing ONNX models")
    parser.add_argument("--output-dir", default="./onnxmodel", help="Output directory for quantized models")
    parser.add_argument("--skip-quantize", action="store_true", help="Skip quantization, only copy files and test")
    parser.add_argument("--skip-test", action="store_true", help="Skip testing quantized models")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find ONNX files
    onnx_files = find_onnx_files(args.input_dir)
    if not onnx_files:
        print("No ONNX files found.")
        return 1

    print(f"Found {len(onnx_files)} ONNX file(s):")
    for f in onnx_files:
        print(f"  {f}")

    # Copy inference files
    print("\nCopying inference files...")
    copy_inference_files(args.input_dir, args.output_dir)

    # Quantize models
    quantized_models = []
    if not args.skip_quantize:
        print("\nQuantizing models...")
        for model_file in onnx_files:
            quantized = quantize_model(model_file, args.output_dir)
            if quantized:
                quantized_models.append(quantized)
    else:
        # Use existing quantized models in output directory
        quantized_models = list(output_dir.glob("*quantized*.onnx"))
        if not quantized_models:
            # Fallback to original models
            quantized_models = [output_dir / f.name for f in onnx_files]

    # Test models
    if not args.skip_test:
        print("\nTesting models...")
        for model_file in quantized_models:
            if model_file.exists():
                print(f"\nTesting {model_file.name}:")
                test_model(model_file)
            else:
                print(f"Model not found: {model_file}")

    print(f"\nAll files have been placed in: {output_dir.absolute()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())