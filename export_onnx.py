import argparse
import os
import sys
from pathlib import Path

# Workaround for OpenMP duplicate runtime issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8 PyTorch model to ONNX format.")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(ROOT / "best_industrial_defect.pt"),
        help="Path to YOLOv8 .pt weights file",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset version (default: 12)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Model input image dimension (default: 640)",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic axis batch sizing",
    )
    args = parser.parse_args()

    model_path = Path(args.weights)
    if not model_path.exists():
        print(f"Error: Model file {model_path} not found.")
        sys.exit(1)

    print(f"Loading YOLOv8 model from {model_path}...")
    model = YOLO(str(model_path))

    print(f"Exporting model to ONNX format (opset={args.opset}, imgsz={args.imgsz}, dynamic={args.dynamic})...")
    onnx_path = model.export(
        format="onnx",
        opset=args.opset,
        imgsz=args.imgsz,
        dynamic=args.dynamic,
        simplify=True,
    )
    print(f"\nModel exported successfully to: {onnx_path}")

if __name__ == "__main__":
    main()

