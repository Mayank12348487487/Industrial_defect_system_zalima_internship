import argparse
import os
import sys
from pathlib import Path

# Workaround for OpenMP duplicate runtime issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 model on NEU-DET defect dataset.")
    parser.add_argument(
        "--model",
        type=str,
        default=str(ROOT / "best_industrial_defect.pt"),
        help="Path to YOLOv8 PyTorch model weights (.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(ROOT / "data" / "data.yaml"),
        help="Path to data.yaml dataset configuration",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test", "train"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Compute device (e.g., 'cpu', '0', '0,1')",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    data_yaml = Path(args.data)

    if not model_path.exists():
        print(f"Error: Model file {model_path} not found.")
        sys.exit(1)

    if not data_yaml.exists():
        print(f"Error: Dataset configuration file {data_yaml} not found.")
        sys.exit(1)

    print(f"Loading YOLOv8 model from {model_path}...")
    model = YOLO(str(model_path))

    print(f"Running validation on split '{args.split}' using config {data_yaml}...")
    val_kwargs = {"data": str(data_yaml), "split": args.split, "verbose": True}
    if args.device:
        val_kwargs["device"] = args.device

    results = model.val(**val_kwargs)

    print("\n" + "=" * 50)
    print("GLOBAL METRICS:")
    print(f"mAP50:     {results.results_dict.get('metrics/mAP50(B)', 0.0):.4f}")
    print(f"mAP50-95:  {results.results_dict.get('metrics/mAP50-95(B)', 0.0):.4f}")
    print(f"Precision: {results.results_dict.get('metrics/precision(B)', 0.0):.4f}")
    print(f"Recall:    {results.results_dict.get('metrics/recall(B)', 0.0):.4f}")
    print("=" * 50)

    # Class-wise metrics
    print("\nCLASS-WISE METRICS:")
    if hasattr(results, "box") and hasattr(results.box, "maps"):
        for i, name in model.names.items():
            try:
                class_res = results.box.class_result(i)
                p, r, ap50, ap95 = class_res
                print(f"Class {i} ({name}):")
                print(f"  Precision: {p:.4f}")
                print(f"  Recall:    {r:.4f}")
                print(f"  AP50:      {ap50:.4f}")
                print(f"  AP50-95:   {ap95:.4f}")
            except Exception:
                print(f"Class {i} ({name}): Map value: {results.box.maps[i]:.4f}")
    else:
        print("Detailed class-wise metrics object not accessible, check global metrics above.")

if __name__ == "__main__":
    main()

