import argparse
import os
import sys
import json
from pathlib import Path

# Workaround for OpenMP duplicate runtime issue on Windows and MLFlow auto-logging
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["ULTRALYTICS_MLFLOW"] = "false"
os.environ["MLFLOW_TRACKING_URI"] = ""

import torch
from ultralytics import YOLO, settings
import ultralytics.data.augment as aug_module

# Bypass albumentations to prevent GIL CPU bottleneck on Windows dataloader
class DummyAlbumentations:
    def __init__(self, p=1.0, transforms=None):
        self.transform = None
    def __call__(self, labels):
        return labels
    def __repr__(self):
        return "DummyAlbumentations()"

aug_module.Albumentations = DummyAlbumentations

# Disable mlflow integration to avoid tracking backend exception
try:
    settings.update({"mlflow": False, "clearml": False, "comet": False, "tensorboard": False, "wandb": False})
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "data" / "data.yaml"


def train_experiment(
    model_name: str = "yolov8s.pt",
    data_yaml: Path = DATA_YAML,
    imgsz: int = 640,
    epochs: int = 70,
    batch: int = 16,
    lr0: float = 0.002,
    lrf: float = 0.01,
    optimizer: str = "AdamW",
    weight_decay: float = 0.0005,
    warmup_epochs: float = 3.0,
    box_loss_gain: float = 7.5,
    cls_loss_gain: float = 1.2,
    dfl_loss_gain: float = 1.5,
    experiment_name: str = "exp_yolov8s_improved",
    device: str = "",
    seed: int = 42,
):
    print("=" * 60)
    print(f"STARTING EXPERIMENT: {experiment_name}")
    print(f"Model: {model_name} | Imgsz: {imgsz} | Epochs: {epochs} | Batch: {batch}")
    print(f"Optimizer: {optimizer} | lr0: {lr0} | lrf: {lrf} | weight_decay: {weight_decay}")
    print(f"Loss Gains: box={box_loss_gain}, cls={cls_loss_gain}, dfl={dfl_loss_gain}")
    print("=" * 60)

    # Determine device
    if not device:
        device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Using compute device: {device} (CUDA Available: {torch.cuda.is_available()})")

    # Load model
    model = YOLO(model_name)

    # Training parameters with industrial defect augmentations
    train_args = {
        "data": str(data_yaml),
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "device": device,
        "workers": 0,
        "cache": False,
        "patience": 20,
        "name": experiment_name,
        "exist_ok": True,
        "optimizer": optimizer,
        "lr0": lr0,
        "lrf": lrf,
        "weight_decay": weight_decay,
        "warmup_epochs": warmup_epochs,
        "cos_lr": True,
        "close_mosaic": 10,
        "seed": seed,
        "deterministic": True,
        "box": box_loss_gain,
        "cls": cls_loss_gain,
        "dfl": dfl_loss_gain,
        # Targeted augmentations for metal surfaces:
        "hsv_h": 0.015,
        "hsv_s": 0.4,
        "hsv_v": 0.3,
        "degrees": 10.0,      # slight rotation
        "translate": 0.1,     # translation
        "scale": 0.5,         # scaling
        "shear": 2.0,         # shear
        "perspective": 0.0005,
        "fliplr": 0.5,        # horizontal flip
        "flipud": 0.5,        # vertical flip (metal defects have 2D symmetry)
        "mosaic": 1.0,        # mosaic
        "mixup": 0.1,         # mixup for complex backgrounds
        "erasing": 0.2,       # random erasing for occlusion robustness
        "plots": True,
        "save": True,
        "val": True,
    }

    results = model.train(**train_args)
    print("\nTraining completed successfully.")

    # Validation on standard val split
    print("\nEvaluating best checkpoint on validation set...")
    best_weights_path = ROOT / "runs" / "detect" / experiment_name / "weights" / "best.pt"
    if not best_weights_path.exists():
        print(f"Warning: {best_weights_path} not found, validating current model object.")
        val_model = model
    else:
        val_model = YOLO(str(best_weights_path))

    val_res = val_model.val(data=str(data_yaml), split="val", imgsz=imgsz, batch=batch, device=device)

    metrics = {
        "experiment": experiment_name,
        "model": model_name,
        "imgsz": imgsz,
        "mAP50": round(float(val_res.results_dict.get("metrics/mAP50(B)", 0.0)), 4),
        "mAP50-95": round(float(val_res.results_dict.get("metrics/mAP50-95(B)", 0.0)), 4),
        "precision": round(float(val_res.results_dict.get("metrics/precision(B)", 0.0)), 4),
        "recall": round(float(val_res.results_dict.get("metrics/recall(B)", 0.0)), 4),
    }

    class_metrics = {}
    if hasattr(val_res, "box") and hasattr(val_res.box, "maps"):
        for i, name in val_model.names.items():
            try:
                p, r, ap50, ap95 = val_res.box.class_result(i)
                class_metrics[name] = {
                    "precision": round(float(p), 4),
                    "recall": round(float(r), 4),
                    "ap50": round(float(ap50), 4),
                    "ap50_95": round(float(ap95), 4),
                }
            except Exception:
                class_metrics[name] = {"map": round(float(val_res.box.maps[i]), 4)}

    metrics["class_metrics"] = class_metrics

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS SUMMARY:")
    print(json.dumps(metrics, indent=2))
    print("=" * 60)

    # Save summary json
    summary_path = ROOT / "runs" / "detect" / experiment_name / "eval_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train improved YOLOv8 on NEU Metal Surface Defect dataset.")
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="Base model weights (e.g. yolov8s.pt, yolov8m.pt)")
    parser.add_argument("--data", type=str, default=str(DATA_YAML), help="Path to data.yaml")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--epochs", type=int, default=70, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--lr0", type=float, default=0.002, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final learning rate factor")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer (AdamW, SGD)")
    parser.add_argument("--name", type=str, default="exp_yolov8s_improved", help="Experiment name")
    parser.add_argument("--device", type=str, default="", help="Device: '0' for CUDA GPU, 'cpu'")
    parser.add_argument("--cls-gain", type=float, default=1.2, help="Classification loss gain")
    parser.add_argument("--box-gain", type=float, default=7.5, help="Box loss gain")
    args = parser.parse_args()

    train_experiment(
        model_name=args.model,
        data_yaml=Path(args.data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        lr0=args.lr0,
        lrf=args.lrf,
        optimizer=args.optimizer,
        cls_loss_gain=args.cls_gain,
        box_loss_gain=args.box_gain,
        experiment_name=args.name,
        device=args.device,
    )


if __name__ == "__main__":
    main()
