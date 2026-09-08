import json
import os
import sys
from pathlib import Path
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "data" / "data.yaml"

def evaluate(weights_path, imgsz=640, device="0" if torch.cuda.is_available() else "cpu"):
    print(f"Evaluating: {weights_path} on split 'val' (imgsz={imgsz}, device={device})...")
    model = YOLO(str(weights_path))
    val_res = model.val(data=str(DATA_YAML), split="val", imgsz=imgsz, device=device, batch=8, workers=0)
    
    results = {
        "weights": str(weights_path),
        "imgsz": imgsz,
        "mAP50": round(float(val_res.results_dict.get("metrics/mAP50(B)", 0.0)), 4),
        "mAP50-95": round(float(val_res.results_dict.get("metrics/mAP50-95(B)", 0.0)), 4),
        "precision": round(float(val_res.results_dict.get("metrics/precision(B)", 0.0)), 4),
        "recall": round(float(val_res.results_dict.get("metrics/recall(B)", 0.0)), 4),
        "speed": val_res.speed
    }
    
    # Calculate F1
    p = results["precision"]
    r = results["recall"]
    results["f1"] = round(2 * p * r / (p + r + 1e-9), 4)
    
    class_metrics = {}
    if hasattr(val_res, "box") and hasattr(val_res.box, "maps"):
        for i, name in model.names.items():
            try:
                ap50 = float(val_res.box.ap50[i]) if hasattr(val_res.box, "ap50") and len(val_res.box.ap50) > i else 0.0
                ap = float(val_res.box.maps[i]) if len(val_res.box.maps) > i else 0.0
                p_c = float(val_res.box.p[i]) if hasattr(val_res.box, "p") and len(val_res.box.p) > i else 0.0
                r_c = float(val_res.box.r[i]) if hasattr(val_res.box, "r") and len(val_res.box.r) > i else 0.0
                f1_c = round(2 * p_c * r_c / (p_c + r_c + 1e-9), 4)
                class_metrics[name] = {
                    "precision": round(p_c, 4),
                    "recall": round(r_c, 4),
                    "f1": f1_c,
                    "AP50": round(ap50, 4),
                    "AP50-95": round(ap, 4)
                }
            except Exception as e:
                class_metrics[name] = {"error": str(e)}
    results["classes"] = class_metrics
    
    print("\n" + "=" * 60)
    print(f"RESULTS FOR: {Path(weights_path).name}")
    print(f"mAP@50:    {results['mAP50'] * 100:.2f}%")
    print(f"mAP@50-95: {results['mAP50-95'] * 100:.2f}%")
    print(f"Precision: {results['precision'] * 100:.2f}%")
    print(f"Recall:    {results['recall'] * 100:.2f}%")
    print(f"F1 Score:  {results['f1'] * 100:.2f}%")
    print(f"Speed: {results['speed']}")
    print("-" * 60)
    print(f"{'Class':<20} | {'AP@50':<8} | {'AP@50-95':<10} | {'P':<8} | {'R':<8} | {'F1':<8}")
    print("-" * 60)
    for cls_name, cm in class_metrics.items():
        print(f"{cls_name:<20} | {cm.get('AP50', 0)*100:6.2f}% | {cm.get('AP50-95', 0)*100:8.2f}% | {cm.get('precision', 0)*100:6.2f}% | {cm.get('recall', 0)*100:6.2f}% | {cm.get('f1', 0)*100:6.2f}%")
    print("=" * 60)
    return results

if __name__ == "__main__":
    weights = sys.argv[1] if len(sys.argv) > 1 else "best_industrial_defect_baseline_69.7.pt"
    imgsz = int(sys.argv[2]) if len(sys.argv) > 2 else 640
    res = evaluate(weights, imgsz=imgsz)
    out_file = ROOT / "scratch" / f"eval_{Path(weights).stem}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved evaluation metrics to: {out_file}")
