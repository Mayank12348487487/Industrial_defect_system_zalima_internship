import json
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "data" / "data.yaml"

def calibrate(weights_path="best_industrial_defect.pt", imgsz=640):
    print(f"Calibrating class-wise confidence thresholds for {weights_path}...")
    model = YOLO(weights_path)
    
    # Run validation with a low conf threshold to get all candidate predictions
    val_res = model.val(data=str(DATA_YAML), split="val", imgsz=imgsz, conf=0.05, iou=0.6, device="0" if torch.cuda.is_available() else "cpu", workers=0)
    
    # Class mapping
    names = model.names
    print("\nClass names:", names)
    
    # Let's inspect F1 curve data from val_res if available
    class_best_conf = {}
    if hasattr(val_res, "box") and hasattr(val_res.box, "f1"):
        # val_res.box.f1 is shape (nc, 1000) or similar
        # val_res.box.x is confidence thresholds
        f1_array = np.array(val_res.box.f1)
        confs = np.array(val_res.box.x) if hasattr(val_res.box, "x") else np.linspace(0, 1, f1_array.shape[-1])
        
        print(f"F1 array shape: {f1_array.shape}, Confs shape: {confs.shape}")
        print("\nOptimal F1 confidence thresholds per class:")
        if f1_array.ndim == 2:
            for i, name in names.items():
                if i < f1_array.shape[0]:
                    curve = f1_array[i]
                    best_idx = int(np.argmax(curve))
                    best_conf = float(confs[best_idx])
                    max_f1 = float(curve[best_idx])
                    class_best_conf[i] = {
                        "class": name,
                        "best_conf": round(best_conf, 3),
                        "max_f1": round(max_f1, 4)
                    }
                    print(f"  Class {i} ({name:<18}): Best Conf = {best_conf:.3f}, Max F1 = {max_f1*100:.2f}%")
        elif f1_array.ndim == 1:
            best_idx = int(np.argmax(f1_array))
            best_conf = float(confs[best_idx])
            max_f1 = float(f1_array[best_idx])
            print(f"  Overall Best Conf = {best_conf:.3f}, Max F1 = {max_f1*100:.2f}%")
            for i, name in names.items():
                class_best_conf[i] = {"class": name, "best_conf": round(best_conf, 3), "max_f1": round(max_f1, 4)}
    
    # Save results
    out = ROOT / "scratch" / "calibrated_thresholds.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(class_best_conf, f, indent=2)
    print(f"\nSaved calibrated thresholds to: {out}")
    return class_best_conf

if __name__ == "__main__":
    calibrate()
