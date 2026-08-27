import os
import sys

# Workaround for OpenMP duplicate runtime issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

def main():
    model_path = "best_industrial_defect.pt"
    data_yaml = "data/data.yaml"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        sys.exit(1)
        
    print(f"Loading YOLOv8 model from {model_path}...")
    model = YOLO(model_path)
    
    print("Running validation on NEU-DET validation split...")
    # Run validation with verbose=True to print detailed logs
    results = model.val(data=data_yaml, split='val', verbose=True)
    
    print("\n" + "="*50)
    print("GLOBAL METRICS:")
    print(f"mAP50:    {results.results_dict.get('metrics/mAP50(B)', 0.0):.4f}")
    print(f"mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 0.0):.4f}")
    print(f"Precision: {results.results_dict.get('metrics/precision(B)', 0.0):.4f}")
    print(f"Recall:    {results.results_dict.get('metrics/recall(B)', 0.0):.4f}")
    print("="*50)
    
    # Class-wise metrics
    print("\nCLASS-WISE METRICS:")
    if hasattr(results, 'box') and hasattr(results.box, 'maps'):
        for i, name in model.names.items():
            try:
                # Get class-wise results: class_result(i) returns (precision, recall, ap50, ap95)
                class_res = results.box.class_result(i)
                p, r, ap50, ap95 = class_res
                print(f"Class {i} ({name}):")
                print(f"  Precision: {p:.4f}")
                print(f"  Recall:    {r:.4f}")
                print(f"  AP50:      {ap50:.4f}")
                print(f"  AP50-95:   {ap95:.4f}")
            except Exception as e:
                # Fallback if class_result is not available or has different signature
                print(f"Class {i} ({name}): Map value: {results.box.maps[i]:.4f}")
    else:
        print("Detailed class-wise metrics object not accessible, check global metrics above.")

if __name__ == "__main__":
    main()
