import os
import sys

# Workaround for OpenMP duplicate runtime issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

def main():
    model_path = "best_industrial_defect.pt"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        sys.exit(1)
        
    print(f"Loading YOLOv8 model from {model_path}...")
    model = YOLO(model_path)
    
    print("Exporting model to ONNX format...")
    # Export the model
    # format='onnx' exports to ONNX format.
    # We can also enable dynamic shapes if needed, but static 640x640 is standard and efficient.
    onnx_path = model.export(format="onnx", opset=12, simplify=True)
    
    print(f"\nModel exported successfully to: {onnx_path}")

if __name__ == "__main__":
    main()
