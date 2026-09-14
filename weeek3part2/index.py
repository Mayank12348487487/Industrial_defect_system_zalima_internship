import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Resolve weights path relative to repository root or script directory
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
model_path = REPO_ROOT / "best_industrial_defect.pt"
if not model_path.exists():
    model_path = Path("best_industrial_defect.pt")

# Auto-detect available compute device (GPU 0 if CUDA available, otherwise CPU)
device = 0 if torch.cuda.is_available() else "cpu"

model = YOLO(str(model_path))

print(f"Successfully initialized Simulated High-FPS Video Stream on device: {device}...")
print("Starting real-time benchmarking loop...")

frame_count = 0
total_frames = 300
start_time = time.time()

simulated_frame = np.zeros((640, 640, 3), dtype=np.uint8)

for i in range(total_frames):
    results = model.predict(source=simulated_frame, device=device, verbose=False)
    frame_count += 1
    
    if frame_count % 30 == 0:
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        print(f"Processed Frames: {frame_count}/{total_frames} | Current Performance: {fps:.2f} FPS")

print("Video stream benchmarking task completed successfully!")

