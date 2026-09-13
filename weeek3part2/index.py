import cv2
import time
import numpy as np
from ultralytics import YOLO

model = YOLO("best_industrial_defect.pt")

print("Successfully initialized Simulated High-FPS Video Stream...")
print("Starting real-time benchmarking loop...")

frame_count = 0
total_frames = 300
start_time = time.time()

simulated_frame = np.zeros((640, 640, 3), dtype=np.uint8)

for i in range(total_frames):
    results = model.predict(source=simulated_frame, device=0, verbose=False)
    frame_count += 1
    
    if frame_count % 30 == 0:
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        print(f"Processed Frames: {frame_count}/{total_frames} | Current Performance: {fps:.2f} FPS")

print("Video stream benchmarking task completed successfully!")
