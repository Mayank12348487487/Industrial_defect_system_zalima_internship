import os
import time
import cv2
from pathlib import Path
from onnx_inference import ONNXDetector

class VideoStreamProcessor:
    def __init__(self, source=0, model_path="best_industrial_defect.onnx", conf_threshold=0.25, fallback_dir="data/images/val"):
        self.detector = ONNXDetector(model_path=model_path, conf_threshold=conf_threshold)
        self.source = source
        self.fallback_dir = Path(fallback_dir)
        self.cap = None
        self.image_files = []
        self.current_img_idx = 0
        self.mode = "camera"
        
        # Color palette for classes (BGR format)
        # 0: crazing (Red), 1: inclusion (Green), 2: patches (Blue), 
        # 3: pitted_surface (Magenta), 4: rolled-in_scale (Cyan), 5: scratches (Orange)
        self.colors = [
            (0, 0, 255),      # Red
            (0, 255, 0),      # Green
            (255, 0, 0),      # Blue
            (255, 0, 255),    # Magenta
            (255, 255, 0),    # Cyan
            (0, 165, 255)     # Orange
        ]
        
        self.init_source()

    def init_source(self):
        # Try to initialize camera if source is an integer, or video/image file path
        is_image_file = isinstance(self.source, str) and Path(self.source).suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
        is_video_file = isinstance(self.source, str) and (os.path.exists(self.source) or self.source.startswith("http")) and not is_image_file
        
        if is_image_file:
            print(f"Attempting to open image source: {self.source}...")
            if os.path.exists(self.source):
                self.mode = "single_image"
                return
            else:
                print(f"Image source file {self.source} does not exist. Falling back to directory stream.")
        elif isinstance(self.source, int) or is_video_file:
            print(f"Attempting to open video source: {self.source}...")
            self.cap = cv2.VideoCapture(self.source)
            if self.cap is not None and self.cap.isOpened():
                print(f"Video source {self.source} successfully opened.")
                self.mode = "camera"
                return
            else:
                print(f"Failed to open video source {self.source}. Falling back to directory stream.")
        
        # Directory fallback mode
        self.mode = "directory"
        if self.fallback_dir.exists():
            self.image_files = sorted(list(self.fallback_dir.glob("*.jpg")))
            print(f"Directory stream initialized. Found {len(self.image_files)} images in {self.fallback_dir}")
        else:
            print(f"Error: Fallback directory {self.fallback_dir} does not exist.")

    def draw_detections(self, frame, detections, metrics):
        """
        Draws colored boxes and text annotations on the frame.
        """
        annotated_frame = frame.copy()
        
        # Draw bounding boxes
        for det in detections:
            box = det["box"]
            score = det["score"]
            class_id = det["class_id"]
            class_name = det["class_name"]
            
            # Select color based on class ID
            color = self.colors[class_id % len(self.colors)]
            
            # Draw box
            cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            
            # Label background
            label = f"{class_name} {score:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(annotated_frame, (box[0], box[1] - h - 5), (box[0] + w + 5, box[1]), color, -1)
            
            # Draw text
            cv2.putText(annotated_frame, label, (box[0] + 2, box[1] - 3), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            
        # Draw status overlay panel (glassmorphic style background)
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (5, 5), (260, 95), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
        
        # Add details
        fps = 1000.0 / metrics['total_ms'] if metrics['total_ms'] > 0 else 0.0
        cv2.putText(annotated_frame, f"System Status: ONLINE", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Model: YOLOv8 ONNX (CPU)", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Inference Latency: {metrics['inference_ms']:.1f} ms", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Inference FPS: {fps:.1f} FPS", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Defects Detected: {len(detections)}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255) if len(detections) > 0 else (255, 255, 255), 1, cv2.LINE_AA)
        
        return annotated_frame

    def get_frame(self):
        """
        Retrieves next frame from camera or directory list.
        """
        if self.mode == "camera" and self.cap:
            ret, frame = self.cap.read()
            if ret:
                return frame
            else:
                # If we're playing a video file, loop it!
                if isinstance(self.source, str):
                    print("Video source reached end. Looping...")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if ret:
                        return frame
                print("Failed to grab frame from camera. Re-initializing source...")
                self.init_source()
                return None
                
        elif self.mode == "directory":
            if not self.image_files:
                return None
            img_path = self.image_files[self.current_img_idx]
            frame = cv2.imread(str(img_path))
            # Move to next index (loop around)
            self.current_img_idx = (self.current_img_idx + 1) % len(self.image_files)
            return frame
            
        elif self.mode == "single_image":
            frame = cv2.imread(str(self.source))
            # Throttle slightly to simulate ~30 FPS stream of static image
            time.sleep(0.03)
            return frame
            
        return None

    def start_loop(self, max_frames=50, save_output=True):
        print(f"Starting video stream processing loop (mode: {self.mode}). Press Ctrl+C to stop.")
        output_dir = Path("data/output_stream")
        if save_output:
            output_dir.mkdir(parents=True, exist_ok=True)
            
        frame_count = 0
        gui_disabled = False
        
        try:
            while frame_count < max_frames:
                frame = self.get_frame()
                if frame is None:
                    print("No frame available, sleeping...")
                    time.sleep(0.5)
                    continue
                    
                # Run prediction
                detections, metrics = self.detector.predict(frame)
                
                # Annotate frame
                annotated_frame = self.draw_detections(frame, detections, metrics)
                
                # Try to display GUI
                if not gui_disabled:
                    try:
                        cv2.imshow("Industrial Defect Stream", annotated_frame)
                        # Wait for 30ms or key press
                        if cv2.waitKey(30) & 0xFF == ord('q'):
                            break
                    except Exception:
                        print("GUI/Display is not available. Continuing in headless mode.")
                        gui_disabled = True
                
                # Save to output directory
                if save_output:
                    out_path = output_dir / f"stream_{frame_count:04d}.jpg"
                    cv2.imwrite(str(out_path), annotated_frame)
                    
                frame_count += 1
                
                # Simulate frame rate delay if reading directory files
                if self.mode == "directory":
                    time.sleep(0.1) # 10 FPS
                    
        except KeyboardInterrupt:
            print("Loop stopped by user.")
        finally:
            self.release()
            print("Video stream processor stopped.")

    def release(self):
        if self.cap:
            self.cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

if __name__ == "__main__":
    processor = VideoStreamProcessor(source=0, conf_threshold=0.15)
    # Run a short 20-frame loop to test
    processor.start_loop(max_frames=20, save_output=True)
