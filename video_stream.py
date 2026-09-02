import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
from onnx_inference import CLASS_COLORS_BGR, DEFECT_CLASSES, ONNXDetector


class VideoStreamProcessor:
    """
    Ingests video streams from webcams, video files, single images, or image directory loops.
    Applies ONNX defect detection and renders real-time bounding box annotations with status telemetry.
    """

    def __init__(
        self,
        source: Union[int, str, Path] = 0,
        model_path: Union[str, Path] = "best_industrial_defect.onnx",
        conf_threshold: Optional[Union[float, Dict[int, float]]] = None,
        fallback_dir: Union[str, Path] = "data/images/val",
    ):
        self.detector = ONNXDetector(model_path=model_path, conf_threshold=conf_threshold)
        self.source = source
        self.fallback_dir = Path(fallback_dir)
        self.cap: Optional[cv2.VideoCapture] = None
        self.image_files: List[Path] = []
        self.current_img_idx = 0
        self.mode = "camera"
        self.colors = CLASS_COLORS_BGR

        self.init_source()

    def init_source(self):
        """
        Initializes video capture stream or image fallback directory loop.
        """
        source_str = str(self.source)
        is_image_file = (
            isinstance(self.source, (str, Path))
            and Path(source_str).suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        )
        is_video_file = (
            isinstance(self.source, (str, Path))
            and (os.path.exists(source_str) or source_str.startswith("http") or source_str.startswith("rtsp"))
            and not is_image_file
        )

        if is_image_file:
            print(f"Attempting to open image source: {self.source}...")
            if os.path.exists(source_str):
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
            # Support multiple image formats
            all_images = []
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
                all_images.extend(self.fallback_dir.glob(ext))
            self.image_files = sorted(all_images)
            print(f"Directory stream initialized. Found {len(self.image_files)} images in {self.fallback_dir}")
        else:
            print(f"Warning: Fallback directory {self.fallback_dir} does not exist.")

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        metrics: Dict[str, float],
    ) -> np.ndarray:
        """
        Draws colored bounding boxes, defect labels, and a glassmorphic status overlay panel.
        """
        if frame is None:
            return frame

        annotated_frame = frame.copy()

        # Draw bounding boxes and labels
        for det in detections:
            box = det["box"]
            score = det["score"]
            class_id = det["class_id"]
            class_name = det["class_name"]

            # Select color based on class ID
            color = self.colors[class_id % len(self.colors)]

            # Draw box
            cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), color, 2)

            # Label background badge
            label = f"{class_name} {score:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            y_badge = max(box[1] - h - 6, 0)
            cv2.rectangle(
                annotated_frame,
                (box[0], y_badge),
                (box[0] + w + 6, y_badge + h + 6),
                color,
                -1,
            )

            # Draw text label
            cv2.putText(
                annotated_frame,
                label,
                (box[0] + 3, y_badge + h + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Draw status overlay telemetry panel (glassmorphic style background)
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (5, 5), (280, 100), (20, 25, 35), -1)
        cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)

        # Add telemetry details
        fps = 1000.0 / metrics["total_ms"] if metrics.get("total_ms", 0.0) > 0 else 0.0
        cv2.putText(
            annotated_frame,
            "System Status: ONLINE",
            (12, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated_frame,
            f"Device: {self.detector.device_name}",
            (12, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated_frame,
            f"Inference Latency: {metrics.get('inference_ms', 0.0):.1f} ms",
            (12, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated_frame,
            f"Throughput: {fps:.1f} FPS",
            (12, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        defect_color = (0, 0, 255) if len(detections) > 0 else (200, 200, 200)
        cv2.putText(
            annotated_frame,
            f"Defects Detected: {len(detections)}",
            (12, 86),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            defect_color,
            1,
            cv2.LINE_AA,
        )

        return annotated_frame

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Retrieves the next frame from active camera, video stream, single image, or directory list.
        """
        if self.mode == "camera" and self.cap:
            ret, frame = self.cap.read()
            if ret:
                return frame
            else:
                # Loop video file if reaching end
                if isinstance(self.source, (str, Path)):
                    print("Video source reached end. Looping stream...")
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
            self.current_img_idx = (self.current_img_idx + 1) % len(self.image_files)
            return frame

        elif self.mode == "single_image":
            frame = cv2.imread(str(self.source))
            time.sleep(0.03)  # Throttle slightly to simulate ~30 FPS stream of static image
            return frame

        return None

    def start_loop(self, max_frames: int = 50, save_output: bool = True):
        """
        Runs a continuous processing loop for standalone debugging / offline analysis.
        """
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

                # Try GUI display
                if not gui_disabled:
                    try:
                        cv2.imshow("Industrial Defect Stream", annotated_frame)
                        if cv2.waitKey(30) & 0xFF == ord("q"):
                            break
                    except Exception:
                        print("GUI/Display is not available. Continuing in headless mode.")
                        gui_disabled = True

                # Save output frame
                if save_output:
                    out_path = output_dir / f"stream_{frame_count:04d}.jpg"
                    cv2.imwrite(str(out_path), annotated_frame)

                frame_count += 1

                if self.mode == "directory":
                    time.sleep(0.1)

        except KeyboardInterrupt:
            print("Loop stopped by user.")
        finally:
            self.release()
            print("Video stream processor stopped.")

    def release(self):
        """
        Releases capture device and open windows.
        """
        if self.cap:
            self.cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    processor = VideoStreamProcessor(source=0, conf_threshold=0.25)
    processor.start_loop(max_frames=10, save_output=False)

