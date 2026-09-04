import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort

# Defect class definitions for NEU surface defect dataset
DEFECT_CLASSES: List[str] = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

# Optimized class-wise confidence thresholds (maximizing F1-score on validation set)
OPTIMIZED_CONFIDENCE_THRESHOLDS: Dict[int, float] = {
    0: 0.23,  # crazing
    1: 0.39,  # inclusion
    2: 0.50,  # patches
    3: 0.30,  # pitted_surface
    4: 0.25,  # rolled-in_scale
    5: 0.22,  # scratches
}

# Standard BGR color palette for bounding box annotations
CLASS_COLORS_BGR: List[Tuple[int, int, int]] = [
    (0, 0, 255),      # Red (crazing)
    (0, 255, 0),      # Green (inclusion)
    (255, 0, 0),      # Blue (patches)
    (255, 0, 255),    # Magenta (pitted_surface)
    (255, 255, 0),    # Cyan (rolled-in_scale)
    (0, 165, 255),    # Orange (scratches)
]

# Hex color palette for frontend/UI presentation
CLASS_COLORS_HEX: List[str] = [
    "#ef4444",  # crazing
    "#10b981",  # inclusion
    "#3b82f6",  # patches
    "#ec4899",  # pitted_surface
    "#06b6d4",  # rolled-in_scale
    "#f59e0b",  # scratches
]


class ONNXDetector:
    """
    High-performance ONNX Runtime detector for real-time industrial defect detection.
    Supports GPU acceleration (CUDA) with CPU fallback, per-class thresholding,
    and multiple input data formats (numpy ndarray, image file path, raw bytes).
    """

    def __init__(
        self,
        model_path: Union[str, Path] = "best_industrial_defect.onnx",
        conf_threshold: Optional[Union[float, Dict[int, float]]] = None,
    ):
        self.model_path = str(model_path)
        self.classes = DEFECT_CLASSES
        self.colors_bgr = CLASS_COLORS_BGR
        self.colors_hex = CLASS_COLORS_HEX

        # Configure confidence thresholds
        if conf_threshold is None:
            self.conf_threshold: Union[float, Dict[int, float]] = dict(OPTIMIZED_CONFIDENCE_THRESHOLDS)
        else:
            self.conf_threshold = conf_threshold

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"ONNX model file not found at: {self.model_path}")

        print(f"Loading ONNX model from {self.model_path}...")
        available_providers = ort.get_available_providers()
        print(f"Available ONNX execution providers: {available_providers}")

        providers = []
        if "CUDAExecutionProvider" in available_providers:
            providers.append("CUDAExecutionProvider")
        if "CPUExecutionProvider" in available_providers:
            providers.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(
            self.model_path,
            providers=providers if providers else ["CPUExecutionProvider"],
        )
        active_providers = self.session.get_providers()
        self.device_name = (
            "NVIDIA GPU (CUDA)" if "CUDAExecutionProvider" in active_providers else "CPU (ONNX Runtime)"
        )
        print(f"Initialized ONNX session on: {self.device_name}")

        # Input and output tensor metadata
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name

        self.input_width = int(self.input_shape[2])
        self.input_height = int(self.input_shape[3])

    def get_threshold(self, class_id: int) -> float:
        """
        Returns the effective confidence threshold for a given class ID.
        """
        if isinstance(self.conf_threshold, dict):
            return self.conf_threshold.get(class_id, 0.25)
        elif isinstance(self.conf_threshold, (int, float)):
            return float(self.conf_threshold)
        return 0.25

    def get_class_metadata(self) -> List[Dict[str, Any]]:
        """
        Returns structured metadata for all supported defect classes.
        """
        metadata = []
        for i, name in enumerate(self.classes):
            bgr = self.colors_bgr[i] if i < len(self.colors_bgr) else (255, 255, 255)
            hex_color = self.colors_hex[i] if i < len(self.colors_hex) else "#ffffff"
            metadata.append({
                "class_id": i,
                "class_name": name,
                "threshold": self.get_threshold(i),
                "color_bgr": list(bgr),
                "color_hex": hex_color,
            })
        return metadata

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocesses a BGR image array into a normalized (1, 3, H, W) float32 tensor.
        """
        if not isinstance(img, np.ndarray) or img.size == 0:
            raise ValueError("Invalid input image: expected non-empty numpy.ndarray.")

        # Resize to model input dimensions (640x640)
        img_resized = cv2.resize(img, (self.input_width, self.input_height))
        # BGR -> RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        # HWC -> CHW
        img_chw = img_rgb.transpose(2, 0, 1)
        # Normalize to [0.0, 1.0]
        img_normalized = img_chw.astype(np.float32) / 255.0
        # Add batch dimension -> (1, 3, H, W)
        return np.expand_dims(img_normalized, axis=0)

    def predict(self, img: np.ndarray) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Runs object detection inference on a BGR image array.
        Returns:
            detections: List of dicts with keys ['box', 'score', 'class_id', 'class_name']
            metrics: Dict with latency breakdown in milliseconds
        """
        if img is None or not isinstance(img, np.ndarray) or img.size == 0:
            raise ValueError("Input image is None or empty.")

        orig_h, orig_w = img.shape[:2]

        t0 = time.perf_counter()
        img_batch = self.preprocess(img)
        preprocess_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        outputs = self.session.run([self.output_name], {self.input_name: img_batch})
        inference_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        # Output shape is [1, 300, 6] (End-to-End NMS output: [xmin, ymin, xmax, ymax, conf, class_id])
        raw_detections = outputs[0][0]

        scale_x = orig_w / float(self.input_width)
        scale_y = orig_h / float(self.input_height)

        results: List[Dict[str, Any]] = []
        for det in raw_detections:
            conf = float(det[4])
            class_id = int(det[5])
            thresh = self.get_threshold(class_id)

            if conf < thresh:
                continue

            xmin, ymin, xmax, ymax = det[0:4]
            class_name = self.classes[class_id] if class_id < len(self.classes) else "unknown"

            # Rescale box back to original image coordinates
            xmin_orig = max(0, int(round(xmin * scale_x)))
            ymin_orig = max(0, int(round(ymin * scale_y)))
            xmax_orig = min(orig_w, int(round(xmax * scale_x)))
            ymax_orig = min(orig_h, int(round(ymax * scale_y)))

            results.append({
                "box": [xmin_orig, ymin_orig, xmax_orig, ymax_orig],
                "score": round(conf, 4),
                "class_id": class_id,
                "class_name": class_name,
            })

        postprocess_ms = (time.perf_counter() - t0) * 1000

        metrics = {
            "preprocess_ms": round(preprocess_ms, 2),
            "inference_ms": round(inference_ms, 2),
            "postprocess_ms": round(postprocess_ms, 2),
            "total_ms": round(preprocess_ms + inference_ms + postprocess_ms, 2),
        }

        return results, metrics

    def predict_image(
        self, image_input: Union[str, Path, bytes, np.ndarray]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Convenience method to run prediction on file path, raw bytes, or numpy ndarray.
        """
        if isinstance(image_input, (str, Path)):
            path_str = str(image_input)
            if not os.path.exists(path_str):
                raise FileNotFoundError(f"Image path not found: {path_str}")
            img = cv2.imread(path_str)
            if img is None:
                raise ValueError(f"Could not decode image from path: {path_str}")
            return self.predict(img)

        elif isinstance(image_input, bytes):
            np_arr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image from provided byte stream.")
            return self.predict(img)

        elif isinstance(image_input, np.ndarray):
            return self.predict(image_input)

        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def annotate_image(
        self,
        img: np.ndarray,
        detections: List[Dict[str, Any]],
        show_confidence: bool = True,
    ) -> np.ndarray:
        """
        Draws defect bounding boxes and labels directly onto a copy of the given image.
        """
        if img is None or not isinstance(img, np.ndarray) or img.size == 0:
            raise ValueError("Input image is None or empty.")

        annotated = img.copy()
        for det in detections:
            box = det["box"]
            score = det.get("score", 1.0)
            class_id = det.get("class_id", 0)
            class_name = det.get("class_name", "defect")

            color = self.colors_bgr[class_id % len(self.colors_bgr)]
            cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), color, 2)

            label = f"{class_name} {score:.2f}" if show_confidence else class_name
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            y_badge = max(box[1] - h - 6, 0)
            cv2.rectangle(
                annotated,
                (box[0], y_badge),
                (box[0] + w + 6, y_badge + h + 6),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (box[0] + 3, y_badge + h + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return annotated

    def predict_and_annotate(
        self,
        image_input: Union[str, Path, bytes, np.ndarray],
        show_confidence: bool = True,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, float]]:
        """
        Runs detection and returns an annotated BGR image along with detections and metrics.
        """
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
            if img is None:
                raise ValueError(f"Could not decode image from path: {image_input}")
        elif isinstance(image_input, bytes):
            np_arr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image from provided byte stream.")
        elif isinstance(image_input, np.ndarray):
            img = image_input
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        detections, metrics = self.predict(img)
        annotated_img = self.annotate_image(img, detections, show_confidence=show_confidence)
        return annotated_img, detections, metrics


if __name__ == "__main__":
    detector = ONNXDetector()
    sample_img_path = "data/images/val/crazing_1.jpg" if os.path.exists("data/images/val/crazing_1.jpg") else None
    if sample_img_path:
        detections, metrics = detector.predict_image(sample_img_path)
        print(f"Sample detection on {sample_img_path}:")
        print(f"Metrics: {metrics}")
        print(f"Detections ({len(detections)}): {detections}")
    else:
        # Create a dummy image to test execution
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        detections, metrics = detector.predict(dummy_img)
        print("ONNXDetector initialized and tested successfully with synthetic frame.")
        print(f"Metrics: {metrics}")
