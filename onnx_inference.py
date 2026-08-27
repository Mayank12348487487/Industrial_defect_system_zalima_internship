import os
import time
import cv2
import numpy as np
import onnxruntime as ort

class ONNXDetector:
    def __init__(self, model_path="best_industrial_defect.onnx", conf_threshold=0.25):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model file not found at: {model_path}")
            
        print(f"Loading ONNX model from {model_path}...")
        # Dynamically select execution providers, prioritizing GPU (CUDA)
        available_providers = ort.get_available_providers()
        print(f"Available ONNX execution providers: {available_providers}")
        providers = []
        if 'CUDAExecutionProvider' in available_providers:
            providers.append('CUDAExecutionProvider')
        if 'CPUExecutionProvider' in available_providers:
            providers.append('CPUExecutionProvider')
            
        self.session = ort.InferenceSession(
            model_path, 
            providers=providers if providers else ['CPUExecutionProvider']
        )
        self.device_name = "NVIDIA GPU (ONNX)" if "CUDAExecutionProvider" in self.session.get_providers() else "Intel CPU (ONNX)"
        
        # Get input and output details
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name
        
        self.input_width = self.input_shape[2]
        self.input_height = self.input_shape[3]
        
        self.classes = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

    def preprocess(self, img):
        """
        Preprocesses a BGR image for YOLOv8/ONNX model ingestion.
        """
        # Resize to input dimensions (typically 640x640)
        img_resized = cv2.resize(img, (self.input_width, self.input_height))
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        # Transpose HWC -> CHW
        img_chw = img_rgb.transpose(2, 0, 1)
        # Normalize to 0-1 range
        img_normalized = img_chw.astype(np.float32) / 255.0
        # Add batch dimension (1, 3, H, W)
        img_batch = np.expand_dims(img_normalized, axis=0)
        return img_batch

    def predict(self, img):
        """
        Runs inference on the image and returns formatted detections.
        """
        orig_h, orig_w = img.shape[:2]
        
        t0 = time.perf_counter()
        img_batch = self.preprocess(img)
        preprocess_time = (time.perf_counter() - t0) * 1000
        
        t0 = time.perf_counter()
        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: img_batch})
        inference_time = (time.perf_counter() - t0) * 1000
        
        t0 = time.perf_counter()
        # Output shape is [1, 300, 6] (End-to-End NMS output)
        detections = outputs[0][0]
        
        results = []
        for det in detections:
            # det is: [xmin, ymin, xmax, ymax, confidence, class_id]
            conf = float(det[4])
            if conf < self.conf_threshold:
                continue
                
            xmin, ymin, xmax, ymax = det[0:4]
            class_id = int(det[5])
            class_name = self.classes[class_id] if class_id < len(self.classes) else "unknown"
            
            # Map back to original image coordinates
            scale_x = orig_w / self.input_width
            scale_y = orig_h / self.input_height
            
            xmin_orig = max(0, int(xmin * scale_x))
            ymin_orig = max(0, int(ymin * scale_y))
            xmax_orig = min(orig_w, int(xmax * scale_x))
            ymax_orig = min(orig_h, int(ymax * scale_y))
            
            results.append({
                "box": [xmin_orig, ymin_orig, xmax_orig, ymax_orig],
                "score": conf,
                "class_id": class_id,
                "class_name": class_name
            })
            
        postprocess_time = (time.perf_counter() - t0) * 1000
        
        metrics = {
            "preprocess_ms": preprocess_time,
            "inference_ms": inference_time,
            "postprocess_ms": postprocess_time,
            "total_ms": preprocess_time + inference_time + postprocess_time
        }
        
        return results, metrics

if __name__ == "__main__":
    # Test script
    try:
        detector = ONNXDetector()
        img = cv2.imread("data/images/train/crazing_1.jpg")
        if img is not None:
            detections, metrics = detector.predict(img)
            print("Successfully processed test image!")
            print(f"Metrics: {metrics}")
            print(f"Detections: {detections}")
        else:
            print("Test image not found, make sure you ran prepare_dataset.py first.")
    except Exception as e:
        print(f"Error during ONNX test prediction: {e}")
