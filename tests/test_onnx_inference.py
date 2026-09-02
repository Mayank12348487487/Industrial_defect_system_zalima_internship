import numpy as np
import pytest
import cv2
from onnx_inference import ONNXDetector, DEFECT_CLASSES, OPTIMIZED_CONFIDENCE_THRESHOLDS


@pytest.fixture(scope="module")
def detector():
    return ONNXDetector(model_path="best_industrial_defect.onnx")


def test_detector_initialization(detector):
    assert detector is not None
    assert len(detector.classes) == 6
    assert detector.input_width == 640
    assert detector.input_height == 640
    assert detector.classes == DEFECT_CLASSES


def test_class_metadata(detector):
    metadata = detector.get_class_metadata()
    assert len(metadata) == 6
    for item in metadata:
        assert "class_id" in item
        assert "class_name" in item
        assert "threshold" in item
        assert "color_bgr" in item
        assert "color_hex" in item
        assert item["class_name"] in DEFECT_CLASSES
        assert item["threshold"] == OPTIMIZED_CONFIDENCE_THRESHOLDS[item["class_id"]]


def test_preprocess(detector):
    dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    tensor = detector.preprocess(dummy_img)

    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert tensor.max() <= 1.0
    assert tensor.min() >= 0.0


def test_preprocess_invalid_input(detector):
    with pytest.raises(ValueError):
        detector.preprocess(np.array([]))


def test_predict_synthetic_frame(detector):
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    detections, metrics = detector.predict(dummy_img)

    assert isinstance(detections, list)
    assert isinstance(metrics, dict)
    assert "preprocess_ms" in metrics
    assert "inference_ms" in metrics
    assert "postprocess_ms" in metrics
    assert "total_ms" in metrics
    assert metrics["total_ms"] > 0


def test_predict_image_bytes(detector):
    dummy_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    _, encoded = cv2.imencode(".jpg", dummy_img)
    raw_bytes = encoded.tobytes()

    detections, metrics = detector.predict_image(raw_bytes)
    assert isinstance(detections, list)
    assert isinstance(metrics, dict)


def test_predict_image_invalid_path(detector):
    with pytest.raises(FileNotFoundError):
        detector.predict_image("non_existent_file_path_12345.jpg")


def test_custom_threshold_override():
    custom_detector = ONNXDetector(
        model_path="best_industrial_defect.onnx",
        conf_threshold=0.99
    )
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    detections, _ = custom_detector.predict(dummy_img)
    # High threshold should filter out almost everything
    assert len(detections) == 0
