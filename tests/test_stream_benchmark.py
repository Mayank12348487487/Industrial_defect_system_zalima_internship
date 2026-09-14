import numpy as np
import pytest
from pathlib import Path


def test_simulated_stream_benchmark_logic():
    from onnx_inference import ONNXDetector

    detector = ONNXDetector(model_path="best_industrial_defect.onnx")
    simulated_frame = np.zeros((640, 640, 3), dtype=np.uint8)

    # Perform a few test iterations to ensure synthetic frame prediction works cleanly
    for _ in range(5):
        dets, metrics = detector.predict(simulated_frame)
        assert isinstance(dets, list)
        assert "inference_ms" in metrics
        assert metrics["inference_ms"] > 0
