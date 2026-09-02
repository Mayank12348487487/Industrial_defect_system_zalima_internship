import numpy as np
import pytest
from video_stream import VideoStreamProcessor


def test_video_stream_processor_directory_mode():
    processor = VideoStreamProcessor(source="directory", fallback_dir="data/images/val")
    assert processor.mode == "directory"
    
    # Grab frame
    frame = processor.get_frame()
    if len(processor.image_files) > 0:
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert len(frame.shape) == 3
    
    processor.release()


def test_draw_detections_overlay():
    processor = VideoStreamProcessor(source="directory", fallback_dir="data/images/val")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = [
        {"box": [50, 50, 200, 200], "score": 0.85, "class_id": 0, "class_name": "crazing"},
        {"box": [250, 100, 350, 250], "score": 0.92, "class_id": 2, "class_name": "patches"}
    ]
    metrics = {"preprocess_ms": 2.0, "inference_ms": 15.0, "postprocess_ms": 1.0, "total_ms": 18.0}
    
    annotated = processor.draw_detections(dummy_frame, detections, metrics)
    assert annotated is not None
    assert annotated.shape == dummy_frame.shape
    # Frame should have been modified with annotations (not all zeros)
    assert np.any(annotated > 0)
    
    processor.release()
