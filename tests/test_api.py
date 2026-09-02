import io
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "version" in data
    assert "detector" in data


def test_status_endpoint():
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "camera_online" in data
    assert "uptime_seconds" in data
    assert "device" in data


def test_classes_endpoint():
    response = client.get("/api/classes")
    assert response.status_code == 200
    data = response.json()
    assert "classes" in data
    assert len(data["classes"]) == 6
    class_names = [c["class_name"] for c in data["classes"]]
    assert "crazing" in class_names
    assert "inclusion" in class_names
    assert "scratches" in class_names


def test_detections_endpoint():
    response = client.get("/api/detections")
    assert response.status_code == 200
    data = response.json()
    assert "defect_count" in data
    assert "detections" in data
    assert "metrics" in data


def test_export_report_endpoint():
    response = client.get("/api/export_report")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "timestamp" in data
    assert "defect_classes" in data
    assert "total_active_clients" in data


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "defect_system_processed_frames_total" in response.text or "python_info" in response.text


def test_detect_endpoint_valid_image():
    # Create a small valid JPG in memory
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", img)
    file_bytes = io.BytesIO(encoded.tobytes())

    response = client.post(
        "/api/detect",
        files={"file": ("test_sample.jpg", file_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert "defect_count" in data
    assert "detections" in data
    assert "metrics" in data
    assert data["image_dimensions"]["width"] == 100
    assert data["image_dimensions"]["height"] == 100


def test_detect_endpoint_invalid_extension():
    file_bytes = io.BytesIO(b"fake data")
    response = client.post(
        "/api/detect",
        files={"file": ("malicious.exe", file_bytes, "application/octet-stream")}
    )
    assert response.status_code == 400
    assert "Unsupported image type" in response.json()["detail"]


def test_upload_endpoint_validation():
    # Valid file upload
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".png", img)
    file_bytes = io.BytesIO(encoded.tobytes())

    response = client.post(
        "/api/upload",
        files={"file": ("valid_defect_test.png", file_bytes, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["filename"] == "valid_defect_test.png"

    # Invalid extension
    bad_bytes = io.BytesIO(b"text file content")
    response_bad = client.post(
        "/api/upload",
        files={"file": ("readme.txt", bad_bytes, "text/plain")}
    )
    assert response_bad.status_code == 400
    assert "Unsupported file type" in response_bad.json()["detail"]


def test_set_source_endpoint():
    # Test directory source
    res_dir = client.post("/api/set_source", data={"source": "directory"})
    assert res_dir.status_code == 200
    assert res_dir.json()["status"] == "SUCCESS"

    # Test invalid path
    res_inv = client.post("/api/set_source", data={"source": "invalid_path_xyz_987"})
    assert res_inv.status_code == 200
    assert res_inv.json()["status"] == "ERROR"
