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


def test_detect_visualize_endpoint_valid_image():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", img)
    file_bytes = io.BytesIO(encoded.tobytes())

    response = client.post(
        "/api/detect/visualize",
        files={"file": ("test_sample.jpg", file_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert len(response.content) > 0


def test_detect_visualize_endpoint_invalid_extension():
    file_bytes = io.BytesIO(b"fake data")
    response = client.post(
        "/api/detect/visualize",
        files={"file": ("bad_file.txt", file_bytes, "text/plain")}
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


def test_root_index_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "<!DOCTYPE html>" in response.text or "<html" in response.text
    assert "NEU DETECT" in response.text


def test_detect_endpoint_custom_threshold():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", img)
    file_bytes = io.BytesIO(encoded.tobytes())

    response = client.post(
        "/api/detect?conf_threshold=0.99",
        files={"file": ("test_high_conf.jpg", file_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["defect_count"] == 0


def test_set_source_endpoint():
    # Test directory source
    res_dir = client.post("/api/set_source", data={"source": "directory"})
    assert res_dir.status_code == 200
    assert res_dir.json()["status"] == "SUCCESS"

    # Test webcam index source
    res_webcam = client.post("/api/set_source", data={"source": "webcam"})
    assert res_webcam.status_code == 200
    assert res_webcam.json()["status"] == "SUCCESS"

    # Test video source
    res_vid = client.post("/api/set_source", data={"source": "industry_video.mp4"})
    assert res_vid.status_code == 200
    assert res_vid.json()["status"] == "SUCCESS"

    # Test invalid path
    res_inv = client.post("/api/set_source", data={"source": "invalid_path_xyz_987"})
    assert res_inv.status_code == 200
    assert res_inv.json()["status"] == "ERROR"


def test_samples_endpoint():
    response = client.get("/api/samples")
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data
    assert isinstance(data["samples"], list)
    if len(data["samples"]) > 0:
        sample = data["samples"][0]
        assert "class_name" in sample
        assert "label" in sample
        assert "filename" in sample
        assert "url" in sample

        # Test fetching the sample image file directly
        file_res = client.get(f"/api/samples/{sample['filename']}")
        assert file_res.status_code == 200
        assert file_res.headers["content-type"] == "image/jpeg"
        assert len(file_res.content) > 0


def test_samples_endpoint_nonexistent():
    response = client.get("/api/samples/non_existent_defect_file_12345.jpg")
    assert response.status_code == 404


def test_detect_endpoint_returns_annotated_base64():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", img)
    file_bytes = io.BytesIO(encoded.tobytes())

    response = client.post(
        "/api/detect",
        files={"file": ("test_annotation.jpg", file_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "annotated_image_base64" in data
    assert data["annotated_image_base64"].startswith("data:image/jpeg;base64,")


def test_batch_detect_endpoint_success():
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img2 = np.ones((120, 120, 3), dtype=np.uint8) * 200
    _, enc1 = cv2.imencode(".jpg", img1)
    _, enc2 = cv2.imencode(".png", img2)

    response = client.post(
        "/api/detect/batch",
        files=[
            ("files", ("batch_img1.jpg", io.BytesIO(enc1.tobytes()), "image/jpeg")),
            ("files", ("batch_img2.png", io.BytesIO(enc2.tobytes()), "image/png")),
        ]
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["total_images_processed"] == 2
    assert "defective_images_count" in data
    assert "clean_images_count" in data
    assert "defect_rate_percentage" in data
    assert "class_distribution" in data
    assert "batch_summary_metrics" in data
    assert len(data["results"]) == 2
    assert data["results"][0]["filename"] == "batch_img1.jpg"
    assert data["results"][1]["filename"] == "batch_img2.png"


def test_batch_detect_endpoint_invalid_extension():
    response = client.post(
        "/api/detect/batch",
        files=[
            ("files", ("invalid.pdf", io.BytesIO(b"fake pdf"), "application/pdf")),
        ]
    )
    assert response.status_code == 400
    assert "Unsupported file" in response.json()["detail"]


def test_get_thresholds_endpoint():
    response = client.get("/api/config/thresholds")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert "current_thresholds" in data
    assert "default_thresholds" in data
    assert "crazing" in data["current_thresholds"]
    assert "scratches" in data["current_thresholds"]


def test_update_thresholds_endpoint():
    # Update per-class threshold
    response = client.post(
        "/api/config/thresholds",
        json={"thresholds": {"crazing": 0.42, "scratches": 0.33}}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["current_thresholds"]["crazing"] == 0.42
    assert data["current_thresholds"]["scratches"] == 0.33

    # Update global threshold
    res_global = client.post(
        "/api/config/thresholds",
        json={"global": 0.37}
    )
    assert res_global.status_code == 200
    global_data = res_global.json()
    for cls_name, thresh in global_data["current_thresholds"].items():
        assert thresh == 0.37

    # Reset back to defaults
    res_reset = client.post("/api/config/thresholds/reset")
    assert res_reset.status_code == 200
    reset_data = res_reset.json()
    assert reset_data["status"] == "SUCCESS"
    assert reset_data["current_thresholds"]["crazing"] == 0.23


def test_update_thresholds_invalid_values():
    # Out of range threshold (> 0.99)
    res_high = client.post(
        "/api/config/thresholds",
        json={"thresholds": {"crazing": 1.5}}
    )
    assert res_high.status_code == 400

    # Unknown defect class
    res_unknown = client.post(
        "/api/config/thresholds",
        json={"thresholds": {"unknown_defect_xyz": 0.5}}
    )
    assert res_unknown.status_code == 400


def test_export_audit_csv_endpoint():
    response = client.get("/api/export_audit_csv")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment; filename=" in response.headers["content-disposition"]
    csv_text = response.text
    assert "timestamp,unix_time,source_type,filename" in csv_text


