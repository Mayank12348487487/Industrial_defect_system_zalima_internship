from pathlib import Path
import pytest
from augment_dataset import (
    ALBUMENTATIONS_AVAILABLE,
    get_train_transforms,
    load_yolo_labels,
    save_yolo_labels,
)
from prepare_dataset import convert_coordinates


def test_convert_coordinates():
    size = (640, 480)
    # Box: xmin=100, ymin=120, xmax=300, ymax=360
    box = (100.0, 120.0, 300.0, 360.0)
    x, y, w, h = convert_coordinates(size, box)

    assert pytest.approx(x, 0.001) == (200.0 / 640.0)
    assert pytest.approx(y, 0.001) == (240.0 / 480.0)
    assert pytest.approx(w, 0.001) == (200.0 / 640.0)
    assert pytest.approx(h, 0.001) == (240.0 / 480.0)


def test_load_and_save_yolo_labels_roundtrip(tmp_path: Path):
    label_file = tmp_path / "sample_label.txt"
    original_bboxes = [
        [0.250000, 0.350000, 0.100000, 0.200000],
        [0.600000, 0.700000, 0.150000, 0.250000],
    ]
    original_classes = [0, 4]

    save_yolo_labels(label_file, original_bboxes, original_classes)
    assert label_file.exists()

    loaded_bboxes, loaded_classes = load_yolo_labels(label_file)
    assert len(loaded_bboxes) == 2
    assert len(loaded_classes) == 2
    assert loaded_classes == [0, 4]
    for orig, loaded in zip(original_bboxes, loaded_bboxes):
        assert pytest.approx(orig, 1e-4) == loaded


def test_load_yolo_labels_nonexistent_file(tmp_path: Path):
    non_existent = tmp_path / "does_not_exist.txt"
    bboxes, class_labels = load_yolo_labels(non_existent)
    assert bboxes == []
    assert class_labels == []


def test_load_yolo_labels_malformed_lines(tmp_path: Path):
    label_file = tmp_path / "malformed_label.txt"
    content = "0 0.5 0.5 0.2 0.2\ninvalid line with words\n1 0.3 0.4 not_a_float 0.1\n2 0.8 0.8 0.1 0.1\n"
    label_file.write_text(content, encoding="utf-8")

    bboxes, class_labels = load_yolo_labels(label_file)
    assert len(bboxes) == 2
    assert class_labels == [0, 2]


def test_train_transforms_structure():
    transforms = get_train_transforms()
    if ALBUMENTATIONS_AVAILABLE:
        assert transforms is not None
    else:
        assert transforms is None
