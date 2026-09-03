import argparse
import os
import random
from pathlib import Path

import cv2

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data"


def get_train_transforms():
    if not ALBUMENTATIONS_AVAILABLE:
        return None
    # We define a pipeline for rotations, noise, and lighting variations
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.ISONoise(p=1.0),
        ], p=0.3),
        A.OneOf([
            A.MotionBlur(p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
            A.Blur(blur_limit=3, p=1.0),
        ], p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def load_yolo_labels(label_path):
    bboxes = []
    class_labels = []
    if not label_path.exists():
        return bboxes, class_labels
        
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])
                # Albumentations expects [x_center, y_center, width, height] for YOLO format
                bboxes.append([x, y, w, h])
                class_labels.append(class_id)
    return bboxes, class_labels

def save_yolo_labels(label_path, bboxes, class_labels):
    with open(label_path, 'w') as f:
        for bbox, class_id in zip(bboxes, class_labels):
            # Write class_id x_center y_center width height
            f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

def augment_dataset(
    num_samples: int = 10,
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Optional[Path] = None,
):
    if not ALBUMENTATIONS_AVAILABLE:
        print("Albumentations library is not installed.")
        print("Please run: py -3.13 -m pip install albumentations")
        return

    images_dir = data_dir / "images" / "train"
    labels_dir = data_dir / "labels" / "train"
    aug_dir = output_dir if output_dir else data_dir / "augmented"
    aug_images_dir = aug_dir / "images"
    aug_labels_dir = aug_dir / "labels"

    print(f"Creating augmented dataset samples in {aug_dir}...")
    aug_images_dir.mkdir(parents=True, exist_ok=True)
    aug_labels_dir.mkdir(parents=True, exist_ok=True)

    # Get all training images
    image_paths = list(images_dir.glob("*.jpg"))
    if not image_paths:
        print(f"No images found in {images_dir}. Run prepare_dataset.py first.")
        return

    transform = get_train_transforms()
    samples_to_augment = random.sample(image_paths, min(len(image_paths), num_samples))

    for i, img_path in enumerate(samples_to_augment):
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        # Load image and labels
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes, class_labels = load_yolo_labels(label_path)
        if not bboxes:
            continue

        try:
            # Apply transformation
            transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            transformed_image = transformed["image"]
            transformed_bboxes = transformed["bboxes"]
            transformed_class_labels = transformed["class_labels"]

            # Save augmented image
            dst_img_name = f"aug_{img_path.name}"
            dst_img_path = aug_images_dir / dst_img_name
            # Convert back to BGR for cv2
            transformed_image_bgr = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(dst_img_path), transformed_image_bgr)

            # Save augmented label
            dst_lbl_path = aug_labels_dir / (f"aug_{img_path.stem}.txt")
            save_yolo_labels(dst_lbl_path, transformed_bboxes, transformed_class_labels)

            print(f"Augmented {i+1}/{num_samples}: {img_path.name} -> {dst_img_name}")
        except Exception as e:
            print(f"Error augmenting {img_path.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Augment training images and labels with Albumentations.")
    parser.add_argument("--samples", type=int, default=10, help="Number of training samples to augment")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="Path to base data directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory for augmented dataset")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else None
    augment_dataset(num_samples=args.samples, data_dir=Path(args.data_dir), output_dir=out_dir)

if __name__ == "__main__":
    main()

