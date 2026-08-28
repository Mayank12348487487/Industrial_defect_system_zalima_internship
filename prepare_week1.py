from pathlib import Path
import xml.etree.ElementTree as ET
import random
import shutil
import cv2
import albumentations as A

ROOT = Path(__file__).resolve().parent

SRC = ROOT / "dataset" / "NEU-DET" / "NEU-DET"
IMAGES = SRC / "IMAGES"
ANNOTATIONS = SRC / "ANNOTATIONS"

OUT = ROOT / "dataset" / "neu_yolo"

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

CLASS_ID = {name: i for i, name in enumerate(CLASSES)}

random.seed(42)


def convert_xml(xml_file):

    root = ET.parse(xml_file).getroot()

    width = float(root.find("size/width").text)
    height = float(root.find("size/height").text)

    labels = []

    for obj in root.findall("object"):

        name = obj.find("name").text.strip()

        if name not in CLASS_ID:
            continue

        box = obj.find("bndbox")

        xmin = float(box.find("xmin").text)
        ymin = float(box.find("ymin").text)
        xmax = float(box.find("xmax").text)
        ymax = float(box.find("ymax").text)

        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height

        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height

        labels.append(
            f"{CLASS_ID[name]} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{box_width:.6f} "
            f"{box_height:.6f}"
        )

    return labels


# Create folders
for split in ["train", "val", "test"]:

    (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)


# Find image/XML pairs
pairs = []

for xml in ANNOTATIONS.glob("*.xml"):

    image = IMAGES / f"{xml.stem}.jpg"

    if image.exists():
        pairs.append((image, xml))


print("Total image/XML pairs:", len(pairs))


# Shuffle
random.shuffle(pairs)

total = len(pairs)

train_end = int(total * 0.70)
val_end = int(total * 0.90)

train = pairs[:train_end]
val = pairs[train_end:val_end]
test = pairs[val_end:]


print("Train:", len(train))
print("Validation:", len(val))
print("Test:", len(test))


def copy_dataset(items, split):

    for image, xml in items:

        shutil.copy2(
            image,
            OUT / "images" / split / image.name
        )

        labels = convert_xml(xml)

        label_file = OUT / "labels" / split / f"{image.stem}.txt"

        label_file.write_text(
            "\n".join(labels) + "\n",
            encoding="utf-8"
        )


copy_dataset(train, "train")
copy_dataset(val, "val")
copy_dataset(test, "test")


# -------------------------
# DATA AUGMENTATION
# -------------------------

transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.5
        ),
        A.GaussianBlur(
            blur_limit=(3, 5),
            p=0.15
        ),
        A.GaussNoise(p=0.15)
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.2
    )
)


train_images = list(
    (OUT / "images" / "train").glob("*.jpg")
)

augmented = 0

for image_file in train_images:

    label_file = (
        OUT / "labels" / "train" / f"{image_file.stem}.txt"
    )

    image = cv2.imread(str(image_file))

    if image is None:
        continue

    boxes = []
    classes = []

    for line in label_file.read_text().splitlines():

        parts = line.split()

        if len(parts) != 5:
            continue

        classes.append(int(parts[0]))

        boxes.append(
            [float(x) for x in parts[1:]]
        )

    result = transform(
        image=image,
        bboxes=boxes,
        class_labels=classes
    )

    output_name = image_file.stem + "_aug.jpg"

    output_image = (
        OUT / "images" / "train" / output_name
    )

    output_label = (
        OUT / "labels" / "train" /
        f"{image_file.stem}_aug.txt"
    )

    cv2.imwrite(
        str(output_image),
        result["image"]
    )

    with open(output_label, "w") as f:

        for cls, box in zip(
            result["class_labels"],
            result["bboxes"]
        ):

            f.write(
                str(cls) + " " +
                " ".join(
                    f"{x:.6f}" for x in box
                ) +
                "\n"
            )

    augmented += 1


# -------------------------
# data.yaml
# -------------------------

yaml = f"""path: {OUT.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: crazing
  1: inclusion
  2: patches
  3: pitted_surface
  4: rolled-in_scale
  5: scratches
"""

(OUT / "data.yaml").write_text(
    yaml,
    encoding="utf-8"
)


print()
print("================================")
print("WEEK 1 DATASET PREPARATION DONE")
print("================================")
print("Augmented images:", augmented)
print("Dataset:", OUT)
print("YAML:", OUT / "data.yaml")