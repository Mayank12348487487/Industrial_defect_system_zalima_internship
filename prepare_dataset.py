import argparse
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_SRC = ROOT / "dataset" / "NEU-DET"
DEFAULT_DST = ROOT / "data"

CLASSES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
CLASS_MAP = {c: i for i, c in enumerate(CLASSES)}

def convert_coordinates(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    return x * dw, y * dh, w * dw, h * dh

def process_split(split_name: str, src_dir: Path, dst_dir: Path):
    print(f"Processing split: {split_name}...")
    split_src = src_dir / split_name
    
    # Destination directories
    img_dst_dir = dst_dir / "images" / ("train" if split_name == "train" else "val")
    lbl_dst_dir = dst_dir / "labels" / ("train" if split_name == "train" else "val")
    
    img_dst_dir.mkdir(parents=True, exist_ok=True)
    lbl_dst_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all XML annotations
    xml_dir = split_src / "annotations"
    xml_files = list(xml_dir.glob("*.xml"))
    print(f"Found {len(xml_files)} annotation files in {split_name}")
    
    copied_count = 0
    converted_count = 0
    
    for xml_file in xml_files:
        # Parse XML
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Get size
            size_elem = root.find("size")
            width = int(size_elem.find("width").text)
            height = int(size_elem.find("height").text)
            if width == 0 or height == 0:
                print(f"Skipping {xml_file.name} due to zero dimension.")
                continue
                
            filename = root.find("filename").text
            if not Path(filename).suffix:
                filename = filename + ".jpg"
            
            # Find the image in split/images/{class_folder}
            found_img_path = None
            for c in CLASSES:
                candidate_path = split_src / "images" / c / filename
                if candidate_path.exists():
                    found_img_path = candidate_path
                    break
            
            if not found_img_path:
                candidates = list((split_src / "images").rglob(filename))
                if candidates:
                    found_img_path = candidates[0]
            
            if not found_img_path or not found_img_path.exists():
                print(f"Warning: Image file {filename} not found for annotation {xml_file.name}")
                continue
            
            # Create labels txt content
            yolo_boxes = []
            for obj in root.findall("object"):
                class_name = obj.find("name").text
                if class_name not in CLASS_MAP:
                    continue
                class_id = CLASS_MAP[class_name]
                
                bndbox = obj.find("bndbox")
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)
                
                # Convert coordinates to YOLO normalized format
                x_center, y_center, w, h = convert_coordinates((width, height), (xmin, ymin, xmax, ymax))
                # Bound between 0 and 1
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                w = max(0.0, min(1.0, w))
                h = max(0.0, min(1.0, h))
                
                yolo_boxes.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
            
            if yolo_boxes:
                # Copy image
                dst_img_path = img_dst_dir / found_img_path.name
                shutil.copy2(found_img_path, dst_img_path)
                copied_count += 1
                
                # Save label
                lbl_name = xml_file.stem + ".txt"
                dst_lbl_path = lbl_dst_dir / lbl_name
                with open(dst_lbl_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(yolo_boxes) + "\n")
                converted_count += 1
                
        except Exception as e:
            print(f"Error processing {xml_file.name}: {e}")
            
    print(f"Finished split: {split_name}. Copied {copied_count} images, converted {converted_count} annotations.")

def create_yaml(dst_dir: Path, relative_path: str = "data"):
    yaml_content = f"""path: {relative_path}
train: images/train
val: images/val
test: images/val

names:
  0: crazing
  1: inclusion
  2: patches
  3: pitted_surface
  4: rolled-in_scale
  5: scratches
"""
    yaml_path = dst_dir / "data.yaml"
    dst_dir.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"Created data.yaml at {yaml_path}")

def main():
    parser = argparse.ArgumentParser(description="Prepare NEU-DET VOC dataset for YOLO training.")
    parser.add_argument("--src", type=str, default=str(DEFAULT_SRC), help="Path to source NEU-DET directory")
    parser.add_argument("--dst", type=str, default=str(DEFAULT_DST), help="Path to output data directory")
    args = parser.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)

    if not src_dir.exists():
        print(f"Source directory {src_dir} does not exist. Please specify valid path with --src.")
        return

    create_yaml(dst_dir)
    process_split("train", src_dir, dst_dir)
    process_split("validation", src_dir, dst_dir)
    print("Dataset preparation complete!")

if __name__ == "__main__":
    main()

