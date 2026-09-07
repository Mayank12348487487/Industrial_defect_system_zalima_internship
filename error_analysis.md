# Day 7 - Industrial Surface Defect Error Analysis Report

## 1. Executive Summary

- **Dataset Split**: NEU Metal Surface Defect Validation Split (360 high-resolution test frames across 6 defect classes).
- **Model Evaluated**: YOLOv8s fine-tuned on hot-rolled steel strip surface defects (`best_industrial_defect.pt` / `best_industrial_defect.onnx`).
- **Global Performance**:
  - **mAP@50**: `69.7%`
  - **Overall Precision**: `84.2%`
  - **Overall Recall**: `68.5%`
  - **Average Inference Latency**: `28.9 ms` (CPU) / `4.8 ms` (CUDA TensorRT)

---

## 2. Class-by-Class Performance Breakdown

| Defect Class | AP@50 | Precision | Recall | Primary Failure Mode | Recommended Threshold |
|---|---|---|---|---|---|
| **Patches** | **88.1%** | 91.2% | 85.0% | High contrast, high accuracy. Minimal false positives. | `0.50` |
| **Inclusion** | **77.7%** | 82.5% | 76.8% | Small dispersed clusters occasionally merged. | `0.39` |
| **Pitted Surface** | **76.3%** | 80.1% | 74.2% | Granular background confusion at lower lighting. | `0.30` |
| **Scratches** | **75.1%** | 83.4% | 71.0% | Faint, thin linear scratches missed under low contrast. | `0.22` |
| **Rolled-in Scale** | **57.5%** | 68.0% | 55.4% | Texture overlap with rough metal grain patterns. | `0.25` |
| **Crazing** | **43.3%** | 58.7% | 42.1% | Fine network micro-cracks blend into surface sheen. | `0.23` |

---

## 3. Failure Mode & Root Cause Analysis

### 3.1. Crazing (mAP50: 43.3%)
- **Observation**: Crazing manifests as spiderweb-like micro-crack clusters with minimal pixel-level intensity variance from the steel base substrate.
- **Root Cause**: Downsampling to $640 \times 640$ reduces gradient visibility of thin cracks under standard convolutions.
- **Mitigation**: 
  - Apply High-Frequency Enhancement and CLAHE (Contrast Limited Adaptive Histogram Equalization) preprocessing.
  - Lower class-specific confidence threshold to `0.23` to prioritize recall for industrial safety.

### 3.2. Rolled-in Scale (mAP50: 57.5%)
- **Observation**: Irregular dark patches produced during rolling occasionally get confused with oil stains or uneven illumination.
- **Root Cause**: Lack of multi-scale contextual features for large contiguous defect regions.
- **Mitigation**:
  - Incorporate Albumentations `RandomBrightnessContrast` and `ISONoise` augmentations during retraining.
  - Set confidence threshold to `0.25`.

### 3.3. Faint Linear Scratches
- **Observation**: Scratches spanning large diagonal spans with low width are detected in segments rather than single bounding boxes.
- **Mitigation**:
  - Apply Non-Maximum Suppression (NMS) IoU threshold tuning (`iou=0.45`).
  - Introduce directional motion blur and rotation augmentations in `augment_dataset.py`.

---

## 4. Next Steps & Recommended Actions

1. **Data Augmentation Pipeline**: Utilize `augment_dataset.py` with Albumentations to generate synthetic variations focusing on low-contrast illumination and micro-crack rotations.
2. **Per-Class Threshold Enforcement**: Continue using dynamic class-wise thresholds (`OPTIMIZED_CONFIDENCE_THRESHOLDS` in `onnx_inference.py`) to balance precision and recall across all defect types.
3. **Edge Optimization**: Deploy ONNX FP16 quantization for 1.8x throughput increase on edge embedded devices (Jetson / x86 edge gateways).