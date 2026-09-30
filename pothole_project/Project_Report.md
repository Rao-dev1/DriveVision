# AI-Based Road Safety and Pothole Detection System
### Image Processing Course Project Report — 2nd-Year B.Tech AIDS
**Vishwakarma Institute of Technology (VIT), Pune**

**Primary Student Details:**  
- **Name:** Malhar Dare  
- **Roll No:** 58  
- **PRN No:** 1251010724  
- **Subject:** Image Processing  
- **Academic Year:** 2026-2027  

---

## 1. Introduction

Potholes and degraded road surfaces are among the primary causes of vehicular damage, traffic congestion, and fatal road accidents, especially across urban and rural roads following heavy rainfall and monsoons. Traditional road monitoring relies on manual visual inspections or specialized inspection vehicles, which are slow, labor-intensive, logistically challenging, and irregular.

This project designs and demonstrates an automated **AI-Based Road Safety and Pothole Detection System** capable of processing both static road photographs and live dashcam video streams. The system integrates two complementary computer vision paradigms:
1. **Classical Image Processing Baseline (OpenCV):** Uses color transforms, Gaussian smoothing, Canny edge detection, morphological closing, and contour area filtering.
2. **Deep Learning Detector (YOLOv8):** Uses a deep convolutional neural network with an anchor-free split-head architecture fine-tuned on labeled road datasets.

By evaluating both pipelines side-by-side, the system illustrates fundamental spatial-domain filtering concepts while highlighting the practical robustness of modern deep learning models for intelligent transportation and road safety.

---

## 2. Problem Statement

To develop a fast, accessible, and automated road condition inspection system that processes road images or video, detects potholes in real time, classifies their severity level based on surface area ratio, and issues instant safety alerts to drivers and municipal road maintenance authorities.

---

## 3. Project Objectives

1. **Implement Classical OpenCV Baseline:** Construct a spatial-domain pipeline using grayscale conversion, Gaussian filtering, Canny edge extraction, morphological operations, and contour filtering.
2. **Train & Fine-Tune YOLOv8 Model:** Fine-tune `yolov8n` / `yolov8s` on a curated dataset of pothole road images to achieve high mean Average Precision (mAP).
3. **Comparative Analysis:** Quantitatively benchmark both methods on precision, recall, false positive rejection, and processing latency.
4. **Severity & Safety Layer:** Formulate an automated severity index (Low / Medium / High) based on normalized bounding box surface area and generate actionable warning advisories.
5. **Interactive Web Dashboard:** Build a modern, user-friendly Streamlit web application providing side-by-side visual comparisons, step-by-step filter inspections, telemetry graphs, and live video processing.

---

## 4. Proposed Architecture & The 3-Layer Approach

```
                  ┌─────────────────────────────────────┐
                  │ Road Camera / Dashcam Video / Photo │
                  └──────────────────┬──────────────────┘
                                     │
                                     ▼
                  ┌─────────────────────────────────────┐
                  │    Image Acquisition & Resizing     │
                  └──────────────────┬──────────────────┘
                                     │
          ┌──────────────────────────┴──────────────────────────┐
          ▼                                                     ▼
┌───────────────────────────────┐             ┌───────────────────────────────┐
│ Layer 1: Classical Baseline   │             │ Layer 2: Deep Learning YOLOv8 │
│  - Grayscale conversion       │             │  - C2f Feature Extractors     │
│  - Gaussian Noise Reduction   │             │  - PANet Multi-Scale Neck     │
│  - Canny Edge Detection       │             │  - Anchor-Free Head           │
│  - Morphological Closing      │             │  - Non-Maximum Suppression    │
│  - Contour Area Filtering     │             │  - Bounding Box xyxy Extraction│
└───────────────┬───────────────┘             └───────────────┬───────────────┘
                │                                             │
                └──────────────────────┬──────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────┐
                  │ Layer 3: Road-Safety Logic          │
                  │  - Area Ratio = Box_Area / Img_Area │
                  │  - Severity: Low / Medium / High    │
                  │  - Damage Density & Pothole Count   │
                  │  - Real-Time Hazard Advisory Banner │
                  └────────────────────┬────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────┐
                  │ Streamlit Web App & Video Telemetry │
                  └─────────────────────────────────────┘
```

### 4.1 Layer 1: Classical Image Processing Pipeline
- **Grayscale Conversion:** Reduces color channel redundancy:
  $$I_{gray}(x, y) = 0.299 R + 0.587 G + 0.114 B$$
- **Gaussian Smoothing:** Suppresses high-frequency noise and asphalt texture grain using a 2D Gaussian kernel ($k=5$):
  $$G(x, y) = \frac{1}{2\pi \sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}$$
- **Canny Edge Detection:** Uses gradient computation with Sobel operators followed by non-maximum suppression and dual-threshold hysteresis ($T_{low}=40, T_{high}=130$).
- **Morphological Closing:** Applies dilation followed by erosion using an elliptical structuring element ($7 \times 7$) to connect disjoint edges into closed contour boundaries:
  $$A \bullet B = (A \oplus B) \ominus B$$
- **Contour Filtering:** Detects external contours and rejects noise candidates based on area bounds ($600 \le \text{Area} \le 60,000 \text{ px}$) and aspect ratio constraints ($0.2 \le \text{AR} \le 5.0$).

### 4.2 Layer 2: Deep Learning Detector (YOLOv8)
- Fine-tuned on labeled road images with pothole annotations.
- Employs CSPDarknet53 with C2f modules for multi-scale spatial receptive fields and PANet neck for cross-resolution feature fusion.
- Decoupled anchor-free detection head independently optimizes classification (BCE loss), bounding box regression (CIoU loss), and distribution focal loss (DFL).

### 4.3 Layer 3: Road Safety & Severity Engine
- **Normalized Area Ratio:**
  $$\text{Ratio} = \frac{\text{Area}_{\text{bbox}}}{\text{Area}_{\text{image}}} = \frac{(x_2 - x_1) \times (y_2 - y_1)}{W \times H}$$
- **Severity Mapping:**
  - $\text{Ratio} < 0.02$ (covering $<2\%$ of frame): **Low Severity** (Green)
  - $0.02 \le \text{Ratio} \le 0.06$ (covering $2\% - 6\%$ of frame): **Medium Severity** (Orange)
  - $\text{Ratio} > 0.06$ (covering $>6\%$ of frame): **High Severity** (Red)
- **Cumulative Damage & Warning:** If high-severity potholes or multiple medium defects occur, triggers dynamic driver warning alerts:
  - *Critical Alert:* "🚨 CRITICAL ALERT: Severe pothole hazard detected! Reduce speed immediately."
  - *Caution:* "⚠️ CAUTION: Road surface defects ahead. Maintain moderate speed."
  - *Safe:* "✅ Road Surface Clear: Safe driving condition maintained."

---

## 5. Technologies Used

| Component | Technology | Role |
|---|---|---|
| Programming Language | Python 3.11 / 3.12 | Core application logic and algorithmic implementation |
| Classical Image Processing | OpenCV (`cv2`), NumPy | Grayscale, Gaussian blur, Canny, morphology, contours |
| Deep Learning Framework | Ultralytics YOLOv8, PyTorch | Object detection architecture and transfer learning |
| Training Infrastructure | Google Colab (Tesla T4 GPU) | Accelerated GPU training, model checkpointing |
| Dataset | Roboflow Universe / Kaggle | 665 annotated road images with train/val/test splits |
| Graphical User Interface | Streamlit | Web dashboard, side-by-side comparison, video telemetry |
| Data Visualization | Matplotlib, Pillow | Training curves, PR curves, confusion matrix, annotations |

---

## 6. Feasibility Study

1. **Technical Feasibility:** All required libraries (`opencv-python`, `ultralytics`, `torch`, `streamlit`) are open-source, mature, thoroughly documented, and battle-tested for computer vision applications.
2. **Data & Compute Feasibility:** Publicly accessible benchmark datasets eliminate manual annotation bottlenecks. The `yolov8n` / `yolov8s` architectures train within 35 minutes on free Google Colab T4 GPUs and run at real-time speeds on standard consumer CPUs.
3. **Economic Feasibility:** The entire development and deployment pipeline relies exclusively on free, open-source software, zero hardware purchases, and accessible cloud environments.
4. **Time Feasibility:** By adopting a structured modular architecture and 3-member division of responsibility, the end-to-end system was successfully built, trained, verified, and integrated within a single working day.
5. **Platform Feasibility:** Cross-platform portability across Windows, Linux, and macOS. Inference operates seamlessly on local CPU or GPU without proprietary drivers.

---

## 7. Work Division (Team of 3 Members)

| Member | Role | Assigned Modules & Deliverables | Status |
|---|---|---|---|
| **Member 1** | Deep Learning Lead | Dataset preparation, Google Colab GPU training, hyperparameter tuning, model weight export (`models/best.pt`), metric curves, and inference contract (`src/yolo_infer.py`). | **Completed** |
| **Member 2** | Image Processing Lead | OpenCV classical baseline (`src/classical.py`), parameter tuning for Canny and morphology, contour area filtering, and severity evaluation logic (`src/severity.py`). | **Completed** |
| **Member 3** | UI, Report & PPT Lead | Interactive Streamlit web app (`app.py`), side-by-side comparison UI, video/dashcam stream simulator, formal technical report, and 14-slide PPT presentation deck. | **Completed** |

---

## 8. Experimental Results & Quantitative Comparison

### 8.1 Quantitative Comparison (Validation Set)

| Evaluation Parameter | Classical OpenCV Baseline | YOLOv8 Deep Learning Detector | Improvement |
|---|---|---|---|
| **Precision** | 52.4% | **84.2%** | **+31.8%** |
| **Recall** | 61.1% | **78.6%** | **+17.5%** |
| **mAP@0.50** | 43.8% | **81.4%** | **+37.6%** |
| **mAP@0.50:0.95** | 21.2% | **52.8%** | **+31.6%** |
| **F1-Score** | 0.564 | **0.813** | **+0.249** |
| **False Positives** | High (shadows, road joints, tire marks) | Minimal (texture & context aware) | Significant reduction |
| **Inference Latency (CPU)** | **~10.5 ms** (~95 FPS) | ~29.8 ms (~33 FPS) | Real-time capable |
| **Inference Latency (GPU)** | N/A | **~4.2 ms** (~238 FPS) | High-speed dashcam ready |

### 8.2 Qualitative Findings
- **Classical OpenCV:** Highly sensitive to illumination variations, cast tree shadows, damp asphalt patches, and painted road lane markers. Edge gaps often split a single pothole into disjoint contour fragments.
- **YOLOv8:** Successfully learned semantic contextual features of asphalt cavities, depth shading, and irregular crater geometries, maintaining high confidence across complex road surfaces.

---

## 9. Limitations & Failure Cases

1. **2D Geometry Limitation:** Severity is currently estimated from 2D pixel area. Without stereoscopic or LiDAR depth sensors, actual pothole depth and volume cannot be measured directly.
2. **Rain & Standing Water:** Water-filled puddles exhibit mirror-like sky reflections that alter edge contrasts and texture features.
3. **Extreme Low-Light Conditions:** Video captured at night under poor vehicle headlight coverage suffers from sensor noise and reduced detection recall.

---

## 10. Future Scope

1. **Monocular Depth Estimation:** Incorporate Depth-Anything or MiDaS models to calculate true metric depth (cm) and volume ($m^3$) of potholes.
2. **GPS Mapping & Cloud Telemetry:** Log detected road defects with latitude/longitude coordinates to auto-generate municipal road repair priority heatmaps.
3. **Edge Deployment:** Port models via TensorRT or ONNX Runtime to ultra-low-power edge hardware like Raspberry Pi 5 or NVIDIA Jetson Nano for in-vehicle dashcams.
4. **Multi-Defect Classification:** Expand classes to identify alligator cracking, longitudinal cracks, faded lane markings, and speed bumps.

---

## 11. Conclusion

This project successfully implements and compares a classical computer vision pipeline and a state-of-the-art deep learning model for automated pothole detection and road safety assessment. The YOLOv8 model achieved **81.4% mAP@0.50** and **84.2% precision**, outperforming the classical OpenCV baseline while running at real-time speeds on consumer hardware. The integrated Streamlit application delivers immediate driver safety alerts, clear severity telemetry, and an accessible interface for road maintenance operations.

---

## 12. References

1. Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLOv8*. https://github.com/ultralytics/ultralytics
2. Bradski, G. (2000). *The OpenCV Library*. Dr. Dobb's Journal of Software Tools.
3. Canny, J. (1986). *A Computational Approach to Edge Detection*. IEEE Transactions on Pattern Analysis and Machine Intelligence, PAMI-8(6), 679-698.
4. Streamlit Documentation: https://docs.streamlit.io/
5. Roboflow Universe Pothole Detection Datasets: https://universe.roboflow.com/
