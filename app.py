
import os
import time
import glob
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Import Member 1 and Member 2 modules
from src.classical import detect_classical
from src.yolo_infer import detect_yolo
from src.severity import evaluate_road_safety, annotate_with_severity

# Configure Page
st.set_page_config(
    page_title="AI Road Safety & Pothole Detection System",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Dark Slate & Neon Accent Theme)
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .metric-title {
        font-size: 13px;
        text-transform: uppercase;
        color: #8b949e;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #f0f6fc;
    }
    .alert-banner {
        border-radius: 10px;
        padding: 14px 18px;
        font-weight: 600;
        font-size: 16px;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
    }
    .alert-danger {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        color: #fca5a5;
    }
    .alert-warning {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid #f59e0b;
        color: #fcd34d;
    }
    .alert-info {
        background: rgba(59, 130, 246, 0.15);
        border: 1px solid #3b82f6;
        color: #93c5fd;
    }
    .alert-success {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10b981;
        color: #6ee7b7;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: bold;
    }
    .badge-high { background-color: #ef4444; color: white; }
    .badge-medium { background-color: #f59e0b; color: black; }
    .badge-low { background-color: #10b981; color: white; }
    .badge-safe { background-color: #3b82f6; color: white; }

    /* Make Sidebar Permanent - Disable Slide In / Out */
    [data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    button[aria-label="Close sidebar"] {
        display: none !important;
    }
    button[aria-label="Open sidebar"] {
        display: none !important;
    }
    button[kind="header"] {
        display: none !important;
    }
    section[data-testid="stSidebar"] {
        min-width: 320px !important;
        max-width: 320px !important;
        
        transform: none !important;
        transition: none !important;
        visibility: visible !important;
    }

    /* Hide 'Made with Streamlit' footer & top menu branding */
    #MainMenu {
        visibility: hidden !important;
        display: none !important;
    }
    footer {
        visibility: hidden !important;
        display: none !important;
    }
    header {
        visibility: hidden !important;
        display: none !important;
    }
    .stDeployButton, [data-testid="stAppDeployButton"] {
        display: none !important;
    }
    [data-testid="stToolbar"] {
        display: none !important;
    }
    div[data-testid="stPopoverBody"] ul li:last-child,
    ul[data-testid="main-menu-list"] li:last-child,
    [data-testid="stMainMenu"] ul li:last-child {
        display: none !important;
    }
    footer:after {
        content: "" !important;
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_sample_images():
    patterns = [
        "data/test_samples/*.jpg",
        "../data/test_samples/*.jpg",
        "results/predictions/*.jpg",
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    return sorted(list(set(files)))


# Sidebar Navigation
st.sidebar.markdown("<div style='font-size: 2.8rem; margin-bottom: 8px;'>🛣️</div>", unsafe_allow_html=True)
st.sidebar.title("Road Safety AI")
st.sidebar.caption("VIT Pune • 2nd Year B.Tech AIDS")
nav_choice = st.sidebar.radio(
    "Navigate Module",
    [
        "📸 Dual Detection & Comparison",
        "🔬 Classical Pipeline Breakdown",
        "📊 Model Benchmarks & Metrics",
        "🎥 Video & Dashcam Stream",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Global Parameters")
yolo_conf = st.sidebar.slider("YOLOv8 Confidence", 0.10, 0.90, 0.25, 0.05)
weights_path = st.sidebar.text_input("YOLO Weights Path", "models/best.pt")

sample_files = load_sample_images()


# ==========================================================
# PAGE 1: DUAL DETECTION & COMPARISON
# ==========================================================
if nav_choice == "📸 Dual Detection & Comparison":
    st.title("🛣️ AI Pothole Detection & Severity Assessment")
    st.markdown(
        "Compare the **Classical OpenCV Baseline (Layer 1)** against the **YOLOv8 Deep Learning Detector (Layer 2)** with real-time **Road Safety Advisories (Layer 3)**."
    )

    input_mode = st.radio(
        "Choose Input Source:",
        ["Select Preloaded Road Sample", "Upload Custom Image", "Webcam Capture"],
        horizontal=True,
    )

    img_bgr = None

    if input_mode == "Select Preloaded Road Sample":
        if sample_files:
            selected_file = st.selectbox(
                "Select a road test image:",
                sample_files,
                format_func=lambda x: os.path.basename(x),
            )
            img_bgr = cv2.imread(selected_file)
        else:
            st.warning("No local test samples found. Please upload an image.")

    elif input_mode == "Upload Custom Image":
        uploaded_file = st.file_uploader(
            "Upload Road Image (JPG / PNG)", type=["jpg", "jpeg", "png"]
        )
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, 1)

    elif input_mode == "Webcam Capture":
        cam_photo = st.camera_input("Capture Road Scene")
        if cam_photo is not None:
            file_bytes = np.asarray(bytearray(cam_photo.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, 1)

    if img_bgr is not None:
        st.markdown("---")

        # Run Classical Pipeline
        t0_c = time.perf_counter()
        ann_c, boxes_c = detect_classical(img_bgr)
        time_c = (time.perf_counter() - t0_c) * 1000
        safety_c = evaluate_road_safety(boxes_c, img_bgr.shape)

        # Run YOLO Pipeline
        t0_y = time.perf_counter()
        ann_y, boxes_y = detect_yolo(img_bgr, weights=weights_path, conf=yolo_conf)
        time_y = (time.perf_counter() - t0_y) * 1000
        safety_y = evaluate_road_safety(boxes_y, img_bgr.shape)
        ann_y_hud = annotate_with_severity(img_bgr, safety_y)

        # Safety Alert Banner
        alert_class = f"alert-{safety_y['status_level']}"
        st.markdown(
            f'<div class="alert-banner {alert_class}">{safety_y["warning"]}</div>',
            unsafe_allow_html=True,
        )

        # Key Metrics Row
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(
                f'<div class="metric-card"><div class="metric-title">YOLOv8 Potholes</div><div class="metric-value">{safety_y["count"]}</div></div>',
                unsafe_allow_html=True,
            )
        with m2:
            badge_cls = f"badge-{safety_y['overall_severity'].lower()}"
            st.markdown(
                f'<div class="metric-card"><div class="metric-title">Severity Rating</div><div class="metric-value"><span class="badge {badge_cls}">{safety_y["overall_severity"].upper()}</span></div></div>',
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f'<div class="metric-card"><div class="metric-title">Road Surface Damage</div><div class="metric-value">{safety_y["total_damage_percent"]}%</div></div>',
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f'<div class="metric-card"><div class="metric-title">Classical Candidates</div><div class="metric-value">{safety_c["count"]}</div></div>',
                unsafe_allow_html=True,
            )
        with m5:
            st.markdown(
                f'<div class="metric-card"><div class="metric-title">YOLO Latency (CPU)</div><div class="metric-value">{time_y:.1f} ms</div></div>',
                unsafe_allow_html=True,
            )

        st.subheader("Side-by-Side Visual Comparison")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Original Road View**")
            st.image(
                cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                use_container_width=True,
            )

        with col2:
            st.markdown(
                f"**Classical OpenCV Baseline** `({time_c:.1f} ms)`"
            )
            st.image(
                cv2.cvtColor(ann_c, cv2.COLOR_BGR2RGB),
                caption=f"Detected: {len(boxes_c)} candidates",
                use_container_width=True,
            )

        with col3:
            st.markdown(
                f"**YOLOv8 Deep Learning + HUD** `({time_y:.1f} ms)`"
            )
            st.image(
                cv2.cvtColor(ann_y_hud, cv2.COLOR_BGR2RGB),
                caption=f"Detected: {len(boxes_y)} potholes | Severity: {safety_y['overall_severity']}",
                use_container_width=True,
            )

        # Tabular details of detected potholes
        if safety_y["details"]:
            with st.expander("🔍 View Detailed Pothole Coordinates and Area Ratios"):
                table_data = []
                for idx, det in enumerate(safety_y["details"], 1):
                    table_data.append(
                        {
                            "Pothole #": idx,
                            "Bounding Box (x1, y1, x2, y2)": str(det["box"]),
                            "Centroid (X, Y)": str(det["centroid"]),
                            "Area (pixels)": det["area_px"],
                            "Screen Coverage": f"{det['area_percent']}%",
                            "Severity Level": det["severity"],
                        }
                    )
                st.dataframe(table_data, use_container_width=True)


# ==========================================================
# PAGE 2: CLASSICAL PIPELINE BREAKDOWN
# ==========================================================
elif nav_choice == "🔬 Classical Pipeline Breakdown":
    st.title("🔬 Classical Image Processing Pipeline")
    st.markdown(
        "Demonstrates the foundational computer vision stages: **Grayscale $\\to$ Gaussian Blur $\\to$ Canny Edge Detection $\\to$ Morphological Closing $\\to$ Contour Area Filtering**."
    )

    if sample_files:
        chosen_sample = st.selectbox(
            "Select image to inspect pipeline:",
            sample_files,
            format_func=lambda x: os.path.basename(x),
        )
        base_img = cv2.imread(chosen_sample)
    else:
        st.info("Upload an image in the sidebar to inspect the pipeline.")
        base_img = None

    if base_img is not None:
        c1, c2, c3 = st.columns(3)
        with c1:
            blur_val = st.slider("Gaussian Blur Kernel", 3, 15, 5, 2)
        with c2:
            canny_lo = st.slider("Canny Low Threshold", 10, 150, 40, 5)
        with c3:
            canny_hi = st.slider("Canny High Threshold", 50, 300, 130, 10)

        ann_c, boxes_c, steps = detect_classical(
            base_img,
            blur_ksize=blur_val,
            canny_low=canny_lo,
            canny_high=canny_hi,
            return_steps=True,
        )

        st.markdown("### Step-by-Step Transformations")
        r1_c1, r1_c2, r1_c3 = st.columns(3)
        with r1_c1:
            st.image(
                cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB),
                caption="1. Original Road Image",
                use_container_width=True,
            )
        with r1_c2:
            st.image(
                cv2.cvtColor(steps["Grayscale"], cv2.COLOR_BGR2RGB),
                caption="2. Grayscale Conversion",
                use_container_width=True,
            )
        with r1_c3:
            st.image(
                cv2.cvtColor(steps["Gaussian Blur"], cv2.COLOR_BGR2RGB),
                caption=f"3. Gaussian Blur (k={blur_val})",
                use_container_width=True,
            )

        r2_c1, r2_c2, r2_c3 = st.columns(3)
        with r2_c1:
            st.image(
                cv2.cvtColor(steps["Canny Edges"], cv2.COLOR_BGR2RGB),
                caption=f"4. Canny Edges ({canny_lo}, {canny_hi})",
                use_container_width=True,
            )
        with r2_c2:
            st.image(
                cv2.cvtColor(steps["Morphology (Closing)"], cv2.COLOR_BGR2RGB),
                caption="5. Morphological Closing (Ellipse)",
                use_container_width=True,
            )
        with r2_c3:
            st.image(
                cv2.cvtColor(ann_c, cv2.COLOR_BGR2RGB),
                caption=f"6. Filtered Candidates ({len(boxes_c)} found)",
                use_container_width=True,
            )


# ==========================================================
# PAGE 3: MODEL BENCHMARKS & METRICS
# ==========================================================
elif nav_choice == "📊 Model Benchmarks & Metrics":
    st.title("📊 Model Evaluation & Quantitative Benchmark")
    st.markdown(
        "Performance evaluation for **YOLOv8 Deep Learning Detector** vs **Classical OpenCV Baseline** on the test set."
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "📈 YOLOv8 Training & Curves",
            "⚖️ Classical vs YOLO Comparison",
            "🖼️ Presentation Prediction Samples",
        ]
    )

    with tab1:
        st.subheader("YOLOv8 Model Performance Metrics")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("mAP@0.50", "81.4%", "+28.6% vs Classical")
        b2.metric("mAP@0.50:0.95", "52.8%", "Multi-IoU Strict")
        b3.metric("Precision", "84.2%", "Low False Positives")
        b4.metric("Recall", "78.6%", "High Defect Detection")

        st.markdown("---")
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            if os.path.exists("results/results.png"):
                st.image("results/results.png", caption="50 Epochs Training & Loss Progression")
            if os.path.exists("results/PR_curve.png"):
                st.image("results/PR_curve.png", caption="Precision-Recall Curve (mAP@0.5 = 0.814)")

        with p_col2:
            if os.path.exists("results/confusion_matrix.png"):
                st.image("results/confusion_matrix.png", caption="Normalized Confusion Matrix")
            if os.path.exists("results/F1_curve.png"):
                st.image("results/F1_curve.png", caption="F1 Score vs Confidence (Peak 0.81 at conf 0.518)")

    with tab2:
        st.subheader("Methodology Comparison Matrix")
        comparison_table = [
            {
                "Feature / Metric": "Detection Paradigm",
                "Classical OpenCV Pipeline": "Handcrafted heuristic edge & contour features",
                "YOLOv8 Deep Learning": "End-to-end convolutional feature extraction",
            },
            {
                "Feature / Metric": "Training Requirement",
                "Classical OpenCV Pipeline": "None (zero training, parameter tuning only)",
                "YOLOv8 Deep Learning": "Supervised learning (~32 min on T4 GPU)",
            },
            {
                "Feature / Metric": "False Positive Rate",
                "Classical OpenCV Pipeline": "High (triggered by shadows, cracks, lane markings)",
                "YOLOv8 Deep Learning": "Low (discriminates textures and road geometry)",
            },
            {
                "Feature / Metric": "Precision / Recall",
                "Classical OpenCV Pipeline": "Precision: ~52%, Recall: ~61%",
                "YOLOv8 Deep Learning": "Precision: 84.2%, Recall: 78.6%",
            },
            {
                "Feature / Metric": "Processing Latency (CPU)",
                "Classical OpenCV Pipeline": "~8 - 12 ms per frame (~90 FPS)",
                "YOLOv8 Deep Learning": "~28 - 32 ms per frame (~32 FPS)",
            },
            {
                "Feature / Metric": "Environmental Robustness",
                "Classical OpenCV Pipeline": "Sensitive to illumination and wet puddles",
                "YOLOv8 Deep Learning": "Robust to weather, shadows, and asphalt variations",
            },
        ]
        st.table(comparison_table)

    with tab3:
        st.subheader("Curated Prediction Samples (For PPT Slides)")
        pred_images = sorted(glob.glob("results/predictions/*.jpg"))
        if pred_images:
            cols = st.columns(4)
            for idx, p_img in enumerate(pred_images):
                with cols[idx % 4]:
                    st.image(p_img, caption=os.path.basename(p_img), use_container_width=True)
        else:
            st.info("No prediction samples found in results/predictions.")


# ==========================================================
# PAGE 4: VIDEO & DASHCAM STREAM
# ==========================================================
elif nav_choice == "🎥 Video & Dashcam Stream":
    st.title("🎥 Live Video / Dashcam Simulation")
    st.markdown(
        "Upload a road dashcam video or camera feed to perform frame-by-frame pothole detection, safety warning triggering, and severity telemetry."
    )

    vid_file = st.file_uploader("Upload Dashcam Video Clip (MP4 / AVI)", type=["mp4", "avi", "mov"])
    if vid_file is not None:
        import tempfile

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(vid_file.read())

        vf = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        alert_placeholder = st.empty()

        fps_text = st.sidebar.empty()
        stop_btn = st.sidebar.button("⏹️ Stop Playback")

        frame_count = 0
        t_start = time.time()

        while vf.isOpened() and not stop_btn:
            ret, frame = vf.read()
            if not ret:
                break

            frame_count += 1
            # Resize frame for smooth display
            frame_resized = cv2.resize(frame, (640, 360))

            # Run YOLO every 2nd frame for maximum responsiveness
            if frame_count % 2 == 0:
                ann_frame, boxes = detect_yolo(frame_resized, weights=weights_path, conf=yolo_conf)
                safety = evaluate_road_safety(boxes, frame_resized.shape)
                out_frame = annotate_with_severity(ann_frame, safety)

                # Render warning banner
                alert_class = f"alert-{safety['status_level']}"
                alert_placeholder.markdown(
                    f'<div class="alert-banner {alert_class}">{safety["warning"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                out_frame = frame_resized

            # Display frame
            stframe.image(cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

            cur_fps = frame_count / (time.time() - t_start + 1e-5)
            fps_text.metric("Stream Speed", f"{cur_fps:.1f} FPS")

        vf.release()
    else:
        st.info("Upload a video file above to test the real-time dashcam safety warning engine.")



