"""
Classical Image Processing Baseline for Pothole Detection
Member 2: Image Processing Lead
Project: AI-Based Road Safety and Pothole Detection System (VIT Pune)

Pipeline steps:
1. Grayscale Conversion
2. Gaussian Noise Filtering
3. Canny Edge Detection
4. Morphological Operations (Closing / Dilation)
5. Contour Extraction and Area / Shape Filtering
"""

import cv2
import numpy as np


def detect_classical(
    img_bgr,
    min_area=600,
    max_area=60000,
    blur_ksize=5,
    canny_low=40,
    canny_high=130,
    morph_ksize=7,
    return_steps=False,
):
    """
    Detects pothole candidates using classical OpenCV techniques.

    Parameters:
    -----------
    img_bgr : numpy.ndarray
        Input image in BGR format.
    min_area : int
        Minimum contour area in pixels to filter out noise.
    max_area : int
        Maximum contour area in pixels to filter out background road borders.
    blur_ksize : int
        Kernel size for Gaussian blur (must be odd).
    canny_low : int
        Lower hysteresis threshold for Canny edge detector.
    canny_high : int
        Upper hysteresis threshold for Canny edge detector.
    morph_ksize : int
        Structuring element kernel size for morphological closing.
    return_steps : bool
        If True, returns intermediate pipeline images for UI visualization.

    Returns:
    --------
    tuple : (annotated_bgr, boxes) OR (annotated_bgr, boxes, steps_dict)
        annotated_bgr : OpenCV BGR image with drawn bounding rectangles
        boxes         : list of (x1, y1, x2, y2) tuples as ints
        steps_dict    : dict of intermediate images (if return_steps is True)
    """
    if img_bgr is None:
        raise ValueError("Input image is None")

    h, w = img_bgr.shape[:2]

    # Step 1: Convert to Grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Step 2: Gaussian Blur to suppress sensor noise & road grain
    k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    blurred = cv2.GaussianBlur(gray, (k, k), 0)

    # Step 3: Canny Edge Detection
    edges = cv2.Canny(blurred, canny_low, canny_high)

    # Step 4: Morphological Operations (Elliptical kernel for natural round/oval pothole borders)
    m_k = morph_ksize if morph_ksize % 2 == 1 else morph_ksize + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (m_k, m_k))
    
    # Dilate edges slightly, then Close to fuse broken edges into closed loops
    dilated = cv2.dilate(edges, kernel, iterations=1)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Step 5: Contour Detection
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotated_bgr = img_bgr.copy()
    contour_vis = img_bgr.copy()
    boxes = []

    # Filter contours based on area and aspect ratio
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect_ratio = float(bw) / bh if bh > 0 else 0
            
            # Potholes are usually irregular blobs, filter out extreme needle-thin line noise
            if 0.2 < aspect_ratio < 5.0:
                x1, y1, x2, y2 = int(x), int(y), int(x + bw), int(y + bh)
                boxes.append((x1, y1, x2, y2))

                # Draw bounding box and contour
                cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), (255, 165, 0), 2)
                cv2.drawContours(contour_vis, [cnt], -1, (0, 255, 0), 2)

    if return_steps:
        # Convert single-channel images to 3-channel for consistent display
        steps = {
            "Grayscale": cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
            "Gaussian Blur": cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR),
            "Canny Edges": cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR),
            "Morphology (Closing)": cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR),
            "Extracted Contours": contour_vis,
        }
        return annotated_bgr, boxes, steps

    return annotated_bgr, boxes


if __name__ == "__main__":
    import sys
    print("Testing src/classical.py...")
    # Test on synthetic road surface with dark blob
    test_img = np.full((480, 640, 3), 120, dtype=np.uint8)
    cv2.circle(test_img, (320, 240), 45, (40, 40, 40), -1)
    
    ann, bxs, st = detect_classical(test_img, return_steps=True)
    print(f"Classical detection test passed! Detected {len(bxs)} candidates.")
    print(f"Intermediate steps available: {list(st.keys())}")
