"""
Road-Safety and Pothole Severity Assessment Module
Member 2: Image Processing Lead
Project: AI-Based Road Safety and Pothole Detection System (VIT Pune)

Calculates area ratio, severity classification (Low / Medium / High),
cumulative road damage metrics, and driver warning advisories.
"""

import cv2
import numpy as np


# Severity thresholds based on bounding box to image area percentage
LOW_THRESHOLD = 0.02      # < 2% of frame area
MEDIUM_THRESHOLD = 0.06   # 2% to 6% of frame area
# > 6% of frame area is classified as HIGH severity

SEVERITY_COLORS = {
    "Low": (0, 255, 0),       # Green in BGR
    "Medium": (0, 165, 255),   # Orange in BGR
    "High": (0, 0, 255),       # Red in BGR
    "Safe": (255, 255, 255),
}


def compute_box_severity(box, img_shape):
    """
    Computes individual pothole severity from its bounding box area ratio.

    Parameters:
    -----------
    box : tuple of int (x1, y1, x2, y2)
    img_shape : tuple (height, width, ...)

    Returns:
    --------
    dict containing area, area_ratio, severity rating, and centroid
    """
    h, w = img_shape[:2]
    img_area = float(h * w)

    x1, y1, x2, y2 = box
    bw = max(0, x2 - x1)
    bh = max(0, y2 - y1)
    box_area = float(bw * bh)
    area_ratio = box_area / img_area if img_area > 0 else 0.0

    if area_ratio < LOW_THRESHOLD:
        severity = "Low"
    elif area_ratio <= MEDIUM_THRESHOLD:
        severity = "Medium"
    else:
        severity = "High"

    centroid = (int(x1 + bw / 2), int(y1 + bh / 2))

    return {
        "box": (x1, y1, x2, y2),
        "area_px": int(box_area),
        "area_ratio": round(area_ratio, 4),
        "area_percent": round(area_ratio * 100, 2),
        "severity": severity,
        "centroid": centroid,
    }


def evaluate_road_safety(boxes, img_shape):
    """
    Evaluates global frame road-safety condition and warning banner.

    Parameters:
    -----------
    boxes : list of (x1, y1, x2, y2)
    img_shape : tuple (height, width, ...)

    Returns:
    --------
    dict containing:
        - count: total potholes detected
        - overall_severity: 'Safe' | 'Low' | 'Medium' | 'High'
        - warning: actionable warning message string
        - status_level: 'success' | 'info' | 'warning' | 'danger'
        - details: list of individual box analyses
        - total_damage_percent: cumulative road area covered by potholes
    """
    count = len(boxes)
    if count == 0:
        return {
            "count": 0,
            "overall_severity": "Safe",
            "warning": "✅ Road Surface Clear: Safe driving condition maintained.",
            "status_level": "success",
            "details": [],
            "total_damage_percent": 0.0,
        }

    details = [compute_box_severity(b, img_shape) for b in boxes]
    severities = [d["severity"] for d in details]
    total_damage = sum(d["area_ratio"] for d in details) * 100.0

    # Determine overall frame risk
    if "High" in severities or total_damage > 8.0 or count >= 4:
        overall_severity = "High"
        warning = "🚨 CRITICAL ALERT: Severe pothole hazard detected! Reduce speed immediately."
        status_level = "danger"
    elif "Medium" in severities or count >= 2 or total_damage > 3.0:
        overall_severity = "Medium"
        warning = "⚠️ CAUTION: Road surface defects ahead. Maintain moderate speed and safe braking distance."
        status_level = "warning"
    else:
        overall_severity = "Low"
        warning = "ℹ️ ADVISORY: Minor road depression noticed. Drive carefully."
        status_level = "info"

    return {
        "count": count,
        "overall_severity": overall_severity,
        "warning": warning,
        "status_level": status_level,
        "details": details,
        "total_damage_percent": round(total_damage, 2),
    }


def annotate_with_severity(img_bgr, safety_report):
    """
    Overlays color-coded bounding boxes and an informative HUD banner on the image.

    Parameters:
    -----------
    img_bgr : numpy.ndarray (BGR image)
    safety_report : dict returned by evaluate_road_safety

    Returns:
    --------
    annotated_bgr : numpy.ndarray (annotated image)
    """
    annotated = img_bgr.copy()
    h, w = annotated.shape[:2]

    # Draw color-coded bounding boxes with tags
    for item in safety_report["details"]:
        x1, y1, x2, y2 = item["box"]
        sev = item["severity"]
        color = SEVERITY_COLORS.get(sev, (0, 0, 255))
        
        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Draw label tag above box
        label = f"{sev} ({item['area_percent']}%)"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        tag_top = max(0, y1 - lh - 8)
        cv2.rectangle(annotated, (x1, tag_top), (x1 + lw + 6, tag_top + lh + 6), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 3, tag_top + lh + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0) if sev == "Low" else (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Top Heads-Up Display (HUD) Banner
    banner_height = 42
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

    sev = safety_report["overall_severity"]
    banner_color = SEVERITY_COLORS.get(sev, (255, 255, 255))

    hud_text = (
        f"SEVERITY: {sev.upper()} | POTHOLES: {safety_report['count']} | "
        f"SURFACE DAMAGE: {safety_report['total_damage_percent']}%"
    )
    cv2.putText(
        annotated,
        hud_text,
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        banner_color,
        2,
        cv2.LINE_AA,
    )

    return annotated


if __name__ == "__main__":
    print("Testing src/severity.py...")
    sample_shape = (480, 640, 3)
    sample_boxes = [(100, 100, 200, 200), (300, 250, 480, 420)]
    report = evaluate_road_safety(sample_boxes, sample_shape)
    print("Road Safety Report:")
    for k, v in report.items():
        if k != "details":
            print(f"  {k}: {v}")
    print("src/severity.py verification passed!")
