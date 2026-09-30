from ultralytics import YOLO
import cv2
import os

_model = None
_loaded_weights_path = None

def detect_yolo(img_bgr, weights="models/best.pt", conf=0.25):
    """
    img_bgr : numpy array (OpenCV BGR image)
    weights : path to YOLOv8 weights file (default: "models/best.pt")
    conf    : detection confidence threshold (default: 0.25)
    returns : (annotated_bgr, boxes)
              annotated_bgr = image with bounding boxes drawn
              boxes = list of (x1, y1, x2, y2) as ints, in pixel coordinates
    """
    global _model, _loaded_weights_path

    resolved_weights = weights
    # If weights not found directly at working directory, resolve relative paths
    if not os.path.exists(resolved_weights):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.normpath(os.path.join(script_dir, "..", weights)),
            os.path.normpath(os.path.join(script_dir, "models", "best.pt")),
            os.path.normpath(os.path.join(script_dir, "..", "models", "best.pt")),
            os.path.normpath(os.path.join(script_dir, "..", "pothole_project", "models", "best.pt")),
        ]
        for cand in candidates:
            if os.path.exists(cand):
                resolved_weights = cand
                break

    if _model is None or _loaded_weights_path != resolved_weights:
        _model = YOLO(resolved_weights)
        _loaded_weights_path = resolved_weights

    result = _model.predict(img_bgr, conf=conf, verbose=False)[0]
    annotated = img_bgr.copy()
    boxes = []

    if result.boxes is not None and len(result.boxes) > 0:
        for b in result.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, b)
            boxes.append((x1, y1, x2, y2))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return annotated, boxes

if __name__ == "__main__":
    import sys
    print("Testing yolo_infer.py...")
    test_img_path = sys.argv[1] if len(sys.argv) > 1 else None
    if test_img_path and os.path.exists(test_img_path):
        img = cv2.imread(test_img_path)
    else:
        # Create a synthetic image for test verification
        import numpy as np
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(img, (320, 240), 50, (100, 100, 100), -1)

    annotated, detected_boxes = detect_yolo(img)
    print(f"Inference succeeded! Found {len(detected_boxes)} potholes.")
    print(f"Boxes: {detected_boxes}")
