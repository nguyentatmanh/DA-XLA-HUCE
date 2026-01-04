# src/visualization.py
import cv2
import numpy as np
from pathlib import Path


def draw_detections(img_rgb, boxes, scores=None, landmarks=None,
                    show_scores=True):
    """
    Vẽ bounding box (và landmark nếu có) lên ảnh RGB.

    - img_rgb: ảnh RGB gốc (numpy array HxWx3)
    - boxes: np.array (N, 4) [x1, y1, x2, y2]
    - scores: np.array (N,) hoặc None
    - landmarks: list length N, mỗi phần tử là dict (left_eye, right_eye, nose, mouth_left, mouth_right) hoặc None

    Trả về: ảnh RGB đã vẽ.
    """
    img_vis = img_rgb.copy()

    if boxes is None or len(boxes) == 0:
        return img_vis

    boxes = np.asarray(boxes)

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].astype(int)

        # Vẽ khung (màu xanh lá)
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Vẽ score
        if show_scores and scores is not None and len(scores) > i:
            label = f"{scores[i]:.2f}"
            cv2.putText(
                img_vis,
                label,
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        # Vẽ landmark
        if landmarks is not None and len(landmarks) > i and landmarks[i] is not None:
            for (lx, ly) in landmarks[i].values():
                cv2.circle(img_vis, (int(lx), int(ly)), 2, (255, 0, 0), -1)

    return img_vis


def save_rgb_image(path, img_rgb):
    """
    Lưu ảnh RGB (numpy) ra file (BGR cho OpenCV).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), img_bgr)
