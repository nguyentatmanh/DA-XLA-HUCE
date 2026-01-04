# src/evaluation.py
import numpy as np
from pathlib import Path


def boxes_to_centers(boxes):
    """
    Chuyển từ boxes [x1, y1, x2, y2] -> tâm (cx, cy)
    boxes: np.array (N, 4)
    Trả về: np.array (N, 2)
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    centers = np.stack([cx, cy], axis=1)
    return centers


def save_centers_txt(path, centers):
    """
    Lưu toạ độ tâm mặt ra file .txt, mỗi dòng: cx cy
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    centers = np.asarray(centers, dtype=np.float32)
    with open(path, "w", encoding="utf-8") as f:
        for cx, cy in centers:
            f.write(f"{cx:.2f} {cy:.2f}\n")
