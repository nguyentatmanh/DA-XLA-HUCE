# C:\face_detection_mtcnn\src\detection_mtcnn_tf.py
import numpy as np
from mtcnn.mtcnn import MTCNN


class FaceDetectorMTCNN(object):
    def __init__(self,
                 min_confidence=None,           # None = không lọc, giống demo
                 min_face_size=20,
                 steps_threshold=(0.6, 0.7, 0.7),
                 scale_factor=0.709):
        """
        Wrapper cho MTCNN giống demo mtcnn_demo_image,
        thêm tuỳ chọn lọc theo confidence.
        """
        self.detector = MTCNN(
            min_face_size=min_face_size,
            steps_threshold=list(steps_threshold),
            scale_factor=scale_factor,
        )
        # CHÚ Ý: không ép float nếu là None
        self.min_confidence = (
            None if min_confidence is None else float(min_confidence)
        )

    def detect(self, img_rgb):
        """
        img_rgb: ảnh RGB uint8 (H, W, 3)
        Trả về:
        - boxes: (N, 4) [x1, y1, x2, y2]
        - scores: (N,)
        - landmarks: list length N (dict keypoints)
        """
        results = self.detector.detect_faces(img_rgb)

        boxes = []
        scores = []
        landmarks = []

        for r in results:
            conf = float(r.get("confidence", 0.0))

            # Nếu có min_confidence thì mới lọc
            if (self.min_confidence is not None) and (conf < self.min_confidence):
                continue

            x, y, w, h = r["box"]
            x = max(0, x)
            y = max(0, y)
            x1, y1 = x, y
            x2, y2 = x + w, y + h

            boxes.append([x1, y1, x2, y2])
            scores.append(conf)
            landmarks.append(r.get("keypoints", {}))

        if not boxes:
            return (np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    [])

        return (np.array(boxes, dtype=np.float32),
                np.array(scores, dtype=np.float32),
                landmarks)
