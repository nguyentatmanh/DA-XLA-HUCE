
import cv2
import numpy as np
from pathlib import Path


def read_image_rgb(path):
    """
    Đọc ảnh từ đường dẫn và trả về ảnh RGB (numpy array uint8).
    """
    path = Path(path)
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise ValueError(f"Không đọc được ảnh: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def preprocess_for_mtcnn(img_rgb, target_size=None):
    """
    Tiền xử lý ảnh trước khi đưa vào MTCNN:
    - Chuyển sang YCrCb
    - Blur kênh Y bằng Gaussian
    - Cân bằng histogram kênh Y
    - Ghép lại YCrCb -> RGB
    - (Tuỳ chọn) resize về target_size (width, height)

    Trả về ảnh RGB đã tăng cường (numpy array uint8).
    """
    # RGB -> YCrCb
    img_ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(img_ycrcb)

    # Gaussian blur kênh Y
    y_blur = cv2.GaussianBlur(y, (3, 3), 0)

    # Histogram equalization trên kênh Y
    y_eq = cv2.equalizeHist(y_blur)

    # Ghép lại
    img_ycrcb_eq = cv2.merge([y_eq, cr, cb])
    img_eq_rgb = cv2.cvtColor(img_ycrcb_eq, cv2.COLOR_YCrCb2RGB)

    # Resize nếu cần
    if target_size is not None:
        # target_size: (width, height)
        img_eq_rgb = cv2.resize(
            img_eq_rgb,
            target_size,
            interpolation=cv2.INTER_AREA
        )

    return img_eq_rgb
