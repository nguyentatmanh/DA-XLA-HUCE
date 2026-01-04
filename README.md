# Face Detection with MTCNN — So sánh RAW vs PRE (Tiền xử lý ảnh)

Đồ án môn **Xử lý ảnh số**: xây dựng chương trình **phát hiện khuôn mặt trên ảnh tĩnh** bằng **MTCNN (Multi-task Cascaded Convolutional Networks)** và **so sánh hiệu năng** giữa hai nhánh xử lý:

- **RAW**: phát hiện trực tiếp trên ảnh gốc  
- **PRE**: phát hiện trên ảnh sau tiền xử lý (lọc nhiễu + tăng cường tương phản)

Tiền xử lý nhánh **PRE** được thiết kế bám theo nội dung môn học: **Gaussian blur (kernel 3×3) + Histogram Equalization trên kênh độ sáng (Y) trong không gian màu YCrCb**. :contentReference[oaicite:0]{index=0}

---

## 1. Tổng quan pipeline

Với mỗi ảnh đầu vào, hệ thống chạy **cả RAW và PRE**, thu về:
- Bounding boxes (bbox)
- Confidence scores
- 5 landmarks (mắt–mũi–miệng)

Sau đó tính các chỉ số và chọn nhánh tốt nhất theo quy tắc ưu tiên:
**N (số mặt) → s (độ tin cậy trung bình) → t (thời gian xử lý)**. :contentReference[oaicite:1]{index=1}

---

## 2. Tính năng chính

- Phát hiện khuôn mặt bằng **MTCNN** (bbox + 5 landmarks)
- Chạy 2 nhánh **RAW/PRE** trên **toàn bộ ảnh trong `data/input/`**
- Vẽ bbox/landmarks lên ảnh và lưu kết quả
- Tính và lưu **tọa độ tâm** khuôn mặt ra file `.txt`
- Xuất bảng kết quả theo từng ảnh và bảng **trung bình trên tập kiểm thử**

---

## 3. Cấu trúc thư mục (gợi ý)

```text
DA-XLA-HUCE/
├─ src/
│  ├─ detection_mtcnn_tf.py
│  ├─ preprocessing.py
│  ├─ visualization.py
|  ├─ notebook
│  └─ evaluation.py
|    
├─ data/
│  ├─ input/                 # ảnh test (jpg/png)
│  └─ output/
│     └─ compare/             # ảnh đã vẽ bbox + file centers
└─ README.md
