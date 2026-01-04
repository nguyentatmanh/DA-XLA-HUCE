# Face Detection Project (MTCNN) – So sánh RAW vs PRE (Tiền xử lý ảnh)

Đồ án môn **Xử lý ảnh số**: xây dựng chương trình **phát hiện khuôn mặt** trên ảnh tĩnh và so sánh kết quả giữa:
- **RAW**: phát hiện trực tiếp trên ảnh gốc
- **PRE**: phát hiện trên ảnh đã qua **tiền xử lý ảnh** (lọc nhiễu, tăng cường tương phản theo nội dung môn)

Chương trình thực hiện:
1) Chọn 1 ảnh từ `data/input`
2) Tiền xử lý và hiển thị **ảnh gốc vs ảnh sau tiền xử lý**
3) Chạy phát hiện khuôn mặt trên **cả 2 ảnh** (RAW và PRE)
4) In thống kê: **số khuôn mặt phát hiện** và **thời gian xử lý**
5) Vẽ bounding box lên cả 2 ảnh
6) Chọn kết quả tốt hơn và lưu vào `data/output`
7) Lưu tọa độ tâm khuôn mặt ra file `.txt`


## Công cụ và môi trường

### Ngôn ngữ & môi trường
- **Python 3.x**
- **Jupyter Notebook** (VS Code / JupyterLab)

### Thư viện sử dụng
- `mtcnn` (MTCNN – deep learning, backend TensorFlow) phát hiện khuôn mặt và landmark
- `opencv-python` (xử lý ảnh) (OpenCV): đọc/ghi ảnh, chuyển đổi không gian màu, lọc ảnh, tăng cường ảnh
- `numpy` xử lý ma trận ảnh
- `matplotlib` hiển thị ảnh trong notebook



