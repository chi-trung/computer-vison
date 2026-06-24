# Phát Hiện và Phân Vùng Tổn Thương Da
## Nhóm E — Xử Lý Ảnh và Thị Giác Máy Tính (121036)

### Mục tiêu
Xây dựng pipeline xử lý ảnh phát hiện và phân vùng tổn thương da
từ ảnh dermoscopy (soi da), phục vụ hỗ trợ bác sĩ trong chẩn đoán sớm ung thư da.

### Giả thuyết
"Kết hợp GrabCut với Otsu Thresholding sẽ cho kết quả phân đoạn tốt hơn
so với chỉ dùng Otsu đơn lẻ, vì GrabCut sử dụng mô hình GMM mô tả phân bố
màu foreground/background chính xác hơn ngưỡng cứng."

### Tiêu chí thành công
- IoU > 0.5 trên tập test (đạt yêu cầu cơ bản)
- Dice > 0.6 trên tập test
- So sánh IoU(Otsu) vs IoU(Otsu+GrabCut) để kiểm chứng giả thuyết

---

## Cài đặt

```bash
# Cài đặt thư viện
pip install -r requirements.txt
```

### Yêu cầu
- Python >= 3.8
- OpenCV (`opencv-python`)
- NumPy
- scikit-learn (cho module nhận dạng SVM)
- matplotlib (cho trực quan hóa)

---

## Dataset

Dataset sử dụng: **ISIC Archive** (International Skin Imaging Collaboration)
- Nguồn: https://www.isic-archive.com
- 20 ảnh dermoscopy mẫu đã có sẵn trong `du_lieu/`

### Cấu trúc thư mục dữ liệu

```
du_lieu/
├── ISIC_0000010.jpg       # Ảnh dermoscopy
├── ISIC_0000011.jpg
├── ...
├── metadata.csv           # Thông tin bệnh nhân & chẩn đoán
├── attribution.txt
└── licenses/
```

> Ảnh đặt trực tiếp trong `du_lieu/`, không cần tạo thư mục con.

---

## Cách chạy

### 1. Test nhanh (tự động chọn ảnh đầu tiên)
```bash
python test_nhanh.py
```

### 2. Xử lý 1 ảnh cụ thể
```bash
python main.py du_lieu/ISIC_0000010.jpg
```

### 3. Xử lý hàng loạt (batch)
```bash
python main.py --batch du_lieu
```

### 4. Khảo sát tham số
```bash
python khao_sat_tham_so.py du_lieu/ISIC_0000010.jpg
```

### 5. Test từng module riêng
```bash
python tien_xu_ly.py
python phat_hien_canh.py
python phan_doan.py
```

---

## Cấu trúc project

```
Phat_hien_ton_thuong_da/
├── main.py                 # Entry point chính
├── test_nhanh.py           # Test nhanh pipeline
├── tien_xu_ly.py           # Tiền xử lý (Ch.2)
├── phat_hien_canh.py       # Phát hiện cạnh (Ch.3)
├── phan_doan.py            # Phân đoạn ảnh (Ch.4)
├── nhan_dang.py            # Nhận dạng (Ch.5)
├── danh_gia.py             # Đánh giá kết quả
├── tien_ich.py             # Hàm tiện ích dùng chung
├── khao_sat_tham_so.py     # Khảo sát tham số
├── requirements.txt
├── README.md
├── du_lieu/                # Ảnh dermoscopy (ISIC)
│   ├── ISIC_XXXXXXX.jpg
│   └── metadata.csv
└── ket_qua/                # Output (tự động tạo khi chạy)
```

---

## Pipeline

```
Ảnh gốc (dermoscopy)
    │
    ▼
[1] TIỀN XỬ LÝ (Ch.2)
    ├── Gaussian Blur (khử nhiễu)
    ├── Chuyển Grayscale
    └── CLAHE (cải thiện contrast)
    │
    ▼
[2] PHÁT HIỆN CẠNH (Ch.3)
    ├── Canny Edge Detection
    ├── Tìm Contours
    └── Trích xuất Bounding Box
    │
    ▼
[3] PHÂN ĐOẠN (Ch.4)
    ├── Otsu Thresholding (ngưỡng tự động)
    ├── GrabCut (phân đoạn GMM + Graph Cut)
    └── Morphology (làm sạch mask)
    │
    ▼
[4] NHẬN DẠNG (Ch.5) — tùy chọn
    ├── HOG (đặc trưng hình dạng)
    ├── Color Histogram (đặc trưng màu)
    └── SVM (phân loại benign/melanoma)
    │
    ▼
[5] ĐÁNH GIÁ
    ├── IoU (Intersection over Union)
    ├── Dice Coefficient
    └── Pixel Accuracy
```
