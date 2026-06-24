"""
nhan_dang.py - Module nhận dạng và phân loại (Chương 5)
========================================================
Áp dụng các kỹ thuật:
- Trích xuất đặc trưng HOG (Histogram of Oriented Gradients)
- Trích xuất đặc trưng histogram màu
- Phân loại bằng SVM (Support Vector Machine)
"""

import cv2
import numpy as np
import os
import pickle


# ========================
# TRÍCH XUẤT ĐẶC TRƯNG HOG (Ch.5)
# ========================
def trich_xuat_hog(anh_xam, kich_thuoc=(128, 128)):
    """
    Trích xuất đặc trưng HOG (Dalal & Triggs, 2005).

    Pipeline HOG:
    ┌─────────────────────────────────────────────────────────┐
    │ Bước 1: Tính GRADIENT                                   │
    │   - Gx = ảnh * kernel_x (đạo hàm theo x)               │
    │   - Gy = ảnh * kernel_y (đạo hàm theo y)               │
    │   - Cường độ: M = √(Gx² + Gy²)                         │
    │   - Hướng: θ = arctan(Gy / Gx), chia thành 9 bin       │
    │     (0°, 20°, 40°, ..., 160°) — unsigned gradient       │
    │                                                         │
    │ Bước 2: Chia thành CELL 8×8 pixel                       │
    │   - Mỗi cell → histogram 9 bin                          │
    │   - Mỗi bin đếm tổng cường độ M theo hướng θ            │
    │                                                         │
    │ Bước 3: Gom 2×2 cell = 1 BLOCK                          │
    │   - Chuẩn hóa L2 trên mỗi block:                       │
    │     v_norm = v / √(‖v‖² + ε²)                          │
    │   - Giảm ảnh hưởng thay đổi ánh sáng                   │
    │                                                         │
    │ Bước 4: NỐI tất cả block → vector đặc trưng            │
    └─────────────────────────────────────────────────────────┘

    Đặc điểm HOG:
    - Mô tả HÌNH DẠNG và CẤU TRÚC CẠNH
    - Bất biến với thay đổi ánh sáng (nhờ chuẩn hóa block)
    - Phù hợp phân loại tổn thương da (hình dạng bất thường)

    Tham số:
    - anh_xam: ảnh grayscale
    - kich_thuoc: resize về kích thước chuẩn trước khi tính HOG
    """
    # Resize về kích thước chuẩn (HOG cần kích thước cố định)
    anh_resize = cv2.resize(anh_xam, kich_thuoc)

    # Cấu hình HOG descriptor
    win_size = kich_thuoc         # Cửa sổ = toàn ảnh
    block_size = (16, 16)         # Block = 2×2 cell
    block_stride = (8, 8)         # Bước nhảy block (overlap 50%)
    cell_size = (8, 8)            # Cell = 8×8 pixel
    so_bin = 9                    # 9 bin hướng (0°-180°)

    hog = cv2.HOGDescriptor(
        win_size, block_size, block_stride, cell_size, so_bin
    )
    dac_trung = hog.compute(anh_resize)

    return dac_trung.flatten()


# ========================
# TRÍCH XUẤT HISTOGRAM MÀU (Ch.2 + Ch.5)
# ========================
def trich_xuat_histogram_mau(anh, mask=None, so_bin=32):
    """
    Trích xuất đặc trưng histogram màu.

    - Tính histogram cho TỪNG kênh màu (B, G, R)
    - mask: chỉ tính trong vùng tổn thương (nếu có)
    - Chuẩn hóa → bất biến với kích thước ảnh

    Mục đích: Mô tả phân bố MÀU SẮC của tổn thương
    (melanoma thường có nhiều màu tối, nâu sẫm, xanh đen)
    """
    dac_trung_mau = []

    for kenh in range(3):  # B=0, G=1, R=2
        hist = cv2.calcHist(
            [anh],        # Ảnh nguồn
            [kenh],       # Kênh cần tính
            mask,         # Mask (None = toàn ảnh)
            [so_bin],     # Số bin
            [0, 256]      # Khoảng giá trị
        )
        # Chuẩn hóa histogram về [0, 1]
        cv2.normalize(hist, hist)
        dac_trung_mau.extend(hist.flatten())

    return np.array(dac_trung_mau)


# ========================
# KẾT HỢP ĐẶC TRƯNG HOG + MÀU
# ========================
def trich_xuat_dac_trung(anh, mask=None):
    """
    Kết hợp đặc trưng HOG (hình dạng) + Histogram màu.

    - HOG: mô tả cấu trúc CẠNH / HÌNH DẠNG
    - Color histogram: mô tả phân bố MÀU SẮC
    - Nối 2 vector → đặc trưng đầy đủ hơn

    Lý do kết hợp:
    - Tổn thương ác tính (melanoma) có cả đặc điểm hình dạng
      (biên không đều) và màu sắc (đa sắc, tối) khác biệt
    """
    # Chuyển sang ảnh xám cho HOG
    if len(anh.shape) == 3:
        anh_xam = cv2.cvtColor(anh, cv2.COLOR_BGR2GRAY)
    else:
        anh_xam = anh

    # Trích xuất HOG → đặc trưng hình dạng
    dac_trung_hog = trich_xuat_hog(anh_xam)

    # Trích xuất histogram màu → đặc trưng màu
    if len(anh.shape) == 3:
        dac_trung_mau = trich_xuat_histogram_mau(anh, mask)
    else:
        # Nếu ảnh xám → chỉ 1 kênh
        dac_trung_mau = trich_xuat_histogram_mau(
            cv2.cvtColor(anh, cv2.COLOR_GRAY2BGR), mask
        )

    # Nối 2 loại đặc trưng thành 1 vector
    dac_trung = np.concatenate([dac_trung_hog, dac_trung_mau])

    return dac_trung


# ========================
# HUẤN LUYỆN SVM (Ch.5)
# ========================
def huan_luyen_svm(X, y):
    """
    Huấn luyện bộ phân loại SVM (Support Vector Machine).

    SVM:
    - Tìm siêu phẳng (hyperplane) w^T·f + b = 0 có MARGIN LỚN NHẤT
    - Margin = khoảng cách giữa 2 lớp → margin lớn = tổng quát hóa tốt
    - Support vectors = các điểm nằm trên biên margin

    Kernel RBF (Radial Basis Function):
        K(f_i, f_j) = exp(-γ · ‖f_i − f_j‖²)
    - Ánh xạ dữ liệu lên không gian chiều cao hơn
    - Cho phép phân loại PHI TUYẾN (non-linear boundary)

    Tham số C (regularization):
    - C NHỎ → margin rộng, chấp nhận sai nhiều hơn (underfitting)
    - C LỚN → margin hẹp, ép đúng training data (overfitting)
    - C = 1.0 là giá trị mặc định hợp lý

    Tham số:
    - X: ma trận đặc trưng (N mẫu × D chiều)
    - y: nhãn (N mẫu × 1), ví dụ: 0 = benign, 1 = melanoma
    """
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler

    # Chuẩn hóa đặc trưng (zero mean, unit variance)
    # Quan trọng vì SVM nhạy với scale của đặc trưng
    scaler = StandardScaler()
    X_chuan = scaler.fit_transform(X)

    # Huấn luyện SVM với kernel RBF
    svm = SVC(
        kernel='rbf',        # Kernel RBF (phi tuyến)
        C=1.0,               # Regularization
        gamma='scale',       # γ = 1 / (n_features × var(X))
        probability=True     # Cho phép tính xác suất dự đoán
    )
    svm.fit(X_chuan, y)

    print(f"  Huan luyen SVM thanh cong!")
    print(f"  So support vectors: {svm.n_support_}")
    print(f"  Do chinh xac training: {svm.score(X_chuan, y):.4f}")

    return svm, scaler


# ========================
# DỰ ĐOÁN
# ========================
def du_doan(mo_hinh, scaler, dac_trung):
    """
    Dự đoán nhãn và xác suất cho 1 mẫu.

    Trả về:
    - nhan: nhãn dự đoán (0 hoặc 1)
    - xac_suat: xác suất từng lớp [P(benign), P(melanoma)]
    """
    # Chuẩn hóa đặc trưng bằng scaler đã fit
    dac_trung_chuan = scaler.transform(dac_trung.reshape(1, -1))

    nhan = mo_hinh.predict(dac_trung_chuan)
    xac_suat = mo_hinh.predict_proba(dac_trung_chuan)

    return nhan[0], xac_suat[0]


# ========================
# LƯU VÀ TẢI MÔ HÌNH
# ========================
def luu_mo_hinh(mo_hinh, scaler, duong_dan="mo_hinh_svm.pkl"):
    """Lưu mô hình SVM và scaler ra file bằng pickle."""
    thu_muc = os.path.dirname(duong_dan)
    if thu_muc and not os.path.exists(thu_muc):
        os.makedirs(thu_muc)

    with open(duong_dan, 'wb') as f:
        pickle.dump({'mo_hinh': mo_hinh, 'scaler': scaler}, f)
    print(f"  Da luu mo hinh tai: {duong_dan}")


def tai_mo_hinh(duong_dan="mo_hinh_svm.pkl"):
    """Tải mô hình SVM và scaler từ file."""
    with open(duong_dan, 'rb') as f:
        data = pickle.load(f)
    print(f"  Da tai mo hinh tu: {duong_dan}")
    return data['mo_hinh'], data['scaler']
