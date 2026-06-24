"""
tien_xu_ly.py - Module tiền xử lý ảnh (Chương 2)
===================================================
Áp dụng các kỹ thuật tiền xử lý cơ bản:
- Lọc tuyến tính: Gaussian Blur
- Toán tử điểm: Histogram Equalization (CLAHE)
- Chuyển đổi không gian màu: BGR → Gray, BGR → L*a*b*
"""

import cv2



# ========================
# GAUSSIAN BLUR (Lọc tuyến tính - Ch.2)
# ========================
def lam_mo_gaussian(anh, kich_thuoc_kernel=(5, 5)):
    """
    Làm mờ ảnh bằng bộ lọc Gaussian (lọc tuyến tính).

    Nguyên lý:
    - Tích chập (convolution) ảnh với kernel Gaussian 2D:
      G(x,y) = (1 / 2πσ²) · exp(-(x² + y²) / 2σ²)
    - Mỗi pixel mới = trung bình có trọng số của các pixel lân cận
    - Trọng số giảm dần theo khoảng cách (hình chuông)

    Tham số:
    - kich_thuoc_kernel: kích thước kernel (phải là số lẻ)
      + (3,3): mờ nhẹ → giữ nhiều chi tiết
      + (5,5): mờ vừa → cân bằng
      + (7,7): mờ mạnh → mất chi tiết nhưng khử nhiễu tốt

    Mục đích: Khử nhiễu trước khi phát hiện cạnh (Canny)
    """
    return cv2.GaussianBlur(anh, kich_thuoc_kernel, 0)


# ========================
# CHUYỂN ẢNH XÁM (Toán tử điểm - Ch.2)
# ========================
def chuyen_xam(anh):
    """
    Chuyển ảnh BGR sang grayscale (ảnh xám).

    Công thức:
        Gray = 0.299 × R + 0.587 × G + 0.114 × B

    Là toán tử điểm vì mỗi pixel output chỉ phụ thuộc
    vào pixel tương ứng ở input (không phụ thuộc lân cận).
    """
    return cv2.cvtColor(anh, cv2.COLOR_BGR2GRAY)


# ========================
# CÂN BẰNG HISTOGRAM - CLAHE (Toán tử điểm - Ch.2)
# ========================
def can_bang_histogram_clahe(anh_xam, clip_limit=2.0, tile_size=(8, 8)):
    """
    Cân bằng histogram cục bộ bằng CLAHE.
    (Contrast Limited Adaptive Histogram Equalization)

    So sánh với equalizeHist thông thường:
    ┌──────────────────────┬───────────────────────────────┐
    │ equalizeHist         │ CLAHE                         │
    ├──────────────────────┼───────────────────────────────┤
    │ Histogram toàn cục   │ Histogram cục bộ (theo tile)  │
    │ Có thể mất chi tiết  │ Giữ chi tiết cục bộ          │
    │ Dễ khuếch đại nhiễu  │ clip_limit hạn chế nhiễu     │
    └──────────────────────┴───────────────────────────────┘

    Tham số:
    - clip_limit: giới hạn contrast (cao → contrast mạnh hơn)
    - tile_size: kích thước ô tính histogram cục bộ

    Mục đích: Cải thiện contrast cho ảnh da (có thể tối/sáng không đều)
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(anh_xam)



# ========================
# PIPELINE TIỀN XỬ LÝ HOÀN CHỈNH
# ========================
def tien_xu_ly(anh, kich_thuoc_kernel=(5, 5), clip_limit=2.0):
    """
    Pipeline tiền xử lý đầy đủ cho 1 ảnh.

    Quy trình:
        Ảnh gốc (BGR)
            │
            ▼
        [1] Gaussian Blur → khử nhiễu
            │
            ├──→ anh_mo (ảnh màu đã làm mờ, dùng cho GrabCut)
            │
            ▼
        [2] Chuyển Grayscale
            │
            ▼
        [3] CLAHE → cải thiện contrast
            │
            └──→ anh_xam_eq (ảnh xám đã cân bằng, dùng cho Canny/Otsu)

    Trả về:
    - anh_xam_eq: ảnh xám đã xử lý (input cho Canny, Otsu)
    - anh_mo: ảnh màu đã làm mờ (input cho GrabCut)
    """
    # Bước 1: Làm mờ Gaussian → khử nhiễu
    anh_mo = lam_mo_gaussian(anh, kich_thuoc_kernel)

    # Bước 2: Chuyển sang ảnh xám
    anh_xam = chuyen_xam(anh_mo)

    # Bước 3: Cân bằng histogram CLAHE → tăng contrast
    anh_xam_eq = can_bang_histogram_clahe(anh_xam, clip_limit)

    return anh_xam_eq, anh_mo

