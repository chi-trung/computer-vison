"""
tien_ich.py - Module tiện ích dùng chung
=========================================
Chứa các hàm đọc/ghi ảnh, resize, hiển thị
dùng chung cho toàn bộ project.
"""

import cv2
import os
import numpy as np


# ========================
# ĐỌC ẢNH TỪ Ổ CỨNG
# ========================
def doc_anh(duong_dan):
    """
    Đọc ảnh từ đường dẫn.
    Trả về ảnh dạng BGR (mặc định của OpenCV).
    Raise lỗi nếu không tìm thấy file.
    """
    anh = cv2.imread(duong_dan)
    if anh is None:
        raise ValueError(f"Lỗi: Không thể đọc ảnh tại '{duong_dan}'!")
    return anh


# ========================
# ĐỌC MASK (ẢNH NHỊ PHÂN)
# ========================
def doc_mask(duong_dan):
    """
    Đọc mask (ground truth) dạng ảnh xám.
    Chuẩn hóa về nhị phân: 0 (background) và 255 (foreground).
    """
    mask = cv2.imread(duong_dan, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Lỗi: Không thể đọc mask tại '{duong_dan}'!")

    # Chuẩn hóa: pixel > 127 → 255 (foreground), ngược lại → 0
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


# ========================
# RESIZE GIỮ TỈ LỆ
# ========================
def resize_giu_ti_le(anh, chieu_rong=512):
    """
    Resize ảnh giữ nguyên tỉ lệ khung hình.
    Chỉ cần chỉ định chiều rộng, chiều cao tự tính.

    Công thức:
        ti_le = chieu_rong_moi / chieu_rong_cu
        chieu_cao_moi = chieu_cao_cu * ti_le
    """
    h, w = anh.shape[:2]
    ti_le = chieu_rong / w
    chieu_cao_moi = int(h * ti_le)
    return cv2.resize(anh, (chieu_rong, chieu_cao_moi))


# ========================
# LƯU ẢNH RA FILE
# ========================
def luu_anh(anh, duong_dan):
    """
    Lưu ảnh ra file.
    Tự động tạo thư mục nếu chưa tồn tại.
    """
    thu_muc = os.path.dirname(duong_dan)
    if thu_muc and not os.path.exists(thu_muc):
        os.makedirs(thu_muc)
    cv2.imwrite(duong_dan, anh)


# ========================
# TẠO THƯ MỤC
# ========================
def tao_thu_muc(duong_dan):
    """Tạo thư mục nếu chưa tồn tại."""
    if not os.path.exists(duong_dan):
        os.makedirs(duong_dan)


# ========================
# HIỂN THỊ NHIỀU ẢNH
# ========================
def hien_thi_anh(tieu_de_va_anh):
    """
    Hiển thị nhiều ảnh trên các cửa sổ riêng biệt.

    Tham số:
        tieu_de_va_anh: list các tuple (tieu_de, anh)
        Ví dụ: [("Anh goc", img1), ("Canny", img2)]
    """
    for tieu_de, anh in tieu_de_va_anh:
        cv2.imshow(tieu_de, anh)
    print("Nhan phim bat ky de dong cua so...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ========================
# OVERLAY MASK LÊN ẢNH GỐC
# ========================
def overlay_mask(anh, mask, mau=(0, 255, 0), do_trong_suot=0.3):
    """
    Tô màu vùng mask lên ảnh gốc để trực quan hóa kết quả.

    Tham số:
        anh: ảnh gốc BGR
        mask: mask nhị phân (0 hoặc 255)
        mau: màu tô (BGR), mặc định xanh lá
        do_trong_suot: độ trong suốt (0=trong suốt, 1=đặc)
    """
    anh_ket_qua = anh.copy()
    lop_mau = np.zeros_like(anh)
    lop_mau[mask > 0] = mau
    anh_ket_qua = cv2.addWeighted(anh_ket_qua, 1 - do_trong_suot, lop_mau, do_trong_suot, 0)
    return anh_ket_qua
