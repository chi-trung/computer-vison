"""
phat_hien_canh.py - Module phát hiện cạnh (Chương 3)
======================================================
Áp dụng các kỹ thuật:
- Phát hiện cạnh Canny (Canny Edge Detection)
- Tìm đường viền (Contours) + lọc theo diện tích
- Trích xuất bounding box cho GrabCut
"""

import cv2
import numpy as np


# ========================
# PHÁT HIỆN CẠNH CANNY (Ch.3)
# ========================
def phat_hien_canh_canny(anh_xam, nguong_thap=50, nguong_cao=150):
    """
    Phát hiện cạnh bằng thuật toán Canny (John Canny, 1986).
    Là thuật toán phát hiện cạnh tốt nhất (chuẩn công nghiệp).

    Pipeline 4 bước bên trong cv2.Canny():
    ┌─────────────────────────────────────────────────────────┐
    │ Bước 1: Gaussian Smoothing                              │
    │   → Làm mờ nhẹ để khử nhiễu                            │
    │                                                         │
    │ Bước 2: Gradient (Sobel)                                │
    │   → Tính cường độ M = √(Gx² + Gy²)                     │
    │   → Tính hướng θ = arctan(Gy / Gx)                     │
    │                                                         │
    │ Bước 3: Non-Maximum Suppression (NMS)                   │
    │   → Làm mỏng cạnh xuống 1 pixel                        │
    │   → Chỉ giữ pixel là cực đại cục bộ theo hướng θ       │
    │                                                         │
    │ Bước 4: Hysteresis Thresholding (ngưỡng kép)            │
    │   → M > nguong_cao: cạnh MẠNH → GIỮ                    │
    │   → nguong_thap ≤ M ≤ nguong_cao: cạnh YẾU             │
    │     → GIỮ nếu NỐI với cạnh mạnh, BỎ nếu cô lập        │
    │   → M < nguong_thap: NHIỄU → BỎ                        │
    └─────────────────────────────────────────────────────────┘

    Tham số:
    - nguong_thap: ngưỡng dưới (khuyến nghị ≈ nguong_cao / 2 hoặc / 3)
    - nguong_cao: ngưỡng trên
    - Tỉ lệ nguong_cao / nguong_thap ≈ 2-3 là tốt nhất

    Ảnh hưởng tham số:
    - nguong_cao THẤP → nhiều cạnh hơn (nhạy hơn, có thể nhiều nhiễu)
    - nguong_cao CAO → ít cạnh hơn (chặt hơn, có thể mất cạnh yếu)
    """
    return cv2.Canny(anh_xam, nguong_thap, nguong_cao)


# ========================
# TÌM ĐƯỜNG VIỀN (CONTOURS) (Ch.3)
# ========================
def tim_duong_vien(anh_nhi_phan):
    """
    Tìm các đường viền (contours) từ ảnh nhị phân (ảnh cạnh).

    - Contour = danh sách các điểm (x, y) tạo thành đường cong khép kín
    - RETR_EXTERNAL: chỉ lấy contour ngoài cùng
      (bỏ qua contour lồng bên trong → tránh trùng lặp)
    - CHAIN_APPROX_SIMPLE: nén contour, chỉ giữ đỉnh
      (đoạn thẳng chỉ cần 2 điểm đầu-cuối thay vì mọi pixel)

    Trả về: danh sách các contour (mỗi contour là mảng Nx1x2)
    """
    duong_vien, _ = cv2.findContours(
        anh_nhi_phan,
        cv2.RETR_EXTERNAL,       # Chỉ contour ngoài cùng
        cv2.CHAIN_APPROX_SIMPLE  # Nén đường thẳng
    )
    return duong_vien


# ========================
# LỌC ĐƯỜNG VIỀN THEO DIỆN TÍCH
# ========================
def loc_duong_vien(duong_vien_list, dien_tich_toi_thieu=500, kich_thuoc_anh=None):
    """
    Lọc bỏ các contour không phải tổn thương.

    Tiêu chí lọc:
    1. Quá nhỏ (< dien_tich_toi_thieu): nhiễu
    2. Quá lớn (> 80% ảnh): viền kính dermoscope
    3. Chạm biên ảnh: viền kính dermoscope hoặc artifact

    Lý do cần lọc viền kính:
    - Ảnh dermoscopy thường có viền tròn đen (kính soi da)
    - Canny phát hiện viền này → contour bao trùm cả ảnh
    - BBox trùng toàn bộ ảnh → GrabCut thất bại

    Trả về: danh sách contour hợp lệ, sắp xếp theo diện tích giảm dần
    """
    duong_vien_hop_le = []

    for dv in duong_vien_list:
        dien_tich = cv2.contourArea(dv)

        # Lọc 1: quá nhỏ
        if dien_tich < dien_tich_toi_thieu:
            continue

        # Lọc 2: quá lớn (> 80% diện tích ảnh)
        if kich_thuoc_anh is not None:
            h_anh, w_anh = kich_thuoc_anh
            if dien_tich > 0.8 * h_anh * w_anh:
                continue

        # Lọc 3: chạm biên ảnh (viền kính dermoscope)
        if kich_thuoc_anh is not None:
            h_anh, w_anh = kich_thuoc_anh
            x, y, w, h = cv2.boundingRect(dv)
            margin = 5  # cho phép sai số 5 pixel

            cham_trai = x <= margin
            cham_phai = (x + w) >= (w_anh - margin)
            cham_tren = y <= margin
            cham_duoi = (y + h) >= (h_anh - margin)

            # Nếu chạm 3+ cạnh → chắc chắn là viền kính
            so_canh_cham = sum([cham_trai, cham_phai, cham_tren, cham_duoi])
            if so_canh_cham >= 3:
                continue

        duong_vien_hop_le.append(dv)

    # Sắp xếp: contour lớn nhất lên đầu
    duong_vien_hop_le.sort(key=cv2.contourArea, reverse=True)

    return duong_vien_hop_le


# ========================
# VẼ ĐƯỜNG VIỀN LÊN ẢNH
# ========================
def ve_duong_vien(anh, duong_vien_list, mau=(0, 255, 0), do_day=3):
    """
    Vẽ các đường viền lên ảnh (tạo bản sao, KHÔNG sửa ảnh gốc).

    Tham số:
    - mau: màu BGR, mặc định (0,255,0) = xanh lá
    - do_day: độ dày nét vẽ (pixel)
    """
    anh_ve = anh.copy()
    cv2.drawContours(anh_ve, duong_vien_list, -1, mau, do_day)
    return anh_ve


# ========================
# LẤY BOUNDING BOX TỪ CONTOUR LỚN NHẤT
# ========================
def lay_bounding_box(duong_vien_list, margin=10):
    """
    Lấy bounding box (hình chữ nhật bao quanh) từ contour lớn nhất.

    Mục đích:
    - Bounding box làm INPUT cho GrabCut (Ch.4)
    - GrabCut cần biết vùng ước lượng có foreground

    Tham số:
    - margin: mở rộng bbox thêm margin pixel mỗi phía
      (để GrabCut có thêm context xung quanh)

    Trả về: (x, y, w, h) hoặc None nếu không có contour
    """
    if not duong_vien_list:
        return None

    # Lấy contour có diện tích lớn nhất
    contour_lon_nhat = max(duong_vien_list, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour_lon_nhat)

    # Mở rộng bbox thêm margin (nhưng không vượt quá biên ảnh)
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = w + 2 * margin
    h = h + 2 * margin

    return (x, y, w, h)


# ========================
# TEST MODULE
# ========================
if __name__ == "__main__":
    from tien_ich import doc_anh, resize_giu_ti_le
    from tien_xu_ly import tien_xu_ly

    # Đọc và tiền xử lý
    anh = doc_anh("du_lieu/anh_goc/ISIC_0024306.jpg")
    anh = resize_giu_ti_le(anh, 512)
    anh_xam_eq, _ = tien_xu_ly(anh)

    # Phát hiện cạnh Canny
    anh_canh = phat_hien_canh_canny(anh_xam_eq, 50, 150)

    # Tìm và lọc đường viền
    duong_vien = tim_duong_vien(anh_canh)
    duong_vien = loc_duong_vien(duong_vien, 500, kich_thuoc_anh=anh.shape[:2])
    print(f"Tim thay {len(duong_vien)} duong vien hop le")

    # Vẽ đường viền
    anh_vien = ve_duong_vien(anh, duong_vien)

    # Lấy bounding box
    bbox = lay_bounding_box(duong_vien)
    if bbox:
        x, y, w, h = bbox
        cv2.rectangle(anh_vien, (x, y), (x + w, y + h), (0, 0, 255), 2)
        print(f"Bounding box: x={x}, y={y}, w={w}, h={h}")

    # Hiển thị
    cv2.imshow("Anh goc", anh)
    cv2.imshow("Canh Canny", anh_canh)
    cv2.imshow("Duong vien + BBox", anh_vien)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
