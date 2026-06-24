"""
phan_doan.py - Module phân đoạn ảnh (Chương 4)
================================================
Áp dụng các kỹ thuật phân đoạn:
- Phân ngưỡng Otsu (Otsu Thresholding)
- GrabCut (Graph Cut + GMM)
- Phép toán hình thái học (Morphological Operations)
"""

import cv2
import numpy as np


# ========================
# PHÂN NGƯỠNG OTSU (Ch.3/Ch.4)
# ========================
def phan_nguong_otsu(anh_xam):
    """
    Phân ngưỡng tự động bằng phương pháp Otsu (Nobuyuki Otsu, 1979).

    Nguyên lý:
    - Xem histogram ảnh xám như 2 lớp: background (BG) và foreground (FG)
    - Tìm ngưỡng T* TỐI ĐA HÓA phương sai liên lớp (inter-class variance):

        σ²_B(T) = w_BG(T) · w_FG(T) · [μ_BG(T) − μ_FG(T)]²

      Trong đó:
        w_BG, w_FG = tỉ lệ pixel mỗi lớp
        μ_BG, μ_FG = giá trị trung bình mỗi lớp

    - Otsu thử TẤT CẢ 256 giá trị T, chọn T* cho σ²_B lớn nhất

    Ưu điểm: Hoàn toàn tự động, không cần chọn ngưỡng thủ công
    Hạn chế: Chỉ tốt khi histogram có 2 đỉnh rõ (bimodal distribution)

    Trả về:
    - nguong: giá trị ngưỡng Otsu tìm được
    - anh_nhi_phan: ảnh nhị phân (0 hoặc 255)
    """
    # THRESH_BINARY_INV: pixel ≤ T → 255 (trắng), pixel > T → 0 (đen)
    # Vì tổn thương thường TỐI hơn da bình thường
    nguong, anh_nhi_phan = cv2.threshold(
        anh_xam, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return nguong, anh_nhi_phan


# ========================
# PHÂN ĐOẠN GRABCUT (Ch.4)
# ========================
def phan_doan_grabcut(anh, bbox, so_lan_lap=5):
    """
    Phân đoạn ảnh bằng thuật toán GrabCut (Rother, Kolmogorov & Blake, 2004).

    Nguyên lý:
    ┌────────────────────────────────────────────────────────────┐
    │ 1. KHỞI TẠO:                                              │
    │    - Pixel ngoài bbox → definitely background (GC_BGD)     │
    │    - Pixel trong bbox → probably foreground (GC_PR_FGD)    │
    │                                                            │
    │ 2. FIT GMM (Gaussian Mixture Model, K=5):                  │
    │    - Mô hình hóa phân bố màu FG bằng 5 Gaussian           │
    │    - Mô hình hóa phân bố màu BG bằng 5 Gaussian           │
    │                                                            │
    │ 3. XÂY DỰNG ĐỒ THỊ (Graph):                               │
    │    - n-link: kết nối pixel lân cận                          │
    │      → Trọng số = e^(-|I_i - I_j|² / 2σ²)                 │
    │      → Pixel màu giống nhau → khó cắt (giữ cùng nhóm)     │
    │    - t-link: kết nối pixel → source (FG) / sink (BG)       │
    │      → Trọng số = -log(GMM likelihood)                     │
    │                                                            │
    │ 4. MIN-CUT / MAX-FLOW:                                     │
    │    - Tìm cắt có chi phí nhỏ nhất → phân FG/BG             │
    │    - Thuật toán Boykov-Kolmogorov (2004)                   │
    │                                                            │
    │ 5. LẶP LẠI: Cập nhật GMM → Graph Cut → ... (5-10 lần)    │
    └────────────────────────────────────────────────────────────┘

    Tham số:
    - bbox: (x, y, w, h) bounding box bao quanh vùng quan tâm
    - so_lan_lap: số lần lặp (nhiều hơn → chính xác hơn, chậm hơn)

    Trả về: mask nhị phân (0 = background, 255 = foreground)
    """
    h, w = anh.shape[:2]

    # Khởi tạo mask: tất cả pixel = probably background
    mask = np.zeros((h, w), np.uint8)

    # Mô hình GMM cho background và foreground
    # Mỗi mô hình có 65 tham số: 5 Gaussian × 13 tham số/Gaussian
    # (13 = 1 weight + 3 mean + 9 covariance cho ảnh 3 kênh)
    mo_hinh_bg = np.zeros((1, 65), np.float64)
    mo_hinh_fg = np.zeros((1, 65), np.float64)

    # Chạy GrabCut
    cv2.grabCut(
        anh,              # Ảnh đầu vào (BGR)
        mask,             # Mask (sẽ được cập nhật)
        bbox,             # Bounding box (x, y, w, h)
        mo_hinh_bg,       # Mô hình GMM background
        mo_hinh_fg,       # Mô hình GMM foreground
        so_lan_lap,       # Số lần lặp
        cv2.GC_INIT_WITH_RECT  # Chế độ: khởi tạo từ bbox
    )

    # Chuyển mask GrabCut (4 giá trị) → mask nhị phân (2 giá trị)
    # GC_BGD (0) = chắc chắn background
    # GC_FGD (1) = chắc chắn foreground
    # GC_PR_BGD (2) = có thể background
    # GC_PR_FGD (3) = có thể foreground
    # → Giữ GC_FGD và GC_PR_FGD làm foreground
    mask_nhi_phan = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255, 0
    ).astype(np.uint8)

    return mask_nhi_phan


# ========================
# LÀM SẠCH MASK - MORPHOLOGY (Ch.4)
# ========================
def lam_sach_mask(mask, kich_thuoc_kernel=5):
    """
    Làm sạch mask bằng phép toán hình thái học (Morphological Operations).

    Áp dụng 2 phép toán:

    1. OPENING (Mở) = Erosion → Dilation:
       ┌──────────────────────────────────────────┐
       │ Erosion (co):  Thu nhỏ vùng trắng        │
       │   → Xóa các điểm nhiễu nhỏ cô lập       │
       │ Dilation (giãn): Phục hồi vùng lớn       │
       │   → Vùng lớn trở lại kích thước ban đầu  │
       │ Kết quả: Loại bỏ nhiễu nhỏ               │
       └──────────────────────────────────────────┘

    2. CLOSING (Đóng) = Dilation → Erosion:
       ┌──────────────────────────────────────────┐
       │ Dilation (giãn): Mở rộng vùng trắng      │
       │   → Lấp kín các lỗ hổng nhỏ              │
       │ Erosion (co):  Thu lại biên               │
       │   → Giữ nguyên kích thước ban đầu        │
       │ Kết quả: Lấp lỗ hổng bên trong mask      │
       └──────────────────────────────────────────┘

    Kernel hình ELLIPSE: phù hợp với tổn thương da (thường tròn/oval)
    """
    # Kernel hình elip
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kich_thuoc_kernel, kich_thuoc_kernel)
    )

    # Bước 1: Opening → loại bỏ nhiễu nhỏ
    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Bước 2: Closing → lấp lỗ hổng
    mask_clean = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel)

    return mask_clean


# ========================
# PIPELINE PHÂN ĐOẠN ĐẦY ĐỦ
# ========================
def phan_doan_ton_thuong(anh, anh_xam, bbox=None):
    """
    Pipeline phân đoạn tổn thương da đầy đủ.

    Quy trình:
        anh_xam ──→ [Otsu] ──→ mask_otsu (thô)
                                    │
        anh + bbox ──→ [GrabCut] ──→ mask_grabcut
                                    │
                    [AND kết hợp] ──→ mask_ket_hop
                                    │
                    [Morphology] ──→ mask_cuoi (sạch)

    Trả về:
    - mask_otsu: mask từ Otsu (để so sánh, trực quan hóa)
    - mask_cuoi: mask cuối cùng (kết quả tốt nhất)
    """
    # Bước 1: Phân ngưỡng Otsu
    nguong, mask_otsu = phan_nguong_otsu(anh_xam)
    print(f"  Nguong Otsu tim duoc: {nguong:.1f}")

    # Bước 2: GrabCut (nếu có bounding box từ Canny contour)
    if bbox is not None:
        h, w = anh.shape[:2]
        bx, by, bw, bh = bbox

        # Kiểm tra bbox không quá lớn (> 90% ảnh)
        # GrabCut cần vùng background đủ lớn để fit GMM
        ti_le_bbox = (bw * bh) / (w * h)

        if ti_le_bbox > 0.9:
            # Bbox quá lớn → thu nhỏ lại, lấy vùng trung tâm 70% ảnh
            margin_x = int(w * 0.15)
            margin_y = int(h * 0.15)
            bbox = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)
            print(f"  BBox qua lon ({ti_le_bbox:.0%}), thu nho ve 70% vung giua")

        try:
            mask_grabcut = phan_doan_grabcut(anh, bbox)
            # Kết hợp Otsu VÀ GrabCut bằng phép AND
            # → Giữ pixel chỉ khi CẢ HAI phương pháp đều cho là foreground
            mask_ket_hop = cv2.bitwise_and(mask_otsu, mask_grabcut)
            print("  GrabCut thanh cong, da ket hop voi Otsu")
        except cv2.error as e:
            print(f"  GrabCut that bai ({e}), chi dung Otsu")
            mask_ket_hop = mask_otsu
    else:
        print("  Khong co bounding box, chi dung Otsu")
        mask_ket_hop = mask_otsu

    # Bước 3: Làm sạch mask bằng Morphology
    mask_cuoi = lam_sach_mask(mask_ket_hop)

    return mask_otsu, mask_cuoi


# ========================
# TEST MODULE
# ========================
if __name__ == "__main__":
    from tien_ich import doc_anh, resize_giu_ti_le
    from tien_xu_ly import tien_xu_ly
    from phat_hien_canh import phat_hien_canh_canny, tim_duong_vien, \
        loc_duong_vien, lay_bounding_box

    # Đọc và tiền xử lý
    anh = doc_anh("du_lieu/anh_goc/ISIC_0024306.jpg")
    anh = resize_giu_ti_le(anh, 512)
    anh_xam_eq, anh_mo = tien_xu_ly(anh)

    # Phát hiện cạnh → bounding box
    anh_canh = phat_hien_canh_canny(anh_xam_eq, 50, 150)
    duong_vien = tim_duong_vien(anh_canh)
    duong_vien = loc_duong_vien(duong_vien, 500)
    bbox = lay_bounding_box(duong_vien)

    # Phân đoạn
    mask_otsu, mask_cuoi = phan_doan_ton_thuong(anh_mo, anh_xam_eq, bbox)

    # Hiển thị
    cv2.imshow("Anh goc", anh)
    cv2.imshow("Mask Otsu", mask_otsu)
    cv2.imshow("Mask cuoi cung", mask_cuoi)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
