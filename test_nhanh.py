"""
test_nhanh.py - Test nhanh pipeline trên 1 ảnh
=================================================
Chạy: python test_nhanh.py
"""

import cv2
import os
import numpy as np

from tien_ich import doc_anh, resize_giu_ti_le, luu_anh, tao_thu_muc, overlay_mask
from tien_xu_ly import tien_xu_ly
from phat_hien_canh import (phat_hien_canh_canny, tim_duong_vien,
                             loc_duong_vien, ve_duong_vien, lay_bounding_box)
from phan_doan import phan_doan_ton_thuong


def test_pipeline():
    """Chạy test pipeline trên ảnh đầu tiên tìm thấy trong du_lieu/"""

    # Tìm ảnh trong du_lieu/
    thu_muc = "du_lieu"
    cac_anh = [f for f in os.listdir(thu_muc) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not cac_anh:
        print("Khong tim thay anh trong du_lieu/!")
        return

    # Lấy ảnh đầu tiên
    ten_anh = sorted(cac_anh)[0]
    duong_dan = os.path.join(thu_muc, ten_anh)

    print("=" * 60)
    print(f"  TEST PIPELINE: {ten_anh}")
    print("=" * 60)

    # ========================================
    # BƯỚC 0: Đọc ảnh
    # ========================================
    anh_goc = doc_anh(duong_dan)
    anh = resize_giu_ti_le(anh_goc, 512)
    print(f"\n[0] Doc anh: {anh_goc.shape[1]}x{anh_goc.shape[0]} -> resize {anh.shape[1]}x{anh.shape[0]}")

    # ========================================
    # BƯỚC 1: Tiền xử lý (Ch.2)
    # ========================================
    print("\n[1] TIEN XU LY (Ch.2): Gaussian Blur -> Gray -> CLAHE")
    anh_xam_eq, anh_mo = tien_xu_ly(anh)
    print("    OK!")

    # ========================================
    # BƯỚC 2: Phát hiện cạnh (Ch.3)
    # ========================================
    print("\n[2] PHAT HIEN CANH (Ch.3): Canny -> Contours -> BBox")
    anh_canh = phat_hien_canh_canny(anh_xam_eq, nguong_thap=50, nguong_cao=150)
    duong_vien_tho = tim_duong_vien(anh_canh)
    duong_vien = loc_duong_vien(duong_vien_tho, dien_tich_toi_thieu=500, kich_thuoc_anh=anh.shape[:2])
    print(f"    Canny: nguong = (50, 150)")
    print(f"    Tim thay {len(duong_vien_tho)} contour tho -> {len(duong_vien)} hop le")

    # Vẽ: vàng mỏng (tất cả) + xanh dày (hợp lệ) + đỏ (bbox)
    anh_duong_vien = ve_duong_vien(anh, duong_vien_tho, mau=(0, 255, 255), do_day=1)
    anh_duong_vien = ve_duong_vien(anh_duong_vien, duong_vien, mau=(0, 255, 0), do_day=3)
    bbox = lay_bounding_box(duong_vien)
    if bbox:
        x, y, w, h = bbox
        cv2.rectangle(anh_duong_vien, (x, y), (x + w, y + h), (0, 0, 255), 3)
        print(f"    BBox: x={x}, y={y}, w={w}, h={h}")

    # ========================================
    # BƯỚC 3: Phân đoạn (Ch.4)
    # ========================================
    print("\n[3] PHAN DOAN (Ch.4): Otsu + GrabCut -> Morphology")
    mask_otsu, mask_cuoi = phan_doan_ton_thuong(anh_mo, anh_xam_eq, bbox)
    print("    OK!")

    # Tạo overlay kết quả
    anh_ket_qua = overlay_mask(anh, mask_cuoi, mau=(0, 255, 0), do_trong_suot=0.4)

    # ========================================
    # Lưu kết quả
    # ========================================
    thu_muc_kq = "ket_qua"
    tao_thu_muc(thu_muc_kq)
    ten = os.path.splitext(ten_anh)[0]

    luu_anh(anh,            f"{thu_muc_kq}/{ten}_0_goc.png")
    luu_anh(anh_xam_eq,     f"{thu_muc_kq}/{ten}_1_tien_xu_ly.png")
    luu_anh(anh_canh,       f"{thu_muc_kq}/{ten}_2_canh_canny.png")
    luu_anh(anh_duong_vien, f"{thu_muc_kq}/{ten}_3_duong_vien.png")
    luu_anh(mask_otsu,      f"{thu_muc_kq}/{ten}_4_mask_otsu.png")
    luu_anh(mask_cuoi,      f"{thu_muc_kq}/{ten}_5_mask_cuoi.png")
    luu_anh(anh_ket_qua,    f"{thu_muc_kq}/{ten}_6_ket_qua.png")

    print(f"\n[OK] Da luu 7 anh ket qua vao: {thu_muc_kq}/")
    print(f"     {ten}_0_goc.png        - Anh goc (resize)")
    print(f"     {ten}_1_tien_xu_ly.png - Sau Gaussian + CLAHE")
    print(f"     {ten}_2_canh_canny.png - Canh Canny")
    print(f"     {ten}_3_duong_vien.png - Duong vien + BBox")
    print(f"     {ten}_4_mask_otsu.png  - Mask Otsu")
    print(f"     {ten}_5_mask_cuoi.png  - Mask cuoi (Otsu+GrabCut+Morph)")
    print(f"     {ten}_6_ket_qua.png    - Overlay ket qua")

    print("\n" + "=" * 60)
    print("  PIPELINE CHAY THANH CONG!")
    print("=" * 60)


if __name__ == "__main__":
    test_nhanh = test_pipeline()
