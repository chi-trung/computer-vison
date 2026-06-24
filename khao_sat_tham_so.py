"""
khao_sat_tham_so.py - Khảo sát tham số (Parameter Sweep)
==========================================================
Yêu cầu bắt buộc của đề bài:
- Với mỗi kỹ thuật chính, thử ít nhất 3 giá trị tham số khác nhau
- Trình bày kết quả ảnh song song, kèm nhận xét

Kỹ thuật khảo sát:
1. Gaussian Blur: kernel size (3, 5, 7, 9)
2. Canny: ngưỡng (low, high) — 3 cặp ngưỡng
3. GrabCut: số lần lặp (1, 3, 5, 10)
4. Morphology: kernel size (3, 5, 7, 9)
"""

import cv2
import os
import numpy as np

from tien_ich import doc_anh, doc_mask, resize_giu_ti_le, luu_anh, tao_thu_muc
from tien_xu_ly import lam_mo_gaussian, chuyen_xam, can_bang_histogram_clahe
from phat_hien_canh import phat_hien_canh_canny, tim_duong_vien, loc_duong_vien, lay_bounding_box
from phan_doan import phan_nguong_otsu, phan_doan_grabcut, lam_sach_mask
from danh_gia import danh_gia_phan_doan


def khao_sat_gaussian_blur(anh, anh_xam_goc, mask_chuan=None):
    """
    Khảo sát ảnh hưởng của kích thước kernel Gaussian Blur.

    Tham số khảo sát: kernel_size = 3, 5, 7, 9
    - Kernel nhỏ (3): mờ ít, giữ chi tiết, khử nhiễu yếu
    - Kernel lớn (9): mờ mạnh, mất chi tiết, khử nhiễu tốt
    """
    print("\n" + "=" * 60)
    print("KHAO SAT 1: GAUSSIAN BLUR - KICH THUOC KERNEL")
    print("=" * 60)

    kich_thuoc_list = [3, 5, 7, 9]
    ket_qua = []

    for k in kich_thuoc_list:
        # Áp dụng Gaussian Blur
        anh_mo = lam_mo_gaussian(anh, (k, k))
        anh_xam = chuyen_xam(anh_mo)
        anh_xam_eq = can_bang_histogram_clahe(anh_xam)

        # Chạy Canny để thấy ảnh hưởng
        anh_canh = phat_hien_canh_canny(anh_xam_eq, 50, 150)
        so_pixel_canh = np.sum(anh_canh > 0)

        # Phân đoạn Otsu để đo
        _, mask_otsu = phan_nguong_otsu(anh_xam_eq)
        mask_cuoi = lam_sach_mask(mask_otsu)
        so_pixel_fg = np.sum(mask_cuoi > 0)

        # Đánh giá nếu có ground truth
        iou = -1
        if mask_chuan is not None:
            kq = danh_gia_phan_doan(mask_cuoi, mask_chuan)
            iou = kq['IoU']

        ket_qua.append({
            'kernel': k,
            'anh_mo': anh_mo,
            'anh_canh': anh_canh,
            'iou': iou
        })

        iou_str = f"IoU={iou:.4f}" if iou >= 0 else ""
        print(f"  Kernel ({k}x{k}): {so_pixel_canh:6d} pixel canh | {so_pixel_fg:6d} pixel FG  {iou_str}")

    # Hiển thị so sánh
    print("\n  Nhan xet:")
    print("  - Kernel nho (3): giu nhieu chi tiet, nhung co the nhieu canh nhieu")
    print("  - Kernel trung binh (5): can bang giua khu nhieu va giu chi tiet")
    print("  - Kernel lon (7-9): mat chi tiet, nhung canh sach hon")

    return ket_qua


def khao_sat_canny(anh_xam_eq, mask_chuan=None):
    """
    Khảo sát ảnh hưởng của ngưỡng Canny.

    Tham số khảo sát: (nguong_thap, nguong_cao)
    - Ngưỡng thấp: nhạy hơn, nhiều cạnh hơn (có thể nhiễu)
    - Ngưỡng cao: chặt hơn, ít cạnh hơn (có thể mất cạnh yếu)
    - Tỉ lệ cao/thấp nên ≈ 2-3
    """
    print("\n" + "=" * 60)
    print("KHAO SAT 2: CANNY - NGUONG (LOW, HIGH)")
    print("=" * 60)

    nguong_list = [
        (20, 60),    # Ngưỡng thấp → nhiều cạnh
        (50, 150),   # Ngưỡng trung bình (mặc định)
        (100, 200),  # Ngưỡng cao → ít cạnh
        (150, 300),  # Ngưỡng rất cao → rất ít cạnh
    ]

    ket_qua = []

    for (thap, cao) in nguong_list:
        anh_canh = phat_hien_canh_canny(anh_xam_eq, thap, cao)
        so_pixel_canh = np.sum(anh_canh > 0)

        ket_qua.append({
            'nguong': (thap, cao),
            'anh_canh': anh_canh,
            'so_pixel_canh': so_pixel_canh
        })

        print(f"  Nguong ({thap:3d}, {cao:3d}): {so_pixel_canh:6d} pixel canh")

    print("\n  Nhan xet:")
    print("  - Nguong thap (20,60): phat hien nhieu canh, bao gom ca nhieu")
    print("  - Nguong (50,150): can bang, phu hop cho da so anh da")
    print("  - Nguong cao (100,200): chi giu canh manh, co the mat bien ton thuong yeu")
    print("  - Nguong rat cao (150,300): mat nhieu canh, chi giu canh rat ro rang")

    return ket_qua


def khao_sat_grabcut(anh, bbox, mask_chuan=None):
    """
    Khảo sát ảnh hưởng của số lần lặp GrabCut.

    Tham số khảo sát: so_lan_lap = 1, 3, 5, 10
    - Ít lặp (1): nhanh, chất lượng thấp
    - Nhiều lặp (10): chậm, GMM hội tụ tốt hơn
    """
    print("\n" + "=" * 60)
    print("KHAO SAT 3: GRABCUT - SO LAN LAP")
    print("=" * 60)

    if bbox is None:
        print("  Khong co bounding box, bo qua khao sat GrabCut")
        return []

    # Fix: kiểm tra bbox không quá lớn (GrabCut cần vùng BG đủ)
    h, w = anh.shape[:2]
    bx, by, bw, bh = bbox
    ti_le_bbox = (bw * bh) / (w * h)
    if ti_le_bbox > 0.9:
        margin_x = int(w * 0.15)
        margin_y = int(h * 0.15)
        bbox = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)
        print(f"  BBox qua lon ({ti_le_bbox:.0%}), thu nho ve 70% vung giua")

    so_lap_list = [1, 3, 5, 10]
    ket_qua = []

    for so_lap in so_lap_list:
        try:
            mask_gc = phan_doan_grabcut(anh, bbox, so_lap)
            so_pixel_fg = np.sum(mask_gc > 0)

            iou = -1
            if mask_chuan is not None:
                kq = danh_gia_phan_doan(mask_gc, mask_chuan)
                iou = kq['IoU']

            ket_qua.append({
                'so_lap': so_lap,
                'mask': mask_gc,
                'so_pixel_fg': so_pixel_fg,
                'iou': iou
            })

            iou_str = f"IoU={iou:.4f}" if iou >= 0 else ""
            print(f"  {so_lap:2d} lan lap: {so_pixel_fg:6d} pixel FG  {iou_str}")

        except cv2.error as e:
            print(f"  {so_lap:2d} lan lap: LOI - {e}")

    print("\n  Nhan xet:")
    print("  - 1 lap: GMM chua hoi tu, phan doan tho")
    print("  - 3 lap: GMM bat dau on dinh")
    print("  - 5 lap: thuong du tot (mac dinh)")
    print("  - 10 lap: cai thien it, tang thoi gian tinh toan")

    return ket_qua


def khao_sat_morphology(mask_tho, mask_chuan=None):
    """
    Khảo sát ảnh hưởng của kích thước kernel Morphology.

    Tham số khảo sát: kernel_size = 3, 5, 7, 9
    - Kernel nhỏ (3): xử lý nhiễu/lỗ nhỏ
    - Kernel lớn (9): xử lý nhiễu/lỗ lớn, nhưng có thể làm biến dạng biên
    """
    print("\n" + "=" * 60)
    print("KHAO SAT 4: MORPHOLOGY - KICH THUOC KERNEL")
    print("=" * 60)

    kich_thuoc_list = [3, 5, 7, 9]
    ket_qua = []

    so_pixel_goc = np.sum(mask_tho > 0)

    for k in kich_thuoc_list:
        mask_clean = lam_sach_mask(mask_tho, k)
        so_pixel_fg = np.sum(mask_clean > 0)
        thay_doi = so_pixel_fg - so_pixel_goc

        iou = -1
        if mask_chuan is not None:
            kq = danh_gia_phan_doan(mask_clean, mask_chuan)
            iou = kq['IoU']

        ket_qua.append({
            'kernel': k,
            'mask': mask_clean,
            'iou': iou
        })

        iou_str = f"IoU={iou:.4f}" if iou >= 0 else ""
        dau = "+" if thay_doi >= 0 else ""
        print(f"  Kernel ({k}x{k}): {so_pixel_fg:6d} pixel FG ({dau}{thay_doi} pixel)  {iou_str}")

    print("\n  Nhan xet:")
    print("  - Kernel nho (3): it thay doi, chi xu ly nhieu rat nho")
    print("  - Kernel (5): can bang, xu ly nhieu vua va lap lo nho")
    print("  - Kernel lon (7-9): xoa nhieu lon nhung co the lam tron/mat chi tiet bien")

    return ket_qua


# ========================
# CHẠY TẤT CẢ KHẢO SÁT
# ========================
def chay_khao_sat(duong_dan_anh, duong_dan_mask=None):
    """Chạy toàn bộ khảo sát tham số cho 1 ảnh."""

    print("\n" + "#" * 60)
    print("# KHAO SAT THAM SO (PARAMETER SWEEP)")
    print("#" * 60)

    # Đọc ảnh
    anh = doc_anh(duong_dan_anh)
    anh = resize_giu_ti_le(anh, 512)

    mask_chuan = None
    if duong_dan_mask and os.path.exists(duong_dan_mask):
        mask_chuan = doc_mask(duong_dan_mask)
        mask_chuan = cv2.resize(mask_chuan, (anh.shape[1], anh.shape[0]))

    # Tiền xử lý mặc định
    anh_xam = chuyen_xam(anh)
    anh_xam_eq = can_bang_histogram_clahe(anh_xam)

    # 1. Khảo sát Gaussian Blur
    kq_gauss = khao_sat_gaussian_blur(anh, anh_xam, mask_chuan)

    # 2. Khảo sát Canny
    kq_canny = khao_sat_canny(anh_xam_eq, mask_chuan)

    # 3. Chuẩn bị cho GrabCut
    anh_canh = phat_hien_canh_canny(anh_xam_eq, 50, 150)
    duong_vien = tim_duong_vien(anh_canh)
    duong_vien = loc_duong_vien(duong_vien, 500, kich_thuoc_anh=anh.shape[:2])
    bbox = lay_bounding_box(duong_vien)

    # Khảo sát GrabCut
    anh_mo = lam_mo_gaussian(anh, (5, 5))
    kq_grabcut = khao_sat_grabcut(anh_mo, bbox, mask_chuan)

    # 4. Khảo sát Morphology
    _, mask_otsu = phan_nguong_otsu(anh_xam_eq)
    kq_morph = khao_sat_morphology(mask_otsu, mask_chuan)

    # Lưu ảnh kết quả khảo sát
    thu_muc = "ket_qua/khao_sat"
    tao_thu_muc(thu_muc)

    # Lưu ảnh Canny
    for item in kq_canny:
        thap, cao = item['nguong']
        luu_anh(item['anh_canh'], f"{thu_muc}/canny_{thap}_{cao}.png")

    # Lưu ảnh GrabCut
    for item in kq_grabcut:
        luu_anh(item['mask'], f"{thu_muc}/grabcut_{item['so_lap']}lap.png")

    # Lưu ảnh Morphology
    for item in kq_morph:
        luu_anh(item['mask'], f"{thu_muc}/morph_k{item['kernel']}.png")

    print(f"\nDa luu ket qua khao sat vao: {thu_muc}/")
    print("\nHOAN TAT KHAO SAT THAM SO!")


# ========================
# ENTRY POINT
# ========================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Cach dung:")
        print("  python khao_sat_tham_so.py <duong_dan_anh> [duong_dan_mask]")
        print()
        print("Vi du:")
        print("  python khao_sat_tham_so.py du_lieu/anh_goc/ISIC_0024306.jpg")
        print("  python khao_sat_tham_so.py du_lieu/anh_goc/ISIC_0024306.jpg du_lieu/mask_chuan/ISIC_0024306_segmentation.png")
        sys.exit(0)

    duong_dan_anh = sys.argv[1]
    duong_dan_mask = sys.argv[2] if len(sys.argv) > 2 else None

    chay_khao_sat(duong_dan_anh, duong_dan_mask)
