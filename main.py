"""
main.py - Entry point chính của pipeline
==========================================
Chạy toàn bộ pipeline xử lý 1 ảnh:
  Tiền xử lý (Ch.2) → Phát hiện cạnh (Ch.3) → Phân đoạn (Ch.4) → Đánh giá

Cách dùng:
  python main.py <duong_dan_anh> [duong_dan_mask]

Ví dụ:
  python main.py du_lieu/anh_goc/ISIC_0024306.jpg
  python main.py du_lieu/anh_goc/ISIC_0024306.jpg du_lieu/mask_chuan/ISIC_0024306_segmentation.png
"""

import cv2
import os
import sys
import numpy as np

# Import các module trong project
from tien_ich import doc_anh, doc_mask, resize_giu_ti_le, luu_anh, tao_thu_muc, overlay_mask
from tien_xu_ly import tien_xu_ly
from phat_hien_canh import (phat_hien_canh_canny, tim_duong_vien,
                             loc_duong_vien, ve_duong_vien, lay_bounding_box)
from phan_doan import phan_doan_ton_thuong
from danh_gia import danh_gia_phan_doan, in_ket_qua


def xu_ly_mot_anh(duong_dan_anh, duong_dan_mask=None, thu_muc_ket_qua="ket_qua"):
    """
    Pipeline xử lý đầy đủ cho 1 ảnh tổn thương da.

    Pipeline tổng quan:
    ┌──────────────────────────────────────────────────────┐
    │  Ảnh gốc (dermoscopy)                                │
    │      │                                               │
    │      ▼                                               │
    │  [1] TIỀN XỬ LÝ (Ch.2)                              │
    │      Gaussian Blur → Gray → CLAHE                    │
    │      │                                               │
    │      ▼                                               │
    │  [2] PHÁT HIỆN CẠNH (Ch.3)                           │
    │      Canny → Contours → Bounding Box                 │
    │      │                                               │
    │      ▼                                               │
    │  [3] PHÂN ĐOẠN (Ch.4)                                │
    │      Otsu + GrabCut → Morphology → Mask cuối         │
    │      │                                               │
    │      ▼                                               │
    │  [4] ĐÁNH GIÁ (nếu có ground truth)                  │
    │      IoU, Dice, Pixel Accuracy                       │
    └──────────────────────────────────────────────────────┘
    """
    ten_anh = os.path.basename(duong_dan_anh)

    print(f"\n{'#' * 60}")
    print(f"# XU LY: {ten_anh}")
    print(f"{'#' * 60}")

    # ========================================
    # BƯỚC 0: ĐỌC ẢNH
    # ========================================
    print("\n[0] DOC ANH...")
    anh_goc = doc_anh(duong_dan_anh)
    anh = resize_giu_ti_le(anh_goc, 512)
    print(f"  Kich thuoc goc: {anh_goc.shape[1]}x{anh_goc.shape[0]}")
    print(f"  Kich thuoc resize: {anh.shape[1]}x{anh.shape[0]}")

    # Đọc ground truth mask (nếu có)
    mask_chuan = None
    if duong_dan_mask and os.path.exists(duong_dan_mask):
        mask_chuan = doc_mask(duong_dan_mask)
        mask_chuan = cv2.resize(mask_chuan, (anh.shape[1], anh.shape[0]))
        print(f"  Da doc ground truth mask")

    # ========================================
    # BƯỚC 1: TIỀN XỬ LÝ (Ch.2)
    # ========================================
    print("\n[1] TIEN XU LY (Ch.2)...")
    print("  Gaussian Blur (5x5) -> Grayscale -> CLAHE")
    anh_xam_eq, anh_mo = tien_xu_ly(anh)

    # ========================================
    # BƯỚC 2: PHÁT HIỆN CẠNH (Ch.3)
    # ========================================
    print("\n[2] PHAT HIEN CANH (Ch.3)...")
    anh_canh = phat_hien_canh_canny(anh_xam_eq, nguong_thap=50, nguong_cao=150)

    # Tìm và lọc đường viền
    duong_vien = tim_duong_vien(anh_canh)
    duong_vien = loc_duong_vien(duong_vien, dien_tich_toi_thieu=500, kich_thuoc_anh=anh.shape[:2])
    print(f"  Canny: nguong = (50, 150)")
    print(f"  Tim thay {len(duong_vien)} duong vien hop le")

    # Vẽ đường viền lên ảnh
    anh_duong_vien = ve_duong_vien(anh, duong_vien)

    # Lấy bounding box cho GrabCut
    bbox = lay_bounding_box(duong_vien)
    if bbox:
        x, y, w, h = bbox
        cv2.rectangle(anh_duong_vien, (x, y), (x + w, y + h), (0, 0, 255), 2)
        print(f"  Bounding box: x={x}, y={y}, w={w}, h={h}")

    # ========================================
    # BƯỚC 3: PHÂN ĐOẠN (Ch.4)
    # ========================================
    print("\n[3] PHAN DOAN ANH (Ch.4)...")
    mask_otsu, mask_cuoi = phan_doan_ton_thuong(anh_mo, anh_xam_eq, bbox)

    # Tạo ảnh overlay kết quả
    anh_ket_qua = overlay_mask(anh, mask_cuoi, mau=(0, 255, 0), do_trong_suot=0.3)

    # ========================================
    # BƯỚC 4: ĐÁNH GIÁ (nếu có ground truth)
    # ========================================
    ket_qua = None
    if mask_chuan is not None:
        print("\n[4] DANH GIA...")
        ket_qua = danh_gia_phan_doan(mask_cuoi, mask_chuan)
        in_ket_qua(ket_qua, ten_anh)

    # ========================================
    # LƯU KẾT QUẢ TRUNG GIAN
    # ========================================
    tao_thu_muc(thu_muc_ket_qua)
    ten = os.path.splitext(ten_anh)[0]

    luu_anh(anh,            f"{thu_muc_ket_qua}/{ten}_0_goc.png")
    luu_anh(anh_xam_eq,     f"{thu_muc_ket_qua}/{ten}_1_tien_xu_ly.png")
    luu_anh(anh_canh,       f"{thu_muc_ket_qua}/{ten}_2_canh_canny.png")
    luu_anh(anh_duong_vien, f"{thu_muc_ket_qua}/{ten}_3_duong_vien.png")
    luu_anh(mask_otsu,      f"{thu_muc_ket_qua}/{ten}_4_mask_otsu.png")
    luu_anh(mask_cuoi,      f"{thu_muc_ket_qua}/{ten}_5_mask_cuoi.png")
    luu_anh(anh_ket_qua,    f"{thu_muc_ket_qua}/{ten}_6_ket_qua.png")

    if mask_chuan is not None:
        luu_anh(mask_chuan, f"{thu_muc_ket_qua}/{ten}_7_ground_truth.png")

    print(f"\n  Da luu ket qua vao: {thu_muc_ket_qua}/")

    # ========================================
    # HIỂN THỊ KẾT QUẢ
    # ========================================
    cv2.imshow("0. Anh goc", anh)
    cv2.imshow("1. Tien xu ly (CLAHE)", anh_xam_eq)
    cv2.imshow("2. Canh Canny", anh_canh)
    cv2.imshow("3. Duong vien + BBox", anh_duong_vien)
    cv2.imshow("4. Mask Otsu", mask_otsu)
    cv2.imshow("5. Mask cuoi cung", mask_cuoi)
    cv2.imshow("6. Ket qua overlay", anh_ket_qua)

    if mask_chuan is not None:
        # Hiển thị so sánh mask dự đoán vs ground truth
        anh_gt_overlay = overlay_mask(anh, mask_chuan, mau=(255, 0, 0), do_trong_suot=0.3)
        cv2.imshow("7. Ground Truth (do)", anh_gt_overlay)

    print("\nNhan phim bat ky de dong...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return mask_cuoi, ket_qua


def xu_ly_nhieu_anh(thu_muc_anh, thu_muc_mask=None, thu_muc_ket_qua="ket_qua"):
    """
    Chạy pipeline trên nhiều ảnh trong thư mục.
    Tính và in kết quả đánh giá trung bình.
    """
    from danh_gia import danh_gia_hang_loat

    # Lấy danh sách ảnh
    cac_anh = sorted([f for f in os.listdir(thu_muc_anh)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    if not cac_anh:
        print(f"Khong tim thay anh trong: {thu_muc_anh}")
        return

    print(f"\nTim thay {len(cac_anh)} anh trong {thu_muc_anh}")
    print("=" * 60)

    tat_ca_ket_qua = []

    for i, ten_anh in enumerate(cac_anh):
        duong_dan_anh = os.path.join(thu_muc_anh, ten_anh)

        # Tìm mask tương ứng
        duong_dan_mask = None
        if thu_muc_mask:
            ten_mask = os.path.splitext(ten_anh)[0] + "_segmentation.png"
            duong_dan_mask_full = os.path.join(thu_muc_mask, ten_mask)
            if os.path.exists(duong_dan_mask_full):
                duong_dan_mask = duong_dan_mask_full

        # Chạy pipeline (không hiển thị cửa sổ)
        print(f"\n[{i + 1}/{len(cac_anh)}] {ten_anh}")

        try:
            anh = doc_anh(duong_dan_anh)
            anh = resize_giu_ti_le(anh, 512)

            anh_xam_eq, anh_mo = tien_xu_ly(anh)
            anh_canh = phat_hien_canh_canny(anh_xam_eq, 50, 150)
            duong_vien = tim_duong_vien(anh_canh)
            duong_vien = loc_duong_vien(duong_vien, 500, kich_thuoc_anh=anh.shape[:2])
            bbox = lay_bounding_box(duong_vien)

            mask_otsu, mask_cuoi = phan_doan_ton_thuong(anh_mo, anh_xam_eq, bbox)

            # Lưu kết quả
            ten = os.path.splitext(ten_anh)[0]
            luu_anh(mask_cuoi, f"{thu_muc_ket_qua}/{ten}_mask.png")
            anh_ket_qua = overlay_mask(anh, mask_cuoi)
            luu_anh(anh_ket_qua, f"{thu_muc_ket_qua}/{ten}_overlay.png")

            # Đánh giá nếu có mask chuẩn
            if duong_dan_mask:
                mask_chuan = doc_mask(duong_dan_mask)
                mask_chuan = cv2.resize(mask_chuan, (anh.shape[1], anh.shape[0]))
                kq = danh_gia_phan_doan(mask_cuoi, mask_chuan)
                tat_ca_ket_qua.append(kq)
                print(f"  IoU={kq['IoU']:.4f}  Dice={kq['Dice']:.4f}")

        except Exception as e:
            print(f"  LOI: {e}")

    # Tính trung bình
    if tat_ca_ket_qua:
        print(f"\n{'=' * 60}")
        print(f"KET QUA TRUNG BINH ({len(tat_ca_ket_qua)} anh)")
        print(f"{'=' * 60}")
        tb = danh_gia_hang_loat(tat_ca_ket_qua)
        for chi_so, gia_tri in tb.items():
            print(f"  {chi_so:20s}: {gia_tri:.4f} ({gia_tri * 100:.1f}%)")


# ========================
# ENTRY POINT
# ========================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("=" * 60)
        print("  PHAT HIEN VA PHAN VUNG TON THUONG DA")
        print("  Nhom E - Xu ly anh va Thi giac may tinh")
        print("=" * 60)
        print()
        print("Cach dung:")
        print("  1. Xu ly 1 anh:")
        print("     python main.py <duong_dan_anh> [duong_dan_mask]")
        print()
        print("  2. Xu ly nhieu anh:")
        print("     python main.py --batch <thu_muc_anh> [thu_muc_mask]")
        print()
        print("Vi du:")
        print("  python main.py du_lieu/anh_goc/ISIC_0024306.jpg")
        print("  python main.py du_lieu/anh_goc/ISIC_0024306.jpg du_lieu/mask_chuan/ISIC_0024306_segmentation.png")
        print("  python main.py --batch du_lieu/anh_goc du_lieu/mask_chuan")
        sys.exit(0)

    # Chế độ batch
    if sys.argv[1] == "--batch":
        thu_muc_anh = sys.argv[2] if len(sys.argv) > 2 else "du_lieu/anh_goc"
        thu_muc_mask = sys.argv[3] if len(sys.argv) > 3 else "du_lieu/mask_chuan"
        tao_thu_muc("ket_qua")
        xu_ly_nhieu_anh(thu_muc_anh, thu_muc_mask)
    else:
        # Chế độ 1 ảnh
        duong_dan_anh = sys.argv[1]
        duong_dan_mask = sys.argv[2] if len(sys.argv) > 2 else None
        tao_thu_muc("ket_qua")
        xu_ly_mot_anh(duong_dan_anh, duong_dan_mask)
