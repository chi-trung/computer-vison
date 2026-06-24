"""
danh_gia.py - Module đánh giá kết quả
=======================================
Tính các chỉ số đánh giá cho bài toán phân đoạn ảnh:
- IoU (Intersection over Union)
- Dice Coefficient (F1-score cho phân đoạn)
- Pixel Accuracy
"""

import numpy as np


# ========================
# TÍNH IoU (Intersection over Union)
# ========================
def tinh_iou(mask_du_doan, mask_chuan):
    """
    Tính IoU (Intersection over Union).

    Công thức:
        IoU = |A ∩ B| / |A ∪ B|

    Trong đó:
        A = tập pixel foreground trong mask dự đoán
        B = tập pixel foreground trong mask chuẩn (ground truth)
        ∩ = giao (AND)
        ∪ = hợp (OR)

    Ý nghĩa:
    - IoU = 1.0: phân đoạn hoàn hảo (2 mask trùng khớp 100%)
    - IoU = 0.0: không trùng khớp
    - IoU > 0.5: thường được coi là "đúng" trong object detection
    - IoU > 0.7: phân đoạn tốt
    """
    # Chuyển về nhị phân (0 và 1)
    pred = (mask_du_doan > 0).astype(np.uint8)
    gt = (mask_chuan > 0).astype(np.uint8)

    # Tính giao và hợp
    giao = np.sum(pred & gt)   # Số pixel cả 2 đều = 1
    hop = np.sum(pred | gt)    # Số pixel ít nhất 1 cái = 1

    if hop == 0:
        return 0.0

    return giao / hop


# ========================
# TÍNH DICE COEFFICIENT
# ========================
def tinh_dice(mask_du_doan, mask_chuan):
    """
    Tính Dice coefficient (còn gọi là F1-score cho phân đoạn).

    Công thức:
        Dice = 2 × |A ∩ B| / (|A| + |B|)

    So sánh với IoU:
    ┌──────────┬─────────────────────────┬───────────────────────┐
    │          │ IoU                     │ Dice                  │
    ├──────────┼─────────────────────────┼───────────────────────┤
    │ Công thức│ |A∩B| / |A∪B|          │ 2|A∩B| / (|A|+|B|)   │
    │ Giá trị  │ Luôn ≤ Dice            │ Luôn ≥ IoU            │
    │ Ứng dụng │ Object detection        │ Medical segmentation  │
    └──────────┴─────────────────────────┴───────────────────────┘

    Dice phổ biến trong y tế vì trọng số vùng overlap cao hơn.
    """
    pred = (mask_du_doan > 0).astype(np.uint8)
    gt = (mask_chuan > 0).astype(np.uint8)

    giao = np.sum(pred & gt)
    tong = np.sum(pred) + np.sum(gt)

    if tong == 0:
        return 0.0

    return 2 * giao / tong


# ========================
# TÍNH PIXEL ACCURACY
# ========================
def tinh_pixel_accuracy(mask_du_doan, mask_chuan):
    """
    Tính Pixel Accuracy (độ chính xác theo pixel).

    Công thức:
        PA = (số pixel phân loại đúng) / (tổng số pixel)

    Lưu ý: PA có thể bị lệch khi vùng background lớn hơn foreground
    nhiều (class imbalance). Ví dụ: nếu 95% ảnh là background,
    model dự đoán toàn background vẫn đạt PA = 95%.
    → Nên kết hợp với IoU và Dice để đánh giá chính xác hơn.
    """
    pred = (mask_du_doan > 0).astype(np.uint8)
    gt = (mask_chuan > 0).astype(np.uint8)

    so_pixel_dung = np.sum(pred == gt)
    tong_pixel = pred.size

    return so_pixel_dung / tong_pixel


# ========================
# ĐÁNH GIÁ TỔNG HỢP
# ========================
def danh_gia_phan_doan(mask_du_doan, mask_chuan):
    """
    Đánh giá kết quả phân đoạn bằng tất cả các chỉ số.

    Trả về: dictionary chứa các chỉ số
        {
            'IoU': float,
            'Dice': float,
            'Pixel_Accuracy': float
        }
    """
    ket_qua = {
        'IoU': tinh_iou(mask_du_doan, mask_chuan),
        'Dice': tinh_dice(mask_du_doan, mask_chuan),
        'Pixel_Accuracy': tinh_pixel_accuracy(mask_du_doan, mask_chuan)
    }
    return ket_qua


# ========================
# IN KẾT QUẢ RA CONSOLE
# ========================
def in_ket_qua(ket_qua, ten_anh=""):
    """In kết quả đánh giá ra console dạng bảng."""
    print(f"\n{'=' * 50}")
    if ten_anh:
        print(f"  Danh gia: {ten_anh}")
    print(f"{'=' * 50}")
    for chi_so, gia_tri in ket_qua.items():
        thanh = '█' * int(gia_tri * 20)  # Thanh trực quan
        print(f"  {chi_so:20s}: {gia_tri:.4f} ({gia_tri * 100:.1f}%) {thanh}")
    print(f"{'=' * 50}")


# ========================
# ĐÁNH GIÁ HÀNG LOẠT
# ========================
def danh_gia_hang_loat(danh_sach_ket_qua):
    """
    Tính trung bình các chỉ số từ nhiều ảnh.

    Tham số:
        danh_sach_ket_qua: list các dict từ danh_gia_phan_doan()

    Trả về: dict chứa trung bình mỗi chỉ số
    """
    if not danh_sach_ket_qua:
        return {}

    trung_binh = {}
    for chi_so in danh_sach_ket_qua[0].keys():
        cac_gia_tri = [kq[chi_so] for kq in danh_sach_ket_qua]
        trung_binh[chi_so] = np.mean(cac_gia_tri)

    return trung_binh
