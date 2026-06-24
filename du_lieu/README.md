# Hướng Dẫn Tải Dataset

## Dataset: ISIC 2018 - Skin Lesion Analysis

### Nguồn gốc
- **ISIC** = International Skin Imaging Collaboration
- **Challenge**: ISIC 2018 Challenge - Task 1 (Lesion Boundary Segmentation)
- **Website**: https://challenge.isic-archive.com/data/#2018

### Cách tải

1. Truy cập: https://challenge.isic-archive.com/data/#2018
2. Tải về:
   - **ISIC2018_Task1-2_Training_Input** → ảnh gốc (dermoscopy images)
   - **ISIC2018_Task1_Training_GroundTruth** → mask phân đoạn (ground truth)
3. Giải nén và đặt vào thư mục:

```
du_lieu/
├── anh_goc/           ← Đặt ảnh .jpg vào đây
│   ├── ISIC_0024306.jpg
│   ├── ISIC_0024307.jpg
│   └── ...
├── mask_chuan/        ← Đặt mask .png vào đây
│   ├── ISIC_0024306_segmentation.png
│   ├── ISIC_0024307_segmentation.png
│   └── ...
└── README.md
```

### Lưu ý
- Ảnh gốc: định dạng `.jpg`, kích thước đa dạng (~600x450 trở lên)
- Mask: định dạng `.png`, nhị phân (đen = background, trắng = tổn thương)
- Tổng cộng: 2594 ảnh training
- Chỉ cần tải 20-50 ảnh mẫu để test cũng đủ

### Quy ước đặt tên
- Ảnh gốc: `ISIC_XXXXXXX.jpg`
- Mask tương ứng: `ISIC_XXXXXXX_segmentation.png`
- Tên khớp nhau qua mã `ISIC_XXXXXXX`
