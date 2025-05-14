import cv2
import numpy as np
from paddleocr import PaddleOCR
import openpyxl
import os
import re
import csv

# 初始化OCR
ocr = PaddleOCR(use_angle_cls=True, lang="japan")  # 你可以换成'ch'或'en'试试

# 获取pictures文件夹下的所有图片
pictures_dir = "pictures"
image_files = [
    f for f in os.listdir(pictures_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

for img_file in image_files:
    img_path = os.path.join(pictures_dir, img_file)
    print(f"Processing {img_path}...")

    # 1. 读入图片并预处理
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. 提取水平和垂直线
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_v)

    # 3. 合并线条，找到交点
    table_mask = cv2.add(horizontal, vertical)
    cv2.imwrite(f"table_mask_{img_file}.jpg", table_mask)  # 保存表格掩码用于调试

    # 4. 行列投影，找到分割线
    h_proj = np.sum(horizontal, axis=1)
    v_proj = np.sum(vertical, axis=0)

    def find_peaks(proj, min_dist=10, min_val=10):
        peaks = []
        last = -min_dist
        for i, val in enumerate(proj):
            if val > min_val and i - last > min_dist:
                peaks.append(i)
                last = i
        return peaks

    rows = find_peaks(h_proj)
    cols = find_peaks(v_proj)

    # 5. 切割单元格并OCR
    wb = openpyxl.Workbook()
    ws = wb.active

    for i in range(len(rows) - 1):
        row_data = []
        for j in [0, 1, 4, 5, 6]:  # 只保留第1、2、5、6、7欄
            if j >= len(cols) - 1:
                row_data.append("")
                continue
            cell = img[rows[i] : rows[i + 1], cols[j] : cols[j + 1]]
            result = ocr.ocr(cell, cls=True)
            text = ""
            if result and result[0]:
                text = result[0][0][1][0]
            # 過濾內容
            if j == 0:
                # 只保留英文字母
                text = "".join(re.findall(r"[A-Za-z]+", text))
            else:
                # 只保留數字
                text = "".join(re.findall(r"[0-9]+", text))
            row_data.append(text)
        ws.append(row_data)

    # 保存Excel文件，文件名与图片文件名对应
    excel_file = f"output_{os.path.splitext(img_file)[0]}.xlsx"
    wb.save(excel_file)
    print(f"Saved results to {excel_file}")

    # 保存CSV文件
    csv_file = f"output_{os.path.splitext(img_file)[0]}.csv"
    with open(csv_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        for row in ws.iter_rows(values_only=True):
            writer.writerow(row)
    print(f"Saved results to {csv_file}")
