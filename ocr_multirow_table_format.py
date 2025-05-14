import os
import re
import csv
from paddleocr import PaddleOCR
from PIL import Image

# Settings
image_dir = "pictures"
output_dir = "filtered_tables"
os.makedirs(output_dir, exist_ok=True)

# Table format: 8 columns (adjust if needed)
NUM_COLS = 8

# Row Y-ranges (fractions of image height)
row_fractions = [
    (0.08, 0.14),  # Row 1
    (0.14, 0.20),  # Row 2
    (0.32, 0.38),  # Row 5
    (0.38, 0.44),  # Row 6
    (0.44, 0.50),  # Row 7
]

ocr = PaddleOCR(use_angle_cls=True, lang="en")

for image_file in os.listdir(image_dir):
    if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    image_path = os.path.join(image_dir, image_file)
    img = Image.open(image_path)
    img_height = img.height
    row_y_ranges = [
        (int(img_height * start), int(img_height * end)) for start, end in row_fractions
    ]

    result = ocr.ocr(image_path, cls=True)

    # Prepare empty table: 5 rows x 8 columns
    table = [["" for _ in range(NUM_COLS)] for _ in range(5)]

    for line in result:
        for box, (text, conf) in line:
            y_center = (box[0][1] + box[2][1]) / 2
            x_center = (box[0][0] + box[2][0]) / 2
            for row_idx, row_range in enumerate(row_y_ranges):
                if row_range[0] <= y_center <= row_range[1]:
                    # 第一行第一欄：只取英文
                    if row_idx == 0:
                        filtered = re.findall(r"[A-Za-z]+", text)
                        if filtered:
                            # 只放在第一欄
                            table[row_idx][0] = " ".join(filtered)
                    # 2,5,6,7行：只取數字，分配到正確欄位
                    else:
                        filtered = re.findall(r"[0-9.,¥]+", text)
                        if filtered:
                            # 根據 x_center 決定欄位
                            col_idx = int((x_center / img.width) * NUM_COLS)
                            col_idx = min(max(col_idx, 0), NUM_COLS - 1)
                            table[row_idx][col_idx] = " ".join(filtered)

    # 輸出 CSV
    out_csv = os.path.join(
        output_dir, f"{os.path.splitext(image_file)[0]}_filtered.csv"
    )
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in table:
            writer.writerow(row)
    print(f"Saved filtered table for {image_file} to {out_csv}")
