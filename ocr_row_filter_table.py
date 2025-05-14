import os
import re
import csv
from paddleocr import PaddleOCR
from PIL import Image

image_path = "pictures/IMG_6620.jpeg"
img = Image.open(image_path)
img_height = img.height

# Row Y-ranges (fractions of image height)
row_fractions = [
    (0.08, 0.14),  # Row 1
    (0.14, 0.20),  # Row 2
    (0.32, 0.38),  # Row 5
    (0.38, 0.44),  # Row 6
    (0.44, 0.50),  # Row 7
]
row_y_ranges = [
    (int(img_height * start), int(img_height * end)) for start, end in row_fractions
]

ocr = PaddleOCR(use_angle_cls=True, lang="en")
result = ocr.ocr(image_path, cls=True)


def is_in_row(y, row_range):
    return row_range[0] <= y <= row_range[1]


def filter_english_and_digits(text):
    return "".join(re.findall(r"[A-Za-z0-9.,¥]+", text))


# Collect cells for each row
row_cells = [[] for _ in range(len(row_y_ranges))]

for line in result:
    for box, (text, conf) in line:
        y_center = (box[0][1] + box[2][1]) / 2
        x_center = (box[0][0] + box[2][0]) / 2
        for idx, row_range in enumerate(row_y_ranges):
            if is_in_row(y_center, row_range):
                filtered_text = filter_english_and_digits(text)
                if filtered_text:
                    row_cells[idx].append((x_center, filtered_text))

# Sort by x coordinate to restore column order
for idx in range(len(row_cells)):
    row_cells[idx].sort(key=lambda x: x[0])

# Build table: only text, in order
table = [[cell[1] for cell in row] for row in row_cells]

# Output to CSV
with open("filtered_ocr_table.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    for row in table:
        writer.writerow(row)

print("Filtered OCR table saved to filtered_ocr_table.csv")
