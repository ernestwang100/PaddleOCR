import os
import re
import csv
from paddleocr import PaddleOCR
from PIL import Image

# Image file to process
image_path = "pictures/IMG_6620.jpeg"

# Load image to get height
img = Image.open(image_path)
img_height = img.height

# Estimate row Y-ranges (manual tuning may be needed)
# These are fractions of the image height for rows 1,2,5,6,7
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


filtered_results = []
for line in result:
    for box, (text, conf) in line:
        # box: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        y_center = (box[0][1] + box[2][1]) / 2
        for idx, row_range in enumerate(row_y_ranges):
            if is_in_row(y_center, row_range):
                filtered_text = filter_english_and_digits(text)
                if filtered_text:
                    filtered_results.append((idx + 1, filtered_text, conf))

# Save to CSV
with open("filtered_ocr.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Row", "Text", "Confidence"])
    for row in filtered_results:
        writer.writerow(row)

print("Filtered OCR results saved to filtered_ocr.csv")
