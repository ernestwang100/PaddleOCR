import os
from paddleocr import PaddleOCR
from openpyxl import Workbook

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="japan")

# Directory containing images
image_dir = "pictures"
image_files = [
    f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

# Create a new Excel workbook
wb = Workbook()
ws = wb.active
ws.title = "OCR Results"

# Write header
ws.append(["Image", "Line No.", "Text", "Confidence"])

for img_file in image_files:
    img_path = os.path.join(image_dir, img_file)
    result = ocr.ocr(img_path, cls=True)
    line_no = 1
    for line in result:
        for bbox, (text, confidence) in line:
            ws.append([img_file, line_no, text, confidence])
            line_no += 1

# Save the Excel file
output_excel = "ocr_results.xlsx"
wb.save(output_excel)
print(f"OCR results saved to {output_excel}")
