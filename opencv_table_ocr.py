import cv2
import numpy as np
from paddleocr import PaddleOCR
import csv
import os

# Parameters
image_dir = "pictures"
output_dir = "opencv_table_ocr"
os.makedirs(output_dir, exist_ok=True)

ocr = PaddleOCR(use_angle_cls=True, lang="en")

NUM_COLS = 8  # 你表格的欄數


def save_debug_rows_image(img_color, filtered_y, image_file):
    debug_img = img_color.copy()
    for y in filtered_y:
        cv2.line(debug_img, (0, y), (img_color.shape[1], y), (0, 0, 255), 2)
    cv2.imwrite(f"debug_rows_{image_file}.jpg", debug_img)
    print(f"Saved debug row lines image as debug_rows_{image_file}.jpg")


def save_debug_cells_image(img_color, filtered_y, num_cols, image_file):
    debug_img = img_color.copy()
    col_width = img_color.shape[1] // num_cols
    for i in range(len(filtered_y) - 1):
        y1, y2 = filtered_y[i], filtered_y[i + 1]
        for c in range(num_cols):
            x1 = c * col_width
            x2 = (c + 1) * col_width if c < num_cols - 1 else img_color.shape[1]
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite(f"debug_cells_{image_file}.jpg", debug_img)
    print(f"Saved debug cell boxes image as debug_cells_{image_file}.jpg")


def save_debug_boxes_image(img_color, boxes, image_file):
    debug_img = img_color.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imwrite(f"debug_boxes_{image_file}.jpg", debug_img)
    print(f"Saved debug boxes image as debug_boxes_{image_file}.jpg")


for image_file in os.listdir(image_dir):
    if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    image_path = os.path.join(image_dir, image_file)
    img = cv2.imread(image_path, 0)
    img_color = cv2.imread(image_path)
    img_bin = 255 - cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2
    )

    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, img.shape[0] // 30))
    vertical_lines = cv2.erode(img_bin, vertical_kernel, iterations=3)
    vertical_lines = cv2.dilate(vertical_lines, vertical_kernel, iterations=3)

    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (img.shape[1] // 30, 1)
    )
    horizontal_lines = cv2.erode(img_bin, horizontal_kernel, iterations=3)
    horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=3)

    # Combine lines
    table_mask = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    contours, _ = cv2.findContours(table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find all cell boxes
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 1000]
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))  # sort by y, then x

    # Debug: save image with all detected boxes
    save_debug_boxes_image(img_color, boxes, image_file)

    # Group boxes into rows by y coordinate
    rows = []
    current_row = []
    last_y = -100
    for box in boxes:
        x, y, w, h = box
        if abs(y - last_y) > 10 and current_row:
            rows.append(sorted(current_row, key=lambda b: b[0]))
            current_row = []
        current_row.append(box)
        last_y = y
    if current_row:
        rows.append(sorted(current_row, key=lambda b: b[0]))

    # OCR for each cell
    table = []
    for row in rows:
        row_cells = []
        for box in row:
            x, y, w, h = box
            cell_img = img_color[y : y + h, x : x + w]
            if cell_img is None or cell_img.size == 0:
                row_cells.append("")
                continue
            result = ocr.ocr(cell_img, cls=True)
            text = ""
            if result and isinstance(result, list):
                for line in result:
                    if line and isinstance(line, list):
                        for item in line:
                            if (
                                item
                                and isinstance(item, tuple)
                                and len(item) == 2
                                and isinstance(item[1], tuple)
                                and len(item[1]) == 2
                                and isinstance(item[1][0], str)
                            ):
                                txt, conf = item[1]
                                text += str(txt) + " "
            row_cells.append(text.strip())
        table.append(row_cells)

    # Output CSV
    out_csv = os.path.join(
        output_dir, f"{os.path.splitext(image_file)[0]}_opencv_table.csv"
    )
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in table:
            writer.writerow(row)
    print(f"OCR table saved to {out_csv}")
