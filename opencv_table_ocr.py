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


for image_file in os.listdir(image_dir):
    if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    image_path = os.path.join(image_dir, image_file)
    img = cv2.imread(image_path, 0)
    img_color = cv2.imread(image_path)
    img_bin = 255 - cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2
    )

    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (img.shape[1] // 30, 1)
    )
    horizontal_lines = cv2.erode(img_bin, horizontal_kernel, iterations=3)
    horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=3)

    # Find horizontal line positions
    lines = cv2.HoughLinesP(
        horizontal_lines,
        1,
        np.pi / 180,
        threshold=100,
        minLineLength=img.shape[1] // 2,
        maxLineGap=20,
    )
    y_list = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            y_list.append(y1)
    y_list = sorted(list(set(y_list)))
    # 過濾掉太接近的線
    filtered_y = []
    for y in y_list:
        if not filtered_y or abs(y - filtered_y[-1]) > 10:
            filtered_y.append(y)

    # Debug: save image with detected row lines
    save_debug_rows_image(img_color, filtered_y, image_file)

    # Debug: save image with all cell boxes
    save_debug_cells_image(img_color, filtered_y, NUM_COLS, image_file)

    # 以橫線分割行，再等分欄
    table = []
    for i in range(len(filtered_y) - 1):
        y1, y2 = filtered_y[i], filtered_y[i + 1]
        row_cells = []
        row_height = y2 - y1
        col_width = img.shape[1] // NUM_COLS
        for c in range(NUM_COLS):
            x1 = c * col_width
            x2 = (c + 1) * col_width if c < NUM_COLS - 1 else img.shape[1]
            cell_img = img_color[y1:y2, x1:x2]
            # Debug: save each cell image
            cell_debug_path = os.path.join(
                output_dir, f"debug_cell_{image_file}_row{i}_col{c}.jpg"
            )
            cv2.imwrite(cell_debug_path, cell_img)
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
