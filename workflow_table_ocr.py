import os
import cv2
import numpy as np
import json
import requests
import pandas as pd
from dotenv import load_dotenv
import re

# --- Table Line Detection ---
pictures_dir = "pictures"
image_files = [
    f for f in os.listdir(pictures_dir) if f.lower().endswith((".jpeg", ".jpg", ".png"))
]
if not image_files:
    raise FileNotFoundError("No image files found in the 'pictures' directory.")
first_image = sorted(image_files)[0]
IMAGE_PATH = os.path.join(pictures_dir, first_image)

base_name = os.path.splitext(first_image)[0]
OUTPUT_IMAGE = f"detected_lines_{base_name}.jpeg"
MERGED_X_JSON = f"merged_vertical_lines_{base_name}.json"
NODE_IMAGE = f"visualized_nodes_{base_name}.jpeg"

img = cv2.imread(IMAGE_PATH)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 150, apertureSize=3)
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

verti_lst = []
horiz_lst = []
if lines is not None:
    for i in lines:
        x1, y1, x2, y2 = i[0]
        if abs(x1 - x2) < 10:
            verti_lst.append(x1)
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if abs(y1 - y2) < 10:
            horiz_lst.append(y1)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
verti_lst = sorted(list(set(verti_lst)))
horiz_lst = sorted(list(set(horiz_lst)))
verti_lst = [int(x) for x in verti_lst]
horiz_lst = [int(y) for y in horiz_lst]
cv2.imwrite(OUTPUT_IMAGE, img)

# Node visualization
gray_img_nodes = cv2.imread(IMAGE_PATH)
for x in verti_lst:
    for y in horiz_lst:
        cv2.circle(gray_img_nodes, (x, y), 8, (0, 0, 255), -1)
        label = f"({x},{y})"
        cv2.putText(
            gray_img_nodes,
            label,
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )
MERGE_THRESHOLD = 100
merged_x = []
prev_x = None
for x in verti_lst:
    if prev_x is None or abs(x - prev_x) > MERGE_THRESHOLD:
        merged_x.append(x)
    prev_x = x
for x in merged_x:
    cv2.circle(gray_img_nodes, (x, horiz_lst[0]), 12, (0, 255, 0), 2)
    cv2.putText(
        gray_img_nodes,
        str(x),
        (x - 20, horiz_lst[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 128, 0),
        2,
        cv2.LINE_AA,
    )
cv2.imwrite(NODE_IMAGE, gray_img_nodes)
with open(MERGED_X_JSON, "w") as f:
    json.dump(merged_x, f, indent=2)
print(f"Detected lines image saved as {OUTPUT_IMAGE}")
print(f"Node visualization saved as {NODE_IMAGE}")

# --- Azure Table OCR to CSV ---
load_dotenv()
API_KEY = os.getenv("AZURE_CV_KEY1") or os.getenv("AZURE_CV_KEY2")
ENDPOINT = os.getenv("AZURE_CV_ENDPOINT")
OCR_URL = ENDPOINT.rstrip("/") + "/vision/v3.2/read/analyze"

directory = "pictures"
images = [
    os.path.join(directory, f)
    for f in os.listdir(directory)
    if f.lower().endswith(".jpeg")
]
if not images:
    raise FileNotFoundError("No .jpeg images found in the 'pictures' directory.")
first_image = os.path.basename(sorted(images)[0])
base_name = os.path.splitext(first_image)[0]
MERGED_X_JSON = f"merged_vertical_lines_{base_name}.json"
with open(MERGED_X_JSON, "r") as f:
    merged_x = json.load(f)
TARGET_COLUMNS = list(range(len(merged_x) - 1))


def azure_ocr(image_path):
    with open(image_path, "rb") as f:
        headers = {
            "Ocp-Apim-Subscription-Key": API_KEY,
            "Content-Type": "application/octet-stream",
        }
        response = requests.post(OCR_URL, headers=headers, data=f)
        response.raise_for_status()
        operation_url = response.headers["Operation-Location"]
    import time

    while True:
        result = requests.get(
            operation_url, headers={"Ocp-Apim-Subscription-Key": API_KEY}
        )
        result_json = result.json()
        if result_json["status"] in ["succeeded", "failed"]:
            break
        time.sleep(1)
    return result_json


def assign_word_to_column(x, merged_x):
    for i in range(len(merged_x) - 1):
        if merged_x[i] <= x < merged_x[i + 1]:
            return i
    return len(merged_x) - 2


def extract_table_by_bbox(ocr_result, merged_x, target_columns):
    words = []
    for read_result in ocr_result["analyzeResult"]["readResults"]:
        for line in read_result["lines"]:
            for word in line["words"]:
                xs = word["boundingBox"][0::2]
                ys = word["boundingBox"][1::2]
                x_center = int(sum(xs) / 4)
                y_center = int(sum(ys) / 4)
                words.append({"text": word["text"], "x": x_center, "y": y_center})
    words = sorted(words, key=lambda w: w["y"])
    rows = []
    row_threshold = 50
    current_row = []
    last_y = None
    for w in words:
        if last_y is None or abs(w["y"] - last_y) < row_threshold:
            current_row.append(w)
        else:
            rows.append(current_row)
            current_row = [w]
        last_y = w["y"]
    if current_row:
        rows.append(current_row)
    table = []
    for row in rows:
        row_cells = ["" for _ in range(len(merged_x) - 1)]
        for w in row:
            col_idx = assign_word_to_column(w["x"], merged_x)
            if row_cells[col_idx]:
                row_cells[col_idx] += " " + w["text"]
            else:
                row_cells[col_idx] = w["text"]
        filtered_row = [
            row_cells[i] if i < len(row_cells) else "" for i in target_columns
        ]
        table.append(filtered_row)
    return table


def clean_non_name_columns(header, table):
    for col_idx, col_name in enumerate(header):
        if not col_name.startswith("名"):
            for row in table:
                match = re.findall(r"-?\d+(?:\.\d+)?%?", row[col_idx])
                row[col_idx] = "".join(match)
    return table


def save_csv(table, output_path):
    df = pd.DataFrame(table)
    df.to_csv(output_path, index=False, header=False, encoding="utf-8-sig")


all_tables = []
header = None
leading_empty = 0
for idx, img in enumerate(images):
    ocr_result = azure_ocr(img)
    table = extract_table_by_bbox(ocr_result, merged_x, TARGET_COLUMNS)
    if idx == 0:
        header = table[1]
        leading_empty = 0
        for cell in header:
            if cell.strip() == "":
                leading_empty += 1
            else:
                break
        header = header[leading_empty:]
    for i in range(len(table)):
        table[i] = table[i][leading_empty:]
    if idx == 0:
        all_tables.extend(table)
    else:
        all_tables.extend(table[1:])
all_tables = [header] + clean_non_name_columns(header, all_tables[2:])
save_csv(all_tables, "output_combined_bbox.csv")
print("Saved: output_combined_bbox.csv")
