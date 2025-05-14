import os
import requests
import pandas as pd
from dotenv import load_dotenv
import json
import re

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("AZURE_CV_KEY1") or os.getenv("AZURE_CV_KEY2")
ENDPOINT = os.getenv("AZURE_CV_ENDPOINT")
OCR_URL = ENDPOINT.rstrip("/") + "/vision/v3.2/read/analyze"

# List all .jpeg images in the 'pictures' directory
directory = "pictures"
images = [
    os.path.join(directory, f)
    for f in os.listdir(directory)
    if f.lower().endswith(".jpeg")
]

# Dynamically determine the merged_x JSON filename based on the first image
if not images:
    raise FileNotFoundError("No .jpeg images found in the 'pictures' directory.")
first_image = os.path.basename(sorted(images)[0])
base_name = os.path.splitext(first_image)[0]
MERGED_X_JSON = f"merged_vertical_lines_{base_name}.json"

# Load merged vertical x positions (column boundaries)
with open(MERGED_X_JSON, "r") as f:
    merged_x = json.load(f)

# Use all columns
TARGET_COLUMNS = list(range(len(merged_x) - 1))


def azure_ocr(image_path):
    """Send image to Azure OCR and return the result."""
    with open(image_path, "rb") as f:
        headers = {
            "Ocp-Apim-Subscription-Key": API_KEY,
            "Content-Type": "application/octet-stream",
        }
        response = requests.post(OCR_URL, headers=headers, data=f)
        response.raise_for_status()
        # Get operation-location (URL with result)
        operation_url = response.headers["Operation-Location"]

    # Poll for result
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
    """Assign a word to a column index based on its x coordinate and merged_x boundaries."""
    for i in range(len(merged_x) - 1):
        if merged_x[i] <= x < merged_x[i + 1]:
            return i
    # If x is beyond the last boundary, assign to the last column
    return len(merged_x) - 2


def extract_table_by_bbox(ocr_result, merged_x, target_columns):
    """Extract table using word bounding boxes and merged_x column boundaries."""
    # Collect all words with their bounding box center
    words = []
    for read_result in ocr_result["analyzeResult"]["readResults"]:
        for line in read_result["lines"]:
            for word in line["words"]:
                xs = word["boundingBox"][0::2]
                ys = word["boundingBox"][1::2]
                x_center = int(sum(xs) / 4)
                y_center = int(sum(ys) / 4)
                words.append({"text": word["text"], "x": x_center, "y": y_center})
    # Group words by y (rows) using a threshold
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
    # Assign words to columns
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
    """For columns whose header does not start with '名', keep only pure numbers (including decimal, negative, percent) in those columns."""
    for col_idx, col_name in enumerate(header):
        if not col_name.startswith("名"):
            for row in table:
                # Keep only numbers, decimal points, negative sign, and percent sign
                match = re.findall(r"-?\d+(?:\.\d+)?%?", row[col_idx])
                row[col_idx] = " ".join(match)
    return table


def save_csv(table, output_path):
    """Save table to CSV."""
    df = pd.DataFrame(table)
    df.to_csv(output_path, index=False, header=False, encoding="utf-8-sig")


if __name__ == "__main__":
    all_tables = []
    header = None
    leading_empty = 0
    for idx, img in enumerate(images):
        ocr_result = azure_ocr(img)
        table = extract_table_by_bbox(ocr_result, merged_x, TARGET_COLUMNS)
        if idx == 0:
            # Use the 2nd row as header
            header = table[1]
            # Count leading empty columns
            leading_empty = 0
            for cell in header:
                if cell.strip() == "":
                    leading_empty += 1
                else:
                    break
            # Remove leading empty columns from header
            header = header[leading_empty:]
        # Remove leading empty columns from all rows
        for i in range(len(table)):
            table[i] = table[i][leading_empty:]
        # Skip header rows for all but the first image
        if idx == 0:
            all_tables.extend(table)
        else:
            all_tables.extend(table[1:])  # skip first row (title/header)
    # Clean non-name columns
    all_tables = [header] + clean_non_name_columns(header, all_tables[2:])
    save_csv(all_tables, "output_combined_bbox.csv")
    print("Saved: output_combined_bbox.csv")
