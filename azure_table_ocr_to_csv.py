import os
import requests
import pandas as pd
from dotenv import load_dotenv
import json

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

# Load merged vertical x positions (column boundaries)
with open("merged_vertical_lines_IMG_6620.json", "r") as f:
    merged_x = json.load(f)

# Only keep these columns (1,2,5,6,7) - 0-based index in merged_x
# TARGET_COLUMNS = [0, 1, 4, 5, 6]
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
                # boundingBox: [x1, y1, x2, y2, x3, y3, x4, y4]
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
            # Concatenate if multiple words in the same cell
            if row_cells[col_idx]:
                row_cells[col_idx] += " " + w["text"]
            else:
                row_cells[col_idx] = w["text"]
        # Only keep target columns
        filtered_row = [
            row_cells[i] if i < len(row_cells) else "" for i in target_columns
        ]
        table.append(filtered_row)
    return table


def save_csv(table, output_path):
    """Save table to CSV."""
    df = pd.DataFrame(table)
    df.to_csv(output_path, index=False, header=False, encoding="utf-8-sig")


if __name__ == "__main__":
    for img in images:
        ocr_result = azure_ocr(img)
        table = extract_table_by_bbox(ocr_result, merged_x, TARGET_COLUMNS)
        output_csv = f"output_{os.path.basename(img).split('.')[0]}_bbox.csv"
        save_csv(table, output_csv)
        print(f"Saved: {output_csv}")
