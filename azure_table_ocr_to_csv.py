import os
import requests
import pandas as pd
from dotenv import load_dotenv

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

# Only keep these columns (1,2,5,6,7) - 0-based index
# Adjust the indices if your table is 1-based in the OCR result
TARGET_COLUMNS = [0, 1, 4, 5, 6]


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


def extract_table_lines(ocr_result):
    """Extract lines from OCR result."""
    lines = []
    for read_result in ocr_result["analyzeResult"]["readResults"]:
        for line in read_result["lines"]:
            lines.append(line["text"])
    return lines


def parse_table(lines):
    """Parse lines into a table (list of lists)."""
    import re

    table = []
    for line in lines:
        # Split by two or more spaces, or tab
        row = re.split(r"\s{2,}|\t", line)
        table.append(row)
    return table


def filter_columns(table, target_columns):
    """Keep only target columns, fill others with empty string."""
    filtered = []
    max_cols = max(target_columns) + 1
    for row in table:
        new_row = [""] * max_cols
        for idx in target_columns:
            if idx < len(row):
                new_row[idx] = row[idx]
        filtered.append(new_row)
    return filtered


def save_csv(table, output_path):
    """Save table to CSV."""
    df = pd.DataFrame(table)
    df.to_csv(output_path, index=False, header=False, encoding="utf-8-sig")


if __name__ == "__main__":
    for img in images:
        ocr_result = azure_ocr(img)
        lines = extract_table_lines(ocr_result)
        table = parse_table(lines)
        filtered_table = filter_columns(table, TARGET_COLUMNS)
        output_csv = f"output_{os.path.basename(img).split('.')[0]}.csv"
        save_csv(filtered_table, output_csv)
        print(f"Saved: {output_csv}")
