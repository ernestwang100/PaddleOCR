from paddleocr import PPStructure
import os
import re
from bs4 import BeautifulSoup
import csv

# Settings
image_dir = "pictures"
output_dir = "filtered_tables_ppstructure"
os.makedirs(output_dir, exist_ok=True)

table_engine = PPStructure(show_log=True, lang="en")

# 根據表格結構，定義要過濾的行（0-based index）
# 只保留第一行第一欄英文，2,5,6,7行（1,4,5,6）只保留數字
rows_to_filter = [0, 1, 4, 5, 6]


def filter_cell(row_idx, col_idx, text):
    if row_idx == 0 and col_idx == 0:
        return " ".join(re.findall(r"[A-Za-z]+", text))
    elif row_idx in [1, 4, 5, 6]:
        return " ".join(re.findall(r"[0-9.,¥]+", text))
    else:
        return ""


for img_file in os.listdir(image_dir):
    if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    img_path = os.path.join(image_dir, img_file)
    result = table_engine(img_path)
    for table in result:
        if table["type"] == "table":
            html = table["res"]["html"]
            soup = BeautifulSoup(html, "html.parser")
            rows = soup.find_all("tr")
            filtered_table = []
            for row_idx, row in enumerate(rows):
                cells = row.find_all(["td", "th"])
                filtered_row = []
                for col_idx, cell in enumerate(cells):
                    filtered_row.append(filter_cell(row_idx, col_idx, cell.text))
                filtered_table.append(filtered_row)
            out_csv = os.path.join(
                output_dir, f"{os.path.splitext(img_file)[0]}_filtered.csv"
            )
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for row in filtered_table:
                    writer.writerow(row)
            print(f"Saved filtered PPStructure table for {img_file} to {out_csv}")
