# SkyCargo Table OCR Project

This project aims to digitize complex tables from image scans (e.g., SkyCargo manifests) into structured CSV data. The solution has evolved through several phases to achieve high accuracy.

## 🚀 Evolution of the OCR Process

We explored multiple approaches to handle the complex layout and noise in the source images.

### Phase 1: PaddleOCR & PP-Structure ([ocr_ppstructure_filtered.py](ocr_ppstructure_filtered.py))
- **Approach:** Used [PaddleOCR's PP-Structure](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppstructure/README.md) to automatically detect table borders and cells.
- **Outcome:** Worked well for clean, high-contrast tables but failed on images with faint grid lines or complex headers. It often merged columns incorrectly.

### Phase 2: OpenCV Heuristics + PaddleOCR ([opencv_table_ocr.py](opencv_table_ocr.py))
- **Approach:** 
    1.  Used OpenCV for image preprocessing (contrast enhancement, binarization).
    2.  Detected horizontal and vertical lines using morphological operations to build a "grid mask".
    3.  Extracted cells based on contour detection and passed each cell to PaddleOCR.
- **Outcome:** Extremely brittle. If a line was broken or faint (common in scans), the grid detection failed, causing data misalignment.

### Phase 3: Row-Coordinate Filtering ([ocr_row_filter_table.py](ocr_row_filter_table.py))
- **Approach:** 
    1.  Abandoned grid detection.
    2.  Used PaddleOCR to get all text boxes.
    3.  Manually defined **Y-coordinate ranges** (e.g., `0.14` to `0.20` of image height) to group text into rows.
- **Outcome:** Improved text capture but failed if the image was slightly zoomed, rotated, or if row heights varied.

### Phase 4: Azure OCR with Regex Parsing ([azure_table_ocr_to_csv_old.py](azure_table_ocr_to_csv_old.py))
- **Approach:** 
    1.  Switched to **Azure Computer Vision (Read API)** for superior text recognition accuracy compared to PaddleOCR.
    2.  Reconstructed tables by splitting lines using Regex `re.split(r"\s{2,}|\t", line)`.
- **Outcome:** Better text accuracy, but **structural failure**. Empty cells caused columns to shift left, ruining the data integrity.

### Phase 5: Azure OCR + Bounding Box Column Mapping

- **Scripts:** 
    - [azure_table_ocr_to_csv.py](azure_table_ocr_to_csv.py) (uses manual JSON)
    - [workflow_table_ocr.py](workflow_table_ocr.py) (automates line detection + OCR)
- **Approach:**
    1.  **Column Mapping:** We pre-defined specific X-coordinate boundaries for columns in `merged_vertical_lines.json`.
    2.  **Spatial Assignment:** Instead of relying on text layout, every recognized word is assigned to a column based on its **center X-coordinate**.
    3.  **Headers:** dynamically used from the first image; subsequent images are appended.
- **Outcome:** **Success.** This method handles empty cells perfectly (data just doesn't get placed in that column bucket) and is robust to minor shifts.

### Phase 6: Google Gemini OCR (**Current Solution**)
- **Script:** [gemini_ocr_to_csv.py](gemini_ocr_to_csv.py)
- **Approach:** 
    1.  **AI Vision:** Uses Google's Gemini Pro/Flash models to directly "see" and understand the table structure.
    2.  **Prompt Engineering:** Instructs the model to extract the data directly into CSV format, bypassing manual coordinate mapping.
- **Outcome:** **State of the Art.** Extremely robust to layout changes, requires zero manual line configuration, and handles Japanese text and complex layouts natively.

---

## 🛠️ How to Run the Current Solution (Gemini)

1.  **Prerequisites:**
    *   Python 3.10+ (Recommended in `conda` env)
    *   `.env` file with `GEMINI_API_KEY`

2.  **Install Dependencies:**
    ```bash
    conda create -n ocr python=3.10
    conda activate ocr
    pip install -r requirements.txt
    ```

3.  **Run the Script:**
    Populate the `pictures/` folder with `.jpeg` images and run:
    ```bash
    python gemini_ocr_to_csv.py
    ```

4.  **Output:**
    *   `output_gemini.csv`: The final consolidated data file.

---

## 🛠️ How to Run the Legacy Solution (Azure)

1.  **Prerequisites:**
    *   Python 3.8+
    *   `.env` file with `AZURE_CV_KEY1` and `AZURE_CV_ENDPOINT`
    *   `merged_vertical_lines_*.json` (generated column boundaries)

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Script:**
    Populate the `pictures/` folder with `.jpeg` images and run:
    ```bash
    python azure_table_ocr_to_csv.py
    ```

4.  **Output:**
    *   `output_combined_bbox.csv`: The final consolidated data file.
