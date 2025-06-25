import os
import requests
from dotenv import load_dotenv

# 載入 .env 取得 Azure OCR 金鑰與端點
load_dotenv()
API_KEY = os.getenv("AZURE_CV_KEY1") or os.getenv("AZURE_CV_KEY2")
ENDPOINT = os.getenv("AZURE_CV_ENDPOINT")
if not API_KEY or not ENDPOINT:
    raise ValueError(
        "請在 .env 檔案中設定 AZURE_CV_KEY1/AZURE_CV_KEY2 和 AZURE_CV_ENDPOINT"
    )

OCR_URL = ENDPOINT.rstrip("/") + "/vision/v3.2/read/analyze"
IMAGE_PATH = "socialMedia/image.png"
OUTPUT_TXT = "socialMedia/image_ocr.txt"


def azure_ocr(image_path):
    """將圖片送到 Azure OCR 並回傳辨識結果。"""
    with open(image_path, "rb") as f:
        headers = {
            "Ocp-Apim-Subscription-Key": API_KEY,
            "Content-Type": "application/octet-stream",
        }
        response = requests.post(OCR_URL, headers=headers, data=f)
        if response.status_code != 202:
            raise Exception(
                f"Azure OCR 請求失敗: {response.status_code} {response.text}"
            )
        operation_url = response.headers["Operation-Location"]

    # 輪詢取得結果
    import time

    for _ in range(30):  # 最多等 30 秒
        result = requests.get(
            operation_url, headers={"Ocp-Apim-Subscription-Key": API_KEY}
        )
        result_json = result.json()
        if result_json.get("status") in ["succeeded", "failed"]:
            break
        time.sleep(1)
    else:
        raise TimeoutError("Azure OCR 等待超時，請稍後再試。")
    if result_json.get("status") != "succeeded":
        raise Exception(f"Azure OCR 辨識失敗: {result_json}")
    return result_json


def extract_text(ocr_result):
    """從 Azure OCR 結果中提取所有文字。"""
    lines = []
    for read_result in ocr_result.get("analyzeResult", {}).get("readResults", []):
        for line in read_result.get("lines", []):
            lines.append(line.get("text", ""))
    return lines


def main():
    if not os.path.exists(IMAGE_PATH):
        print(f"找不到圖片檔案: {IMAGE_PATH}")
        return
    print(f"開始辨識 {IMAGE_PATH} ...")
    try:
        ocr_result = azure_ocr(IMAGE_PATH)
        lines = extract_text(ocr_result)
        print("\n--- OCR 辨識結果 ---")
        for line in lines:
            print(line)
        with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
        print(f"\n已將結果存成 {OUTPUT_TXT}")
    except Exception as e:
        print(f"辨識過程發生錯誤: {e}")


if __name__ == "__main__":
    main()
