# OCR_for_FB.py 使用說明

## 功能簡介

本腳本會自動將 `socialMedia/image.png` 這張圖片送到 Azure OCR 服務，辨識出所有圖片上的文字，並將結果：

- 顯示在螢幕上
- 存成 `socialMedia/image_ocr.txt` 文字檔

## 使用前準備

1. 你必須在專案根目錄下有 `.env` 檔案，內容需包含：
   ```env
   AZURE_CV_KEY1=你的Azure金鑰
   AZURE_CV_ENDPOINT=你的Azure端點網址
   ```
2. 確認 `socialMedia/image.png` 已經存在。
3. 安裝必要套件（只需執行一次）：
   ```bash
   pip install requests python-dotenv
   ```

## 執行方式

在命令列輸入：

```bash
python OCR_for_FB.py
```

## 輸入說明

- 預設會讀取 `socialMedia/image.png` 這張圖片。

## 輸出說明

- 結果會顯示在螢幕上。
- 結果同時會存成 `socialMedia/image_ocr.txt`。

## 常見問題

- 如果出現金鑰或端點錯誤，請檢查 `.env` 檔案內容。
- 如果圖片不存在，請確認 `socialMedia/image.png` 是否正確放置。
- 若 Azure OCR 辨識失敗，請檢查網路連線與 Azure 配額。

---

如有其他需求，請再告訴我！
