from paddleocr import PPStructure, save_structure_res
import os

# Initialize the table structure engine
engine = PPStructure(show_log=True, lang="japan")

# Directory containing images
image_dir = "pictures"
image_files = [
    f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

# Output directory for Excel files
output_dir = "excel_output"
os.makedirs(output_dir, exist_ok=True)

for img_file in image_files:
    img_path = os.path.join(image_dir, img_file)
    result = engine(img_path)
    save_structure_res(result, output_dir, img_path, save_excel=True)
    print(f"Saved Excel for {img_file} to {output_dir}")
