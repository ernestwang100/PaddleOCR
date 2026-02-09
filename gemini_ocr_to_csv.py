
import os
import csv
import google.generativeai as genai
from dotenv import load_dotenv
import glob

# Load environment variables
load_dotenv()

# Configure Gemini API
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("Error: GEMINI_API_KEY not found in environment variables.")
    print("Please add GEMINI_API_KEY=your_key_here to your .env file.")
    exit(1)

genai.configure(api_key=API_KEY)

# Initialize the model
# Using gemini-flash-latest for better quota handling
model = genai.GenerativeModel('gemini-flash-latest') 

def extract_table_from_image(image_path):
    print(f"Processing {image_path}...")
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        prompt = """
        Extract the table data from this image.
        Output the result strictly as CSV format.
        Do not include markdown formatting like ```csv or ```.
        Return raw CSV text only.
        """
        
        response = model.generate_content([
            {'mime_type': 'image/jpeg', 'data': image_data},
            prompt
        ])
        
        return response.text.strip()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def main():
    print("Starting Gemini OCR script...", flush=True)
    # Find all images in the pictures directory
    image_files = glob.glob("pictures/*.jp*g") + glob.glob("pictures/*.png")
    
    if not image_files:
        print("No images found in 'pictures' directory.")
        return

    all_csv_data = []

    for image_file in image_files:
        csv_text = extract_table_from_image(image_file)
        if csv_text:
            # Add a newline just in case
            if all_csv_data and not all_csv_data[-1].endswith('\n'):
                 all_csv_data.append('\n')
            all_csv_data.append(csv_text + "\n")

    if all_csv_data:
        output_file = "output_gemini.csv"
        with open(output_file, "w", encoding="utf-8") as f:
            f.writelines(all_csv_data)
        print(f"Successfully saved extracted data to {output_file}")
    else:
        print("No data extracted.")

if __name__ == "__main__":
    main()
