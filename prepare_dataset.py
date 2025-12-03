import os
import json
import csv
from tqdm import tqdm  # Progress bar

METADATA_DIR = "metadata1"       # Folder with all .json files
IMAGES_DIR = "bin-images1"       # Folder with all .jpg files
OUTPUT_CSV = "full_dataset_2.csv"

def create_csv():
    print(f"Scanning {METADATA_DIR} for dataset generation...")
    
    # Get list of all json files first
    json_files = [f for f in os.listdir(METADATA_DIR) if f.endswith('.json')]
    total_files = len(json_files)
    print(f"Found {total_files} metadata files.")

    valid_pairs = 0
    
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_path', 'quantity'])  # Header

        # Use tqdm for a progress bar
        for json_file in tqdm(json_files, desc="Processing"):
            json_path = os.path.join(METADATA_DIR, json_file)
            
            # Construct corresponding image path
            # Assuming filenames match: 123.json -> 123.jpg
            img_name = json_file.replace('.json', '.jpg')
            img_path = os.path.join(IMAGES_DIR, img_name)

            # CRITICAL: Only add if the image actually exists
            if os.path.exists(img_path):
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        
                        quantity = data.get('EXPECTED_QUANTITY')
                        
                        if quantity is not None:
                            writer.writerow([img_path, quantity])
                            valid_pairs += 1
                except Exception as e:
                    pass

    print(f"\n✅ CSV Generation Complete!")
    print(f"Saved to: {OUTPUT_CSV}")
    print(f"Total valid image-label pairs: {valid_pairs}")

if __name__ == "__main__":
    create_csv()
