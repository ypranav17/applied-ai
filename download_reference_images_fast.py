import os
import csv
import requests
import time
import random
from PIL import Image
from io import BytesIO

# --- CONFIGURATION ---
CSV_FILE = "asin_to_name_map.csv" 
OUTPUT_DIR = "reference_images"
TARGET_COUNT = 200  # We will keep going until we have this many GOOD images
# ---------------------

def get_image_url(asin):
    return f"https://images-na.ssl-images-amazon.com/images/P/{asin}.01._SCLZZZZZZZ_.jpg"

def download_images_guaranteed():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Reading ASINs from {CSV_FILE}...")
    all_asins = []
    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None) 
        for row in reader:
            if row: all_asins.append(row[0])

    # Shuffle to get a random sample of items, not just the ones starting with '0' or 'A'
    random.shuffle(all_asins)
    print(f"Found {len(all_asins)} total ASINs available.")
    print(f"Aiming for {TARGET_COUNT} valid reference images...")

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36'}
    
    success_count = 0
    
    # Check what we already have in the folder
    existing_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".jpg")]
    success_count = len(existing_files)
    print(f"Found {success_count} existing images in folder.")

    # Loop through ALL asins until we hit the target
    for i, asin in enumerate(all_asins):
        
        # Stop if we hit the target
        if success_count >= TARGET_COUNT:
            break

        save_path = os.path.join(OUTPUT_DIR, f"{asin}.jpg")
        
        # Skip if file already exists (we counted it above)
        if os.path.exists(save_path):
            continue

        url = get_image_url(asin)
        
        try:
            response = requests.get(url, headers=headers, timeout=3)
            
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                
                # --- QUALITY CHECK ---
                # Must be larger than 1x1 pixel to be valid
                if img.size[0] > 10 and img.size[1] > 10:
                    img.convert("RGB").save(save_path)
                    success_count += 1
                    print(f"[{success_count}/{TARGET_COUNT}] ✅ Saved {asin} (Size: {img.size})")
                else:
                    # Silently skip bad images
                    pass
            else:
                pass # Silently skip 404s
                
        except Exception:
            pass # Silently skip errors to keep moving fast
        
        # Tiny delay to be polite
        time.sleep(0.1)

    print("-" * 30)
    if success_count >= TARGET_COUNT:
        print(f"🎉 Success! You now have {success_count} valid reference images.")
    else:
        print(f"⚠️ Stopped. Ran out of ASINs to check. Total obtained: {success_count}")

if __name__ == "__main__":
    download_images_guaranteed()