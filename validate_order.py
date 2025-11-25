import os
import pandas as pd
from PIL import Image
import torch
import requests
from io import BytesIO
from transformers import Owlv2Processor, Owlv2ForObjectDetection

# --- CONFIGURATION ---
CACHE_DIR = "temp_reference_images"  # Where we store downloaded images
AMAZON_URL_TEMPLATE = "https://images-na.ssl-images-amazon.com/images/P/{}.01._SCLZZZZZZZ_.jpg"

# Ensure cache directory exists
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# --- 1. Load Model (Happens once) ---
print("Loading OWLv2 Model...")
processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
print("Model Loaded.")

# --- 2. Load Name Map ---
MAP_FILE = 'asin_to_name_map.csv'
if os.path.exists(MAP_FILE):
    asin_map = pd.read_csv(MAP_FILE).set_index('asin')['product_name'].to_dict()
    print(f"Loaded {len(asin_map)} product names from map.")
else:
    asin_map = {}
    print("⚠️ Warning: asin_to_name_map.csv not found. Text fallback will be limited.")

# --- HELPER: Download Image on Demand ---
def get_reference_image(asin):
    """
    Checks cache for ASIN image. If missing, downloads it.
    Returns path to image if successful, else None.
    """
    save_path = os.path.join(CACHE_DIR, f"{asin}.jpg")
    
    # 1. Check Cache
    if os.path.exists(save_path):
        return save_path
    
    # 2. Download if missing
    print(f"    ⬇️ Downloading reference for {asin}...")
    url = AMAZON_URL_TEMPLATE.format(asin)
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=3)
        
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            # Check for "blank" 1x1 pixel images (Amazon's 404)
            if img.size[0] > 10 and img.size[1] > 10:
                img.convert("RGB").save(save_path)
                return save_path
            else:
                print(f"    ⚠️ Image too small/blank for {asin}")
    except Exception as e:
        print(f"    ❌ Download failed for {asin}: {e}")
        
    return None

# --- 3. Main Validation Function ---
def validate_order(image_path: str, order_dict: dict, 
                   img_conf=0.35, txt_conf=0.05):
    
    try:
        target_image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        return f"Error: Bin image not found at {image_path}"

    print(f"\nScanning bin: {image_path}")
    
    validation_messages = []
    all_verified = True
    target_sizes = torch.Tensor([target_image.size[::-1]])

    for asin, qty_required in order_dict.items():
        item_name = asin_map.get(asin, asin)
        print(f"  Checking item: {asin} (Required: {qty_required})")
        
        found_count = 0
        
        # --- PHASE 1: FETCH & IMAGE SEARCH ---
        ref_img_path = get_reference_image(asin) # <--- Dynamic Fetch!
        
        if ref_img_path:
            try:
                query_image = Image.open(ref_img_path).convert("RGB")
                inputs = processor(images=target_image, query_images=query_image, return_tensors="pt")
                
                # Only attempt image search if the specific method exists
                if hasattr(model, 'image_guided_object_detection'):
                    with torch.no_grad():
                        outputs = model.image_guided_object_detection(**inputs)
                    
                    results = processor.post_process_image_guided_detection(
                        outputs=outputs, 
                        threshold=img_conf, 
                        nms_threshold=0.3, 
                        target_sizes=target_sizes
                    )
                    found_count = len(results[0]["boxes"])
                    print(f"    > Image Search found: {found_count}")
            except Exception:
                # If ANY error occurs (missing method, wrong arguments), skip silently
                pass

        else:
            print(f"    > Reference image unavailable. Skipping Image Search.")

        # --- PHASE 2: TEXT FALLBACK ---
        if found_count < qty_required:
            # Truncate name to prevent token errors
            short_name = " ".join(item_name.split()[:5])
            text_query = f"a photo of {short_name}"
            
            print(f"    > Fallback: Text Search for '{short_name}...'")
            
            try:
                inputs = processor(text=[text_query], images=target_image, return_tensors="pt", 
                                   truncation=True, max_length=16)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                
                results = processor.post_process_object_detection(
                    outputs=outputs, threshold=txt_conf, target_sizes=target_sizes
                )
                
                text_found_count = len(results[0]["boxes"])
                print(f"    > Text Search found: {text_found_count}")
                
                # Take the best result
                found_count = max(found_count, text_found_count)
                
            except Exception as e:
                 print(f"    > Text Search failed: {e}")

        # --- VERDICT ---
        if found_count >= qty_required:
            msg = f"✅ Verified {asin}: Found {found_count}/{qty_required}"
            validation_messages.append(msg)
            print(msg)
        else:
            msg = f"❌ Mismatch {asin}: Found {found_count}/{qty_required}"
            validation_messages.append(msg)
            print(msg)
            all_verified = False

    status = "✅ ORDER VERIFIED" if all_verified else "❌ ORDER MISMATCH"
    return f"{status}\n" + "\n".join(validation_messages)

# --- 4. Test Block ---
