import os
import json
import shutil
from collections import defaultdict

# --- Configuration ---

# 1. Define the ASINs (classes) you want to find and annotate.
#    I've pre-filled this with the items from 00004.json as an example.
TARGET_ASINS = [
    #"B013UJS35G"  # TaoTronics TT-AH002 30W Ultrasonic Humidifier with Cool Mist, Classic Dial Knob Control, 3.5L Large Capacity, Two 360 degree Rotatable Outlets
    #"B00O4OR4GQ",  # Fujifilm Instax Mini Instant Film (3 Twin Packs, 60 Total Pictures) Value Set
    #"B00006JNK2"  # Expo 2 Low-Odor Dry Erase Markers, Chisel Tip, 12-Pack, Black
    #"B00O7CM12W"  # Lifetime Warranty FDA cleared OTC HealthmateForever YK15AB TENS unit with 4 outputs, apply 8 pads at the same time, 15 modes Handheld Electrotherapy device | Electronic Pulse Massager for Electrotherapy Pain Management -- Pain Relief Therapy : Chosen by Sufferers of Tennis Elbow, Carpal Tunnel Syndrome, Arthritis, Bursitis, Tendonitis, Plantar Fasciitis, Sciatica, Back Pain, Fibromyalgia, Shin Splints, Neuropathy and other Inflammation Ailments Patent No. USD723178S
    "B01ATYIK7G"  # Vergiano Super Bright Flexible Dual Head Clip on LED Reading Light - Clamp Lamp Sensitive Eye Care - Rechargeable USB Desk or Table Lamp - Music Stand Light (BLACK)

# 2. Define your directories
META_DIR = "metadata1"
IMAGES_DIR = "bin-images1"
OUTPUT_DIR = "images_to_annotate5" # A new folder to store the selected images

# 3. How many images do you want to find in total?
MAX_IMAGES_TO_COLLECT = 50

# --- Main Script ---

# Create the output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}")

# Keep track of found images to avoid duplicates
found_image_files = set()
asin_counts = defaultdict(int)

print(f"Searching for images containing any of the following ASINs: {TARGET_ASINS}")

# Iterate through all metadata files
for json_filename in os.listdir(META_DIR):
    if not json_filename.endswith(".json"):
        continue
    
    # Stop if we have collected enough images
    if len(found_image_files) >= MAX_IMAGES_TO_COLLECT:
        print(f"\nReached the limit of {MAX_IMAGES_TO_COLLECT} images. Stopping.")
        break
        
    json_path = os.path.join(META_DIR, json_filename)

    with open(json_path, 'r') as f:
        try:
            data = json.load(f)
            # Get the keys (ASINs) from the bin data
            present_asins = data.get("BIN_FCSKU_DATA", {}).keys()

            # Check for a match between present ASINs and our target ASINs
            for asin in present_asins:
                if asin in TARGET_ASINS:
                    # Match found! Let's copy the corresponding image.
                    image_filename = json_filename.replace(".json", ".jpg")
                    source_image_path = os.path.join(IMAGES_DIR, image_filename)
                    destination_image_path = os.path.join(OUTPUT_DIR, image_filename)

                    # Copy the file only if it exists and we haven't already copied it
                    if os.path.exists(source_image_path) and image_filename not in found_image_files:
                        shutil.copy(source_image_path, destination_image_path)
                        found_image_files.add(image_filename)
                        print(f"  [{len(found_image_files)}/{MAX_IMAGES_TO_COLLECT}] Found match in {json_filename}. Copied {image_filename} to {OUTPUT_DIR}")
                        
                        # Update counts for our summary
                        for matched_asin in set(present_asins).intersection(TARGET_ASINS):
                            asin_counts[matched_asin] += 1

                    # Break after finding one match to avoid multiple messages for the same file
                    break 
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {json_filename}")
            continue

print("\n--- Search Complete ---")
print(f"Total unique images collected: {len(found_image_files)}")
print("Images per ASIN (an image can be counted multiple times if it contains multiple target ASINs):")
for asin, count in asin_counts.items():
    print(f"  - {asin}: {count} images")