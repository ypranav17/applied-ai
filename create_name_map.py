# create_name_map.py
import os
import json
import csv
import glob

# --- CONFIGURATION ---
# 1. Set the path to the folder containing ALL your JSON metadata files
META_DIR = "metadata1"  # Change this if your folder has a different name

# 2. Set the name for the output file
OUTPUT_CSV = "asin_to_name_map.csv"
# ---

print(f"Starting to build ASIN-to-Name map from directory: {META_DIR}")

# Use a dictionary to store the mapping {asin: name}
# This automatically handles duplicates; it will save the last name seen for any ASIN.
asin_to_name = {}

# Use glob to find all .json files in the metadata directory
json_files = glob.glob(os.path.join(META_DIR, '*.json'))
total_files = len(json_files)
print(f"Found {total_files} JSON files to process...")

processed_files = 0
for json_path in json_files:
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            bin_data = data.get("BIN_FCSKU_DATA", {})
            
            # Loop through all items listed in this bin
            for asin, details in bin_data.items():
                product_name = details.get("name")
                
                # Add the ASIN and name to our map
                if asin and product_name:
                    asin_to_name[asin] = product_name
                    
    except json.JSONDecodeError:
        print(f"Warning: Could not read (skipping) file: {json_path}")
    except Exception as e:
        print(f"Warning: Error processing file {json_path}: {e}")

    processed_files += 1
    if processed_files % 10000 == 0:  # Print progress every 10,000 files
        print(f"  ...scanned {processed_files}/{total_files} files...")

print("Scan complete. Writing map to CSV file...")

# Write the collected data to a CSV file
try:
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['asin', 'product_name'])  # Write header
        for asin, name in asin_to_name.items():
            writer.writerow([asin, name])
            
    print(f"✅ Success! Map created at: {OUTPUT_CSV}")
    print(f"Total unique products (ASINs) found: {len(asin_to_name)}")

except IOError:
    print(f"Error: Could not write to file '{OUTPUT_CSV}'. Check permissions.")