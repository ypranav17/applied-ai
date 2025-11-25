# analyze_class_distribution.py
import os
import json
from collections import Counter
import pandas as pd

# --- Configuration ---
META_DIR = "metadata1"  # The directory with all 448,500 .json files
OUTPUT_CSV = "asin_distribution.csv"

# --- Main Script ---
print("Starting ASIN distribution analysis...")
print(f"Reading metadata from: {os.path.abspath(META_DIR)}")

# Use Counter for efficient counting
asin_counter = Counter()

# Get total number of files for progress tracking
total_files = len([name for name in os.listdir(META_DIR) if name.endswith(".json")])
processed_files = 0

for json_filename in os.listdir(META_DIR):
    if not json_filename.endswith(".json"):
        continue

    json_path = os.path.join(META_DIR, json_filename)
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            # Get the keys (ASINs) from the bin data
            present_asins = data.get("BIN_FCSKU_DATA", {}).keys()
            # Update the counter with all ASINs found in this file
            asin_counter.update(present_asins)
    except Exception as e:
        # This handles potential empty or corrupted JSON files
        # print(f"Warning: Could not process {json_filename}. Error: {e}")
        pass

    processed_files += 1
    if processed_files % 5000 == 0:
        print(f"  ...processed {processed_files}/{total_files} files...")

print("\nAnalysis complete. Generating report...")

# Convert the Counter to a pandas DataFrame for easy sorting and saving
df = pd.DataFrame(asin_counter.items(), columns=['ASIN', 'ImageCount'])
df_sorted = df.sort_values(by='ImageCount', ascending=False)

# Save the full distribution to a CSV
df_sorted.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Success! ASIN distribution saved to {OUTPUT_CSV}")
print("\n--- Top 10 Most Common Items ---")
print(df_sorted.head(10).to_string(index=False))