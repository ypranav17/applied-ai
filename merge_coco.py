# merge_coco.py
import json
import os
import glob
from collections import defaultdict

# --- CONFIGURATION ---
# 1. Directory containing your separate COCO JSON files
#    (e.g., the folder where label-1_annotations.json, label-2_annotations.json, etc. are)
ANNOTATIONS_DIR = "."  # Use "." if they are in the current directory

# 2. Name for the output merged COCO JSON file
OUTPUT_JSON_FILE = "merged_annotations.json"

# 3. Define your final class names IN THE ORDER YOU WANT (0-indexed for YOLO later)
#    Make sure 'unknown' is the last one if you want it to have the highest index.
FINAL_CLASS_NAMES = [
    "B013UJS35G",   # Example: Make sure this matches your actual ASINs
    "B00O4OR4GQ",   # Example
    "B00006JNK2",   # Example
    "B00O7CM12W",   # Example
    "B01ATYIK7G",   # Example
    "unknown"       # Your 6th class
]
# --- END CONFIGURATION ---

def merge_coco_files(input_dir, output_file, final_class_names):
    """
    Merges multiple COCO JSON annotation files (one per class) into a single file.
    Re-indexes image IDs and category IDs.
    """
    merged_data = {
        "info": {"description": "Merged COCO Dataset"},
        "images": [],
        "annotations": [],
        "categories": []
    }

    global_img_id = 1
    global_ann_id = 1
    image_map = {}  # Maps (original_file_index, original_image_id) -> new_global_img_id
    file_img_map = defaultdict(dict) # Maps original_filename -> img_info
    
    # --- Create the final category list ---
    final_category_map = {} # Maps name -> final_id (0-indexed)
    for i, name in enumerate(final_class_names):
        merged_data["categories"].append({"id": i + 1, "name": name, "supercategory": ""}) # COCO is 1-indexed
        final_category_map[name] = i + 1
    print(f"Final categories (COCO 1-indexed): {merged_data['categories']}")

    # --- Process each JSON file ---
    json_files = sorted(glob.glob(os.path.join(input_dir, 'label*annotations.json')))
    print(f"Found {len(json_files)} annotation files to merge.")

    processed_filenames = set()

    for file_idx, json_path in enumerate(json_files):
        print(f"Processing file {file_idx + 1}/{len(json_files)}: {json_path}")
        with open(json_path, 'r') as f:
            data = json.load(f)

        # --- Process Images ---
        for img_info in data.get('images', []):
            original_img_id = img_info['id']
            img_filename = img_info['file_name']

            # Only add the image if we haven't seen this filename before
            if img_filename not in processed_filenames:
                new_img_info = img_info.copy()
                new_img_info['id'] = global_img_id
                merged_data['images'].append(new_img_info)
                image_map[(file_idx, original_img_id)] = global_img_id
                file_img_map[img_filename] = new_img_info # Store info by filename
                processed_filenames.add(img_filename)
                global_img_id += 1
            else:
                 # If we've seen the filename, map original ID to the existing global ID
                 existing_img_info = file_img_map[img_filename]
                 image_map[(file_idx, original_img_id)] = existing_img_info['id']

        # --- Process Categories (to find the name associated with category_id 1 in *this* file) ---
        current_categories = {cat['id']: cat['name'] for cat in data.get('categories', [])}
        if 1 not in current_categories:
             if len(current_categories) > 0 and 2 in current_categories and current_categories[2] == 'unknown':
                 # Handle files that might only contain 'unknown' annotations
                 current_class_name = 'unknown'
                 original_cat_id_to_use = 2
             else:
                 print(f"  Warning: No category with ID 1 found in {json_path}. Skipping annotations.")
                 continue # Skip annotations if category 1 isn't defined
        else:
            current_class_name = current_categories[1]
            original_cat_id_to_use = 1 # We assume the object of interest is ID 1

        # Check for 'unknown' class (assuming it's ID 2 if present)
        unknown_class_name = None
        original_unknown_cat_id = None
        if 2 in current_categories and current_categories[2] == 'unknown':
             unknown_class_name = 'unknown'
             original_unknown_cat_id = 2


        if current_class_name not in final_category_map:
            print(f"  Warning: Class '{current_class_name}' from {json_path} not in FINAL_CLASS_NAMES. Skipping its annotations.")
            # We might still process 'unknown' if it exists
            if unknown_class_name is None:
                 continue


        # --- Process Annotations ---
        for ann_info in data.get('annotations', []):
            original_img_id = ann_info['image_id']
            original_cat_id = ann_info['category_id']

            # Find the new global image ID
            img_map_key = (file_idx, original_img_id)
            if img_map_key not in image_map:
                print(f"  Warning: Could not find image ID {original_img_id} from file {file_idx}. Skipping annotation {ann_info.get('id', 'N/A')}.")
                continue
            new_global_img_id = image_map[img_map_key]

            # Determine the final category ID based on the annotation's original ID
            final_cat_id = None
            if original_cat_id == original_cat_id_to_use and current_class_name in final_category_map:
                 final_cat_id = final_category_map[current_class_name]
            elif original_cat_id == original_unknown_cat_id and unknown_class_name == 'unknown':
                 final_cat_id = final_category_map['unknown']


            # If we identified a valid final category, add the annotation
            if final_cat_id is not None:
                new_ann_info = ann_info.copy()
                new_ann_info['id'] = global_ann_id
                new_ann_info['image_id'] = new_global_img_id
                new_ann_info['category_id'] = final_cat_id # Use the re-mapped category ID
                merged_data['annotations'].append(new_ann_info)
                global_ann_id += 1
            # else: # Optional: Warn about skipped annotations due to category mismatch
            #     print(f"  Skipping annotation {ann_info.get('id', 'N/A')} for category {original_cat_id} as it wasn't mapped.")


    print(f"\nMerging complete. Found {len(merged_data['images'])} unique images and {len(merged_data['annotations'])} annotations.")
    
    # --- Save the merged file ---
    print(f"Saving merged COCO JSON to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=4)

    print("✅ Success!")


if __name__ == "__main__":
    # IMPORTANT: Verify FINAL_CLASS_NAMES list order before running!
    if len(FINAL_CLASS_NAMES) != 6:
        print("Error: FINAL_CLASS_NAMES should contain exactly 6 classes (5 known + 1 unknown).")
    else:
        merge_coco_files(ANNOTATIONS_DIR, OUTPUT_JSON_FILE, FINAL_CLASS_NAMES)