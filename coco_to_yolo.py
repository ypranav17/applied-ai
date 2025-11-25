import json
import os
import shutil
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
# 1. Path to your COCO JSON file
JSON_FILE_PATH = r"C:\Users\ypran\Downloads\amazon dataset\merged_annotations.json" 

# 2. Path to the folder containing ALL your images
IMAGES_DIR = r"C:\Users\ypran\Downloads\amazon dataset\bin-images1"

# 3. Name of the output directory for your new dataset
OUTPUT_DATASET_DIR = r"C:\Users\ypran\Downloads\amazon dataset\training_1"
# --- END CONFIGURATION ---

def convert_coco_to_yolo(json_file, img_dir, output_dir):
    """
    Converts a COCO JSON annotation file to the YOLOv8 dataset format.
    Creates train/val splits and the required folder structure.
    """
    
    print(f"Loading COCO annotations from: {json_file}")
    with open(json_file, 'r') as f:
        data = json.load(f)

    images = data['images']
    annotations = data['annotations']
    categories = data['categories']

    # --- 1. Create Category Mapping ---
    # Creates a map of {coco_category_id: yolo_class_id}
    # YOLO class IDs are 0-indexed (0, 1, 2, 3, 4, 5)
    category_map = {cat['id']: i for i, cat in enumerate(categories)}
    
    # Create a list of category names in the correct YOLO 0-indexed order
    category_names = [cat['name'] for cat in sorted(categories, key=lambda x: category_map[x['id']])]
    
    print("Found categories:")
    for i, name in enumerate(category_names):
        print(f"  Class {i}: {name}")
        
    if len(category_names) != 6:
        print(f"Warning: Expected 6 classes, but found {len(category_names)}. Verify this is correct.")

    # --- 2. Create Train/Validation Split ---
    print(f"Splitting {len(images)} images into train and validation sets...")
    img_ids = [img['id'] for img in images]
    train_ids, val_ids = train_test_split(img_ids, test_size=0.2, random_state=42)
    
    id_to_split = {**{id: 'train' for id in train_ids}, **{id: 'val' for id in val_ids}}

    # --- 3. Create Output Directories ---
    os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'val'), exist_ok=True)

    # --- 4. Process Images and Annotations ---
    img_id_to_info = {img['id']: img for img in images}
    
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in id_to_split:
            print(f"Warning: Skipping annotation for image ID {img_id} (not in train/val split).")
            continue
            
        split = id_to_split[img_id]
        img_info = img_id_to_info[img_id]
        
        img_w = img_info['width']
        img_h = img_info['height']
        
        # COCO format: [x_min, y_min, width, height]
        bbox = ann['bbox']
        
        # YOLO format: [x_center_norm, y_center_norm, width_norm, height_norm]
        x_center = (bbox[0] + bbox[2] / 2) / img_w
        y_center = (bbox[1] + bbox[3] / 2) / img_h
        w_norm = bbox[2] / img_w
        h_norm = bbox[3] / img_h
        
        yolo_class_id = category_map[ann['category_id']]
        
        # Format the YOLO label line
        yolo_line = f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"
        
        # Write to the label file
        label_filename = os.path.splitext(img_info['file_name'])[0] + ".txt"
        label_path = os.path.join(output_dir, 'labels', split, label_filename)
        
        with open(label_path, 'a') as f:
            f.write(yolo_line)

    # --- 5. Copy Image Files ---
    print("Copying image files...")
    for img_id, split in id_to_split.items():
        img_info = img_id_to_info[img_id]
        src_path = os.path.join(img_dir, img_info['file_name'])
        dst_path = os.path.join(output_dir, 'images', split, img_info['file_name'])
        
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"Warning: Image file not found at {src_path}")

    print("\n--- Conversion Complete! ---")
    print(f"YOLO dataset created at: {os.path.abspath(output_dir)}")
    print("Next step: Create a 'data.yaml' file pointing to this directory.")
    print("Your class names are:")
    print(category_names)


if __name__ == "__main__":
    convert_coco_to_yolo(JSON_FILE_PATH, IMAGES_DIR, OUTPUT_DATASET_DIR)