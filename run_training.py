from ultralytics import YOLO
import os

# --- Configuration ---
DATA_YAML_PATH = 'bin_data.yaml'  # Make sure this file is in the same folder
MODEL_TO_USE = 'yolov8n.pt'     # Start with nano version
EPOCHS = 100
IMG_SIZE = 640
# --- End Configuration ---

def main():
    # Check if the YAML file exists
    if not os.path.exists(DATA_YAML_PATH):
        print(f"Error: Dataset YAML file not found at '{os.path.abspath(DATA_YAML_PATH)}'")
        print("Please make sure 'bin_data.yaml' is in the same directory as this script.")
        return

    # Load the model (starting from pretrained weights)
    model = YOLO(MODEL_TO_USE)

    print(f"Starting training with model '{MODEL_TO_USE}'...")
    print(f"Dataset configuration: '{os.path.abspath(DATA_YAML_PATH)}'")
    print(f"Epochs: {EPOCHS}, Image Size: {IMG_SIZE}")

    # Train the model
    try:
        model.train(
            data=DATA_YAML_PATH,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            # Add other arguments if needed, e.g., batch=16, device=0
        )
        print("\n--- Training Finished ---")
        print("Find results in the 'runs/detect/train' folder.")
    except Exception as e:
        print(f"\n--- An error occurred during training ---")
        print(e)
        # You might want to add more specific error handling here

if __name__ == '__main__':
    # This ensures the main function runs when the script is executed
    main()