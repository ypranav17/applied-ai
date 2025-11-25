import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys
import os

# --- CONFIGURATION ---
MODEL_PATH = "resnet_quantity_regression.pth"  # Ensure this is correct
IMAGE_PATH = "187477.jpg"       # Or any image file you want to test
# ---------------------

def test_resnet():
    # 1. Setup Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("✅ Model loaded.")

    # 2. Setup Image
    # Check if user provided an image arg, else use default
    img_path = sys.argv[1] if len(sys.argv) > 1 else IMAGE_PATH
    
    if not os.path.exists(img_path):
        print(f"❌ Error: Image not found at {img_path}")
        print("Usage: python test_resnet.py <path_to_image>")
        return

    print(f"Testing image: {img_path}")
    image = Image.open(img_path).convert('RGB')

    # 3. Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img_t = transform(image).unsqueeze(0).to(device)

    # 4. Predict
    with torch.no_grad():
        output = model(img_t)
        raw_count = output.item()
    
    print("-" * 30)
    print(f"Raw Prediction: {raw_count:.4f}")
    print(f"Rounded Count:  {int(round(raw_count))}")
    
    # Sanity Check Logic (same as App)
    if raw_count > 30:
        print("⚠️  Result > 30. (App would clamp this to 15)")
    elif raw_count < 0:
        print("⚠️  Result < 0. (App would clamp this to 0)")
    print("-" * 30)

if __name__ == "__main__":
    test_resnet()