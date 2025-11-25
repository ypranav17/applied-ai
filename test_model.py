import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import sys

def predict_quantity(image_path, model, device):
    """
    Loads a single image, preprocesses it, and predicts the quantity using the trained model.
    """
    try:
        # 1. Load and open the image
        img = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file not found at '{image_path}'")
        return None

    # 2. Define the same transformations as the training set, but without random augmentations
    # It's crucial to resize and normalize exactly the same way.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # 3. Apply transformations and prepare the tensor
    img_tensor = transform(img)
    # The model expects a batch of images, so we add a "batch" dimension of size 1
    # [3, 224, 224] -> [1, 3, 224, 224]
    img_tensor = img_tensor.unsqueeze(0)

    # 4. Move the tensor to the correct device
    img_tensor = img_tensor.to(device)

    # 5. Make the prediction
    with torch.no_grad(): # Deactivates autograd for faster inference
        prediction = model(img_tensor)
    
    # The output is a tensor, we get the actual number by using .item()
    return prediction.item()

if __name__ == '__main__':
    # --- Configuration ---
    MODEL_PATH = "resnet_quantity_regression.pth"
    
    # Check if an image path is provided as a command-line argument
    if len(sys.argv) > 1:
        IMAGE_TO_TEST = sys.argv[1]
    else:
        # CHANGE THIS to the path of an image you want to test
        print("Usage: python3 test_model.py <path_to_your_image>")
        # Example:
        # IMAGE_TO_TEST = "bin-images/0001.jpg" 
        sys.exit()

    # --- Model Loading ---
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Initialize the same model architecture
    model = models.resnet34()
    # We must modify the final layer to match the one we trained (outputting 1 number)
    model.fc = nn.Linear(model.fc.in_features, 1)

    # 2. Load the saved weights from the file
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        sys.exit()

    # 3. Move the model to the device
    model.to(device)
    
    # 4. Set the model to evaluation mode
    # This is important as it disables layers like Dropout during inference.
    model.eval()

    # --- Prediction ---
    predicted_value = predict_quantity(IMAGE_TO_TEST, model, device)

    if predicted_value is not None:
        print("-" * 30)
        print(f"Image Path: {IMAGE_TO_TEST}")
        print(f"Predicted Quantity: {predicted_value:.2f}")
        print("-" * 30)