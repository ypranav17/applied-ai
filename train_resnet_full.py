import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm

# --- CONFIGURATION ---
CSV_FILE = "full_dataset_2.csv"
BATCH_SIZE = 128            # Increased for A100
EPOCHS = 15                 # 15 Epochs is usually enough for regression convergence
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "resnet_counter_best.pth"
# ---------------------

class AmazonBinDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]
        label = float(self.df.iloc[idx, 1]) # Regression label must be float

        try:
            image = Image.open(img_path).convert("RGB")
        except:
            # Fallback for corrupted images (rare but possible)
            image = Image.new('RGB', (224, 224))
        
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

def train_model():
    print(f"Using Device: {DEVICE}")
    
    # 1. Load Data
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found. Run prepare_full_dataset.py first.")
        return

    full_df = pd.read_csv(CSV_FILE)
    print(f"Total dataset size: {len(full_df)} images")

    # 2. Define Transforms
    # Standard ImageNet stats
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 3. Split Data (80% Train, 20% Val)
    dataset = AmazonBinDataset(full_df, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    # 4. Initialize Model (ResNet50)
    print("Initializing ResNet50...")
    model = models.resnet50(weights='DEFAULT')
    
    # Replace the final classification layer with a regression layer (1 output)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    
    model = model.to(DEVICE)

    # 5. Loss and Optimizer
    criterion = nn.MSELoss() # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    # 6. Training Loop
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # --- TRAIN ---
        model.train()
        train_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs).squeeze() # Remove extra dimension
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        avg_train_loss = train_loss / len(train_dataset)

        # --- VALIDATE ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
        
        avg_val_loss = val_loss / len(val_dataset)
        
        print(f"Train Loss (MSE): {avg_train_loss:.4f} | Val Loss (MSE): {avg_val_loss:.4f} | RMSE: {avg_val_loss**0.5:.2f}")

        # --- SAVE BEST ---
        if avg_val_loss < best_val_loss:
            print(f"⬇️ Validation loss decreased ({best_val_loss:.4f} -> {avg_val_loss:.4f}). Saving model...")
            torch.save(model.state_dict(), SAVE_PATH)
            best_val_loss = avg_val_loss

        scheduler.step(avg_val_loss)

    print(f"\n✅ Training Complete. Best model saved to {SAVE_PATH}")

if __name__ == "__main__":
    train_model()