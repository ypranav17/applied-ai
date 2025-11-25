import pandas as pd

# --- CONFIGURATION ---
INPUT_CSV = "full_dataset_2.csv"   # Your uploaded file
OUTPUT_CSV = "clean_dataset.csv"   # The new file we will train on
MAX_QTY = 20                       # Cap at 20 to remove outliers
# ---------------------

def filter_data():
    print(f"Reading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    print(f"Original size: {len(df)}")
    print(f"Max quantity found: {df['quantity'].max()}")
    
    # Filter: Keep only rows where quantity is between 1 and MAX_QTY
    df_clean = df[(df['quantity'] >= 1) & (df['quantity'] <= MAX_QTY)]
    
    removed_count = len(df) - len(df_clean)
    print(f"Filtered size: {len(df_clean)}")
    print(f"Removed {removed_count} outliers (quantity > {MAX_QTY})")
    
    # Save
    df_clean.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved clean dataset to {OUTPUT_CSV}")

if __name__ == "__main__":
    filter_data()