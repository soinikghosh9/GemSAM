
import os
import pandas as pd
import numpy as np
import argparse

def split_dataset(data_dir, output_dir=None, seed=42):
    """
    Splits the VinBigData train.csv into train/val/test splits based on unique image_ids.
    Since patient_id is not available in the CSV, we assume image-level independence 
    (or that image_id is sufficient for this stage).
    """
    if output_dir is None:
        output_dir = data_dir

    csv_path = os.path.join(data_dir, "train.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 1. Get Unique Image IDs
    unique_ids = df['image_id'].unique()
    n_total = len(unique_ids)
    print(f"Total Unique Images: {n_total}")
    
    # 2. Shuffle
    np.random.seed(seed)
    np.random.shuffle(unique_ids)
    
    # 3. Calculate Split Indices
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.1)
    n_test = n_total - n_train - n_val
    
    train_ids = unique_ids[:n_train]
    val_ids = unique_ids[n_train:n_train+n_val]
    test_ids = unique_ids[n_train+n_val:]
    
    print(f"Split Sizes -> Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    
    # 4. Filter DataFrame
    train_df = df[df['image_id'].isin(train_ids)]
    val_df = df[df['image_id'].isin(val_ids)]
    test_df = df[df['image_id'].isin(test_ids)]
    
    # 5. Save
    train_df.to_csv(os.path.join(output_dir, "train_split.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val_split.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_split.csv"), index=False)
    
    print(f"Saved splits to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="medical_data/vinbigdata-chest-xray")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    split_dataset(args.data_dir, args.output_dir, args.seed)
