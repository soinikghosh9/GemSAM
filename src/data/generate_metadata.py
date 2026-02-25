import os
import pydicom
import pandas as pd
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor

def process_file(args):
    filename, src_dir = args
    if not filename.lower().endswith('.dicom'):
        return None
    
    path = os.path.join(src_dir, filename)
    try:
        # Read header only (fast)
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        img_id = os.path.splitext(filename)[0]
        return {
            "image_id": img_id,
            "height": ds.Rows,
            "width": ds.Columns
        }
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

def generate_metadata(src_dir, output_csv):
    files = os.listdir(src_dir)
    print(f"Scanning {len(files)} files in {src_dir}...")
    
    # Use ThreadPool for IO-bound task
    with ThreadPoolExecutor(max_workers=8) as executor:
        args = [(f, src_dir) for f in files]
        results = list(tqdm(executor.map(process_file, args), total=len(files)))
    
    # Filter None results
    data = [r for r in results if r is not None]
    
    df = pd.DataFrame(data)
    print(f"Extracted metadata for {len(df)} images.")
    
    df.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="medical_data/vinbigdata-chest-xray/train")
    parser.add_argument("--out", type=str, default="medical_data/vinbigdata-chest-xray/train_meta.csv")
    args = parser.parse_args()
    
    generate_metadata(args.src, args.out)
