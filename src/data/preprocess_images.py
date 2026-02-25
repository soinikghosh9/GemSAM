import os
import pydicom
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import cv2

def preprocess_and_save(src_dir, dest_dir, target_size=448):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        
    files = [f for f in os.listdir(src_dir) if f.endswith('.dicom') or f.endswith('.jpg') or f.endswith('.png')]
    print(f"Found {len(files)} images in {src_dir}")
    print(f"Target size: {target_size}x{target_size}")
    
    skipped = 0
    processed = 0
    
    for filename in tqdm(files, desc="Preprocessing"):
        base_name = os.path.splitext(filename)[0]
        dest_path = os.path.join(dest_dir, f"{base_name}.jpg")
        
        # Skip if exists
        if os.path.exists(dest_path):
            skipped += 1
            continue
            
        src_path = os.path.join(src_dir, filename)
        
        try:
            img_array = None
            if filename.endswith('.dicom'):
                ds = pydicom.dcmread(src_path)
                img_array = ds.pixel_array
                
                # Apply Windowing if available (Basic)
                # Ideally check for WindowCenter/WindowWidth tags
                if 'WindowCenter' in ds and 'WindowWidth' in ds:
                     # This is complex for vectors, simplified:
                     pass
                
                # Normalize to 0-255
                img_array = img_array.astype(float)
                img_array = (np.maximum(img_array, 0) / img_array.max()) * 255.0
                img_array = np.uint8(img_array)
                
                # If monochrome1 (inverted), invert it?
                # VinDr is usually Monochrome2 (0=black)
                if hasattr(ds, "PhotometricInterpretation") and ds.PhotometricInterpretation == "MONOCHROME1":
                    img_array = 255 - img_array
                    
            else:
                # JPG/PNG
                img_array = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
                
            if img_array is None:
                print(f"Failed to read {filename}")
                continue

            # Resize
            # Use OpenCV for speed
            resized = cv2.resize(img_array, (target_size, target_size), interpolation=cv2.INTER_AREA)
            
            # Convert to RGB (MedGemma expects 3D usually, or processor handles it)
            # Saving as grayscale JPG is fine, PIL will convert to RGB on load
            # BUT let's save as 3-channel JPG to be safe and fast for loader
            resized_rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            
            cv2.imwrite(dest_path, resized_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            processed += 1
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    print(f"Finished. Processed: {processed}, Skipped: {skipped}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="medical_data/vinbigdata-chest-xray/train")
    parser.add_argument("--dest", type=str, default="medical_data/vinbigdata-chest-xray/train_448")
    parser.add_argument("--size", type=int, default=448)
    args = parser.parse_args()
    
    preprocess_and_save(args.src, args.dest, args.size)
