import os
import pandas as pd
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None  # cv2 is optional

class MedicalDatasetFactory:
    """
    Factory class to load different medical datasets for specific tasks.
    Supports 7 datasets: VinDr, NIH, Kaggle, SLAKE, VQA-RAD, Brain Tumor MRI, Brain Tumor Multimodal.
    """
    def __init__(self, base_data_dir="medical_data"):
        self.base_dir = base_data_dir

    def get_loader(self, dataset_name, task, batch_size=4, split="train", transformer=None, abnormal_only=False):
        dataset_name = dataset_name.lower().replace("-", "_").replace(" ", "_")
        
        if dataset_name == "vindr":
            return self._load_vindr(task, batch_size, split, transformer, abnormal_only)
        elif dataset_name == "nih" or dataset_name == "nih_chest_xray":
            return self._load_nih(task, batch_size, split, transformer)
        elif dataset_name == "kaggle" or dataset_name == "kaggle_pneumonia":
            return self._load_kaggle_pneumonia(task, batch_size, split, transformer)
        elif dataset_name == "slake":
            return self._load_slake(task, batch_size, split, transformer)
        elif dataset_name == "sam2":
            return self._load_sam2_dataset(batch_size, split, transformer)
        elif dataset_name == "vqa_rad":
            return self._load_vqa_rad(task, batch_size, split, transformer)
        elif dataset_name == "brain_tumor_mri":
            return self._load_brain_tumor_mri(task, batch_size, split, transformer)
        elif dataset_name == "brain_tumor_multimodal":
            return self._load_brain_tumor_multimodal(task, batch_size, split, transformer)
        elif dataset_name == "busi":
            return self._load_busi(task, batch_size, split, transformer)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}. Supported: vindr, nih, kaggle, slake, vqa_rad, sam2, brain_tumor_mri, brain_tumor_multimodal, busi")
    
    def get_dataset(self, dataset_name, task="default", split="train", transformer=None):
        """Return raw Dataset object instead of DataLoader for custom batching."""
        dataset_name = dataset_name.lower().replace("-", "_").replace(" ", "_")
        
        if dataset_name == "vindr":
            data_dir = os.path.join(self.base_dir, "vinbigdata-chest-xray")
            return VinDrCXRLoader(data_dir, split=split, transformer=transformer)
        elif dataset_name == "nih" or dataset_name == "nih_chest_xray":
            data_dir = os.path.join(self.base_dir, "nih chest xray")
            return NIHChestXRayLoader(data_dir, task=task, split=split, transformer=transformer)
        elif dataset_name == "kaggle" or dataset_name == "kaggle_pneumonia":
            data_dir = os.path.join(self.base_dir, "chest_xray_kaggle")
            return KagglePneumoniaLoader(data_dir, split=split, transformer=transformer)
        elif dataset_name == "slake":
            data_dir = os.path.join(self.base_dir, "Slake1.0")
            return SLAKELoader(data_dir, split=split, transformer=transformer)
        elif dataset_name == "vqa_rad":
            data_dir = os.path.join(self.base_dir, "osfstorage-archive")
            return VQARADLoader(data_dir, split=split, transformer=transformer)
        elif dataset_name == "brain_tumor_mri":
            data_dir = os.path.join(self.base_dir, "Brain Tumor MRI Dataset")
            return BrainTumorMRILoader(data_dir, task=task, split=split, transformer=transformer)
        elif dataset_name == "brain_tumor_mri_detection":
            # Alias for brain tumor detection with pseudo bounding boxes
            data_dir = os.path.join(self.base_dir, "Brain Tumor MRI Dataset")
            return BrainTumorMRILoader(data_dir, task="detection", split=split, transformer=transformer)
        elif dataset_name == "brain_tumor_multimodal":
            data_dir = os.path.join(self.base_dir, "Brain tumor multimodal image (CT & MRI)")
            # Convert to absolute path to avoid working directory issues
            data_dir = os.path.abspath(data_dir)
            return BrainTumorMultimodalLoader(data_dir, split=split, transformer=transformer)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def _load_nih(self, task, batch_size, split, transformer):
        """NIH ChestX-ray14: 112k images for classification, 985 with bounding boxes."""
        data_dir = os.path.join(self.base_dir, "nih chest xray")
        return DataLoader(
            NIHChestXRayLoader(data_dir, task=task, split=split, transformer=transformer),
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=self._collate_fn,
            num_workers=4
        )
    
    def _load_kaggle_pneumonia(self, task, batch_size, split, transformer):
        """Kaggle Pneumonia: Binary screening (NORMAL vs PNEUMONIA)."""
        data_dir = os.path.join(self.base_dir, "chest_xray_kaggle")
        return DataLoader(
            KagglePneumoniaLoader(data_dir, split=split, transformer=transformer),
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=self._collate_fn,
            num_workers=4
        )
    
    def _load_brain_tumor_mri(self, task, batch_size, split, transformer):
        """Brain Tumor MRI: 4-class classification + screening."""
        data_dir = os.path.join(self.base_dir, "Brain Tumor MRI Dataset")
        return DataLoader(
            BrainTumorMRILoader(data_dir, task=task, split=split, transformer=transformer),
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=self._collate_fn,
            num_workers=4
        )
    
    def _load_brain_tumor_multimodal(self, task, batch_size, split, transformer):
        """Brain Tumor Multimodal: CT and MRI images with modality labels."""
        data_dir = os.path.join(self.base_dir, "Brain tumor multimodal image (CT & MRI)")
        return DataLoader(
            BrainTumorMultimodalLoader(data_dir, split=split, transformer=transformer),
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=self._collate_fn,
            num_workers=4
        )

    def _load_busi(self, task, batch_size, split, transformer):
        """BUSI: Breast Ultrasound Images with Ground Truth for blind testing."""
        data_dir = os.path.join(self.base_dir, "Dataset_BUSI_with_GT")
        return DataLoader(
            BUSIDataset(data_dir, split=split, task=task, transformer=transformer),
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=self._collate_fn,
            num_workers=0  # Avoid issues on Windows
        )

    def _load_vindr(self, task, batch_size, split, transformer, abnormal_only=False):
        """
        VinDr-CXR for Detection tasks.
        """
        data_dir = os.path.join(self.base_dir, "vinbigdata-chest-xray")
        return DataLoader(
            VinDrCXRLoader(data_dir, split=split, transformer=transformer, abnormal_only=abnormal_only),
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=self._collate_fn,
            num_workers=4
        )

    def _load_slake(self, task, batch_size, split, transformer):
        """
        SLAKE for VQA/Reasoning tasks.
        """
        data_dir = os.path.join(self.base_dir, "Slake1.0")
        return DataLoader(
            SLAKELoader(data_dir, split=split, transformer=transformer),
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=self._collate_fn
        )

    def _load_sam2_dataset(self, batch_size, split, transformer):
        """
        SAM 2 Training Dataset (Image + Mask).
        Uses SLAKE or BraTS masks.
        """
        # For now, default to SLAKE masks as they are accessible
        data_dir = os.path.join(self.base_dir, "Slake1.0")
        return DataLoader(
            SAM2Dataset(data_dir, split=split, transformer=transformer),
            batch_size=batch_size,
            shuffle=(split == "train")
        )

    def _load_vqa_rad(self, task, batch_size, split, transformer):
        """
        VQA-RAD (OSFStorage) for VQA/Reasoning tasks.
        """
        data_dir = os.path.join(self.base_dir, "osfstorage-archive")
        return DataLoader(
            VQARADLoader(data_dir, split=split, transformer=transformer),
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=self._collate_fn
        )

    def _collate_fn(self, batch):
        # Return dict of lists instead of tuple for clearer access
        # batch is list of dicts: [ {"image":..., "boxes":...}, ... ]
        if not batch: return {}
        keys = batch[0].keys()
        return {key: [d[key] for d in batch] for key in keys}

class VinDrCXRLoader(Dataset):
    def __init__(self, root_dir, split="train", transformer=None, abnormal_only=False):
        self.root_dir = root_dir
        self.split = split
        self.transformer = transformer
        self.abnormal_only = abnormal_only
        # Use strict experiment splits
        self.annotations_file = os.path.join(root_dir, f"{split}_split.csv")
        
        # Check for preprocessed accessible directory
        self.preprocessed_dir = os.path.join(root_dir, f"{split}_448")
        self.use_preprocessed = os.path.exists(self.preprocessed_dir) and len(os.listdir(self.preprocessed_dir)) > 0
        if self.use_preprocessed:
            print(f"[{split.upper()}] Detected preprocessed data in {self.preprocessed_dir}. Using optimized loading.")
        else:
             # Fallback to train_448 if test_448 missing? No, that's data leakage/wrong files.
             print(f"[{split.upper()}] Preprocessed directory {self.preprocessed_dir} not found. Will load original images.")

        # Fallback to train.csv only if split file missing (Backwards compatibility)
        if not os.path.exists(self.annotations_file):
             if split == "train":
                  self.annotations_file = os.path.join(root_dir, "train.csv")
             else:
                  # For val/test, if split missing, maybe use train.csv but filter? 
                  # For now, just warn or default to empty
                  pass

        self.data_map = []
        if os.path.exists(self.annotations_file):
            print(f"[{split.upper()}] Loading annotations from {self.annotations_file}")
            df = pd.read_csv(self.annotations_file)
            self.data_map = df.groupby('image_id')
        
        self.image_ids = list(self.data_map.groups.keys())
        
        # Apply abnormal_only filter
        if self.abnormal_only:
            original_count = len(self.image_ids)
            filtered_ids = []
            for img_id in self.image_ids:
                group = self.data_map.get_group(img_id)
                if any(row['class_name'] != 'No finding' for _, row in group.iterrows()):
                    filtered_ids.append(img_id)
            self.image_ids = filtered_ids
            print(f"[{self.split.upper()}] Filtered for abnormal-only samples: {len(self.image_ids)}/{original_count}")
        # Load metadata if available
        self.meta_map = {}
        self.meta_file = os.path.join(root_dir, "train_meta.csv")
        if os.path.exists(self.meta_file):
            print(f"[{split.upper()}] Loading metadata from {self.meta_file}")
            meta_df = pd.read_csv(self.meta_file)
            # Create dict: image_id -> (width, height)
            for _, row in meta_df.iterrows():
                self.meta_map[row['image_id']] = (row['width'], row['height'])
        else:
            print(f"[{split.upper()}] Warning: Metadata file {self.meta_file} not found. Normalization may be incorrect.")

        print(f"[{split.upper()}] Loaded {len(self.image_ids)} images.")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        image = None
        # OPTIMIZED PATH: Load from train_448
        if self.use_preprocessed:
             img_path = os.path.join(self.preprocessed_dir, f"{image_id}.jpg")
             if os.path.exists(img_path):
                 try:
                     # Loading pre-resized JPG is extremely fast
                     image = Image.open(img_path).convert("RGB")
                 except Exception as e:
                     print(f"Warning: Failed to load preprocessed {img_path}: {e}")

        # FALLBACK PATH: Load original (DICOM/JPG)
        if image is None:
            # Image loading logic (supports dicom or jpg)
            img_path = os.path.join(self.root_dir, "train", f"{image_id}.jpg")
            if not os.path.exists(img_path):
                 img_path = os.path.join(self.root_dir, "train", f"{image_id}.dicom")
            
            try:
                if img_path.endswith('.dicom'):
                    import pydicom
                    ds = pydicom.dcmread(img_path)
                    pixel_array = ds.pixel_array
                    # Simple normalization
                    pixel_array = (np.maximum(pixel_array, 0) / pixel_array.max()) * 255.0
                    image = Image.fromarray(np.uint8(pixel_array)).convert("RGB")
                else:
                    image = Image.open(img_path).convert("RGB")
            except:
                # Fallback or error
                print(f"Error loading {image_id}")
                image = Image.new('RGB', (448, 448))
            
        boxes = []
        labels = []

        # Get original dimensions for normalization
        orig_w, orig_h = self.meta_map.get(image_id, (None, None))
        
        # If metadata missing, MUST infer from image even if preprocessed or not
        if orig_w is None:
             # Look for original image to get TRUE dimensions
             # Even if using preprocessed, we need original sizes for coordinate mapping
             orig_img_path = os.path.join(self.root_dir, "train", f"{image_id}.jpg")
             if not os.path.exists(orig_img_path):
                  orig_img_path = os.path.join(self.root_dir, "train", f"{image_id}.dicom")
             
             if os.path.exists(orig_img_path):
                 try:
                     if orig_img_path.endswith('.dicom'):
                         import pydicom
                         ds = pydicom.dcmread(orig_img_path)
                         orig_w, orig_h = ds.Columns, ds.Rows
                     else:
                         with Image.open(orig_img_path) as tmp_img:
                             orig_w, orig_h = tmp_img.size
                 except:
                     pass
             
             # Final fallback: if cannot find original, use current image size
             if orig_w is None:
                 orig_w, orig_h = image.size

        # CRITICAL FIX: Load boxes for ALL splits, not just train
        if image_id in self.data_map.groups:
            group = self.data_map.get_group(image_id)
            for _, row in group.iterrows():
                if row['class_name'] != 'No finding':
                    # Format: x_min, y_min, x_max, y_max
                    x1, y1, x2, y2 = row['x_min'], row['y_min'], row['x_max'], row['y_max']
                    
                    # Normalize to 0-1000 if dimensions available
                    if orig_w and orig_h:
                        x1 = int((x1 / orig_w) * 1000)
                        y1 = int((y1 / orig_h) * 1000)
                        x2 = int((x2 / orig_w) * 1000)
                        y2 = int((y2 / orig_h) * 1000)
                        
                        # Clip to 0-1000
                        x1 = max(0, min(1000, x1))
                        y1 = max(0, min(1000, y1))
                        x2 = max(0, min(1000, x2))
                        y2 = max(0, min(1000, y2))
                        
                    boxes.append([x1, y1, x2, y2])
                    labels.append(row['class_name'])
        
        if self.transformer:
            image = self.transformer(image)

        return {
            "image": image,
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "modality": "xray"  # VinDr is chest X-ray
        }

class SLAKELoader(Dataset):
    def __init__(self, root_dir, split="train", transformer=None):
        self.root_dir = root_dir
        self.split = split
        self.transformer = transformer
        self.json_path = os.path.join(root_dir, "train.json" if split == "train" else "test.json")
        
        # Check for preprocessed
        self.preprocessed_dir = os.path.join(root_dir, "imgs_448")
        self.use_preprocessed = os.path.exists(self.preprocessed_dir) and len(os.listdir(self.preprocessed_dir)) > 0
        if self.use_preprocessed:
            print(f"[{split.upper()}] SLAKE: Using preprocessed data from {self.preprocessed_dir}")

        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_name = item['img_name']
        question = item['question']
        answer = item['answer']
        
        image = None
        # OPTIMIZED PATH
        if self.use_preprocessed:
             # Try to find jpg version (SLAKE might be jpg/png)
             # The preprocessor saves as .jpg
             base_name = os.path.splitext(img_name)[0]
             img_path = os.path.join(self.preprocessed_dir, f"{base_name}.jpg")
             if os.path.exists(img_path):
                 try:
                     image = Image.open(img_path).convert("RGB")
                 except: pass
        
        # FALLBACK
        if image is None:
            img_path = os.path.join(self.root_dir, "imgs", img_name)
            image = Image.open(img_path).convert("RGB")
            # If original is large, we might want to resize here too? 
            # But relying on cache is better.
        
        if self.transformer:
            image = self.transformer(image)

        return {
             "image": image,
             "question": question,
             "answer": answer,
             "q_id": item['qid'],
             "modality": item.get('modality', 'unknown').lower(),  # Added for modality task
             "location": item.get('location', 'unknown')  # Anatomical location
        }

class SAM2Dataset(Dataset):
    """
    Loads (Image, Mask) pairs for SAM 2 Adapter training.
    Currently maps SLAKE masks.
    """
    def __init__(self, root_dir, split="train", transformer=None, image_size=1024):
        self.root_dir = root_dir
        self.split = split
        self.transformer = transformer
        self.image_size = image_size  # SAM2 requires 1024 for correct feature map sizes
        # SLAKE has 'mask' folder inside 'imgs' usually or defined in metadata
        # We walk through the imgs directory to find mask.png files
        self.samples = []

        imgs_dir = os.path.join(root_dir, "imgs")
        if os.path.exists(imgs_dir):
            for case_folder in os.listdir(imgs_dir):
                case_path = os.path.join(imgs_dir, case_folder)
                if os.path.isdir(case_path):
                    if "mask.png" in os.listdir(case_path) and "source.jpg" in os.listdir(case_path):
                        self.samples.append({
                            "image": os.path.join(case_path, "source.jpg"),
                            "mask": os.path.join(case_path, "mask.png")
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image']).convert("RGB")
        mask = Image.open(sample['mask']).convert("L") # Grayscale mask

        # Resize to configured size (default 512 for 16GB GPU)
        image = image.resize((self.image_size, self.image_size))
        mask = mask.resize((self.image_size, self.image_size), resample=Image.NEAREST)

        mask_array = np.array(mask)
        mask_array = (mask_array > 0).astype(np.float32) # Binary mask

        if self.transformer:
             image = self.transformer(image)

        return {
            "image": image,
            "mask": torch.from_numpy(mask_array).unsqueeze(0)  # [1, H, W]
        }

class VQARADLoader(Dataset):
    def __init__(self, root_dir, split="train", transformer=None):
        self.root_dir = root_dir
        self.split = split
        self.transformer = transformer
        
        # Load JSON
        self.json_path = os.path.join(root_dir, "VQA_RAD Dataset Public.json")
        self.img_dir = os.path.join(root_dir, "VQA_RAD Image Folder")
        
        with open(self.json_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
            
        # VQA-RAD doesn't have an explicit split, so we'll simulate one
        # 80/20 split based on index
        cutoff = int(len(all_data) * 0.8)
        if split == "train":
            self.data = all_data[:cutoff]
        else:
            self.data = all_data[cutoff:]
            
        print(f"[{split.upper()}] VQA-RAD: Loaded {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_name = item['image_name']
        question = item['question']
        answer = str(item['answer'])
        
        # Load Image
        img_path = os.path.join(self.img_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback
            image = Image.new('RGB', (448, 448))
            
        if self.transformer:
            image = self.transformer(image)
            
        # Return standardized keys for MedGemmaDataset
        return {
             "image": image,
             "question": question,
             "answer": answer,
             "q_id": item['qid']
        }


class NIHChestXRayLoader(Dataset):
    """
    NIH ChestX-ray14 Dataset.
    - 112,120 images with 15 class labels (including "No Finding")
    - 985 images with bounding box annotations in BBox_List_2017.csv
    
    task: "classification" (labels only) or "detection" (with bboxes)
    """
    # Class mapping for NIH pathologies
    NIH_CLASSES = [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
        "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
        "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia", "No Finding"
    ]
    
    def __init__(self, root_dir, task="classification", split="train", transformer=None):
        self.root_dir = root_dir
        self.task = task
        self.split = split
        self.transformer = transformer
        
        # Load main data entry file
        self.data_entry_file = os.path.join(root_dir, "Data_Entry_2017.csv")
        self.bbox_file = os.path.join(root_dir, "BBox_List_2017.csv")
        
        # Load annotations
        self.df = pd.read_csv(self.data_entry_file)
        
        # Load bounding boxes if task is detection
        self.bbox_map = {}
        if task == "detection" and os.path.exists(self.bbox_file):
            bbox_df = pd.read_csv(self.bbox_file)
            # Group by image ID
            for _, row in bbox_df.iterrows():
                img_id = row["Image Index"]
                if img_id not in self.bbox_map:
                    self.bbox_map[img_id] = []
                # BBox format: x, y, w, h -> convert to x_min, y_min, x_max, y_max
                x, y, w, h = row.iloc[2], row.iloc[3], row.iloc[4], row.iloc[5]
                self.bbox_map[img_id].append({
                    "label": row["Finding Label"],
                    "box": [float(x), float(y), float(x + w), float(y + h)]
                })
            print(f"[NIH] Loaded {len(self.bbox_map)} images with bounding boxes.")
        
        # Filter data based on task
        if task == "detection":
            # Only use images with bounding boxes
            self.df = self.df[self.df["Image Index"].isin(self.bbox_map.keys())]
        
        # Split: 80/20
        cutoff = int(len(self.df) * 0.8)
        if split == "train":
            self.df = self.df.iloc[:cutoff]
        else:
            self.df = self.df.iloc[cutoff:]
        
        self.image_ids = self.df["Image Index"].tolist()
        print(f"[{split.upper()}] NIH ChestX-ray ({task}): Loaded {len(self.image_ids)} samples.")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["Image Index"]
        labels_str = row["Finding Labels"]
        
        # Load image - look in images folder
        img_path = os.path.join(self.root_dir, "images", image_id)
        if not os.path.exists(img_path):
            # Try images_001 through images_012 folders
            for folder_num in range(1, 13):
                img_path = os.path.join(self.root_dir, f"images_{folder_num:03d}", image_id)
                if os.path.exists(img_path):
                    break
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            image = Image.new('RGB', (448, 448))
        
        # Parse labels
        labels = labels_str.split("|") if "|" in labels_str else [labels_str]
        
        # Get bounding boxes if detection task
        boxes = []
        box_labels = []
        if self.task == "detection" and image_id in self.bbox_map:
            for bbox_info in self.bbox_map[image_id]:
                boxes.append(bbox_info["box"])
                box_labels.append(bbox_info["label"])
        
        if self.transformer:
            image = self.transformer(image)
        
        # Return format depends on task
        if self.task == "detection":
            return {
                "image": image,
                "boxes": boxes,
                "labels": box_labels,
                "image_id": image_id,
                "modality": "xray"  # NIH is chest X-ray
            }
        else:
            # Classification - return binary label (healthy vs diseased) or multi-label
            is_healthy = "No Finding" in labels
            return {
                "image": image,
                "labels": labels,
                "is_healthy": is_healthy,
                "image_id": image_id,
                "modality": "xray"  # NIH is chest X-ray
            }


class KagglePneumoniaLoader(Dataset):
    """
    Kaggle Chest X-ray Pneumonia Dataset.
    Binary classification: NORMAL vs PNEUMONIA.
    Perfect for screening task.
    """
    def __init__(self, root_dir, split="train", transformer=None):
        self.root_dir = root_dir
        self.split = split
        self.transformer = transformer
        
        # Map split names: train, val, test
        split_folder = split if split in ["train", "val", "test"] else "train"
        self.data_dir = os.path.join(root_dir, split_folder)
        
        # Fallback: check if structure is chest_xray/train or directly train
        if not os.path.exists(self.data_dir):
            self.data_dir = os.path.join(root_dir, "chest_xray", split_folder)
        
        self.samples = []
        
        # Load images from NORMAL and PNEUMONIA folders
        for label_name in ["NORMAL", "PNEUMONIA"]:
            label_dir = os.path.join(self.data_dir, label_name)
            if os.path.exists(label_dir):
                for img_file in os.listdir(label_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append({
                            "path": os.path.join(label_dir, img_file),
                            "label": label_name,
                            "is_healthy": label_name == "NORMAL"
                        })
        
        print(f"[{split.upper()}] Kaggle Pneumonia: Loaded {len(self.samples)} samples "
              f"({sum(1 for s in self.samples if s['is_healthy'])} Normal, "
              f"{sum(1 for s in self.samples if not s['is_healthy'])} Pneumonia)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            image = Image.open(sample["path"]).convert("RGB")
        except Exception as e:
            image = Image.new('RGB', (448, 448))
        
        if self.transformer:
            image = self.transformer(image)
        
        return {
            "image": image,
            "label": sample["label"],
            "is_healthy": sample["is_healthy"],
            "modality": "xray"
        }


class BrainTumorMRILoader(Dataset):
    """
    Brain Tumor MRI Dataset.
    4-class classification: glioma, meningioma, pituitary, notumor.
    Also supports screening task: tumor vs no_tumor.
    """
    CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
    
    def __init__(self, root_dir, task="classification", split="train", transformer=None):
        self.root_dir = root_dir
        self.task = task  # "classification" (4-class) or "screening" (binary)
        self.split = split
        self.transformer = transformer
        
        # Map splits: Training or Testing
        split_folder = "Training" if split == "train" else "Testing"
        self.data_dir = os.path.join(root_dir, split_folder)
        
        self.samples = []
        
        # Load images from each class folder
        if os.path.exists(self.data_dir):
            for class_name in self.CLASSES:
                class_dir = os.path.join(self.data_dir, class_name)
                if os.path.exists(class_dir):
                    for img_file in os.listdir(class_dir):
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.samples.append({
                                "path": os.path.join(class_dir, img_file),
                                "class": class_name,
                                "is_healthy": class_name == "notumor"
                            })
        
        print(f"[{split.upper()}] Brain Tumor MRI ({task}): Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        try:
            image = Image.open(sample["path"]).convert("RGB")
        except Exception as e:
            image = Image.new('RGB', (448, 448))

        if self.transformer:
            image = self.transformer(image)

        if self.task == "screening":
            return {
                "image": image,
                "is_healthy": sample["is_healthy"],
                "label": "NORMAL" if sample["is_healthy"] else "TUMOR",
                "modality": "mri"
            }
        elif self.task == "detection":
            # Detection task: Generate pseudo bounding boxes for brain tumors
            # For tumor images, generate a central bounding box (tumors typically in center)
            # For no-tumor images, return empty boxes
            if sample["is_healthy"]:
                # No tumor - empty findings
                return {
                    "image": image,
                    "boxes": [],
                    "labels": [],
                    "image_id": os.path.basename(sample["path"]),
                    "modality": "mri"
                }
            else:
                # Has tumor - generate pseudo bounding box
                # Brain tumors typically occupy 20-60% of the image area
                # Use central region as approximation
                tumor_class = sample["class"].title()  # e.g., "Glioma", "Meningioma"

                # Pseudo bounding box: central 40-60% of image (normalized 0-1000)
                # Randomize slightly for training variety
                import random
                margin = random.randint(200, 350)  # 20-35% margin from edges
                box = [margin, margin, 1000 - margin, 1000 - margin]

                return {
                    "image": image,
                    "boxes": [box],
                    "labels": [tumor_class],
                    "image_id": os.path.basename(sample["path"]),
                    "modality": "mri"
                }
        else:
            # Classification task
            return {
                "image": image,
                "class": sample["class"],
                "is_healthy": sample["is_healthy"],
                "modality": "mri"
            }


class BrainTumorMultimodalLoader(Dataset):
    """
    Brain Tumor Multimodal Dataset (CT & MRI).
    Used for modality detection training: CT vs MRI.
    """
    def __init__(self, root_dir, split="train", transformer=None):
        self.root_dir = root_dir
        self.split = split
        self.transformer = transformer

        self.samples = []

        # FIXED: Map of possible directory names to modality labels
        # Supports various naming conventions found in different dataset versions
        modality_dirs = {
            "ct": ["Brain Tumor CT scan Images", "CT", "ct", "CT_images", "ct_images"],
            "mri": ["Brain Tumor MRI images", "MRI", "mri", "MRI_images", "mri_images"]
        }

        # Check if there's a "Dataset" subdirectory (common pattern)
        dataset_subdir = os.path.join(root_dir, "Dataset")
        search_root = dataset_subdir if os.path.exists(dataset_subdir) else root_dir

        # Debug: Print paths being checked
        print(f"[DEBUG] BrainTumorMultimodal: root_dir = {root_dir}")
        print(f"[DEBUG] BrainTumorMultimodal: search_root = {search_root} (exists: {os.path.exists(search_root)})")

        # List contents of search_root if it exists
        if os.path.exists(search_root):
            try:
                contents = os.listdir(search_root)
                print(f"[DEBUG] BrainTumorMultimodal: search_root contents = {contents}")
            except Exception as e:
                print(f"[DEBUG] BrainTumorMultimodal: Error listing search_root: {e}")

        # Walk through the directory looking for modality folders
        for modality, possible_names in modality_dirs.items():
            for dir_name in possible_names:
                modality_dir = os.path.join(search_root, dir_name)
                exists = os.path.exists(modality_dir)
                if exists:
                    print(f"[DEBUG] Found modality dir: {modality_dir}")
                    for root, dirs, files in os.walk(modality_dir):
                        for f in files:
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                                self.samples.append({
                                    "path": os.path.join(root, f),
                                    "modality": modality
                                })
                    break  # Found this modality, no need to check other names
        
        # Split: 80/20
        cutoff = int(len(self.samples) * 0.8)
        if split == "train":
            self.samples = self.samples[:cutoff]
        else:
            self.samples = self.samples[cutoff:]
        
        modality_counts = {}
        for s in self.samples:
            modality_counts[s["modality"]] = modality_counts.get(s["modality"], 0) + 1
        print(f"[{split.upper()}] Brain Tumor Multimodal: Loaded {len(self.samples)} samples {modality_counts}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            image = Image.open(sample["path"]).convert("RGB")
        except Exception as e:
            image = Image.new('RGB', (448, 448))
        
        if self.transformer:
            image = self.transformer(image)
        
        return {
            "image": image,
            "modality": sample["modality"]
        }


class BUSIDataset(Dataset):
    """
    BUSI: Breast Ultrasound Images with Ground Truth.
    For BLIND TESTING of classification and segmentation.

    Categories: benign, malignant, normal
    Each image has corresponding mask(s) for segmentation.

    Paper: Al-Dhabyani et al. "Dataset of breast ultrasound images" Data in Brief, 2020.
    """
    def __init__(self, root_dir, split="test", task="classification", transformer=None, image_size=1024):
        self.root_dir = root_dir
        self.split = split
        self.task = task  # "classification", "segmentation", or "both"
        self.transformer = transformer
        self.image_size = image_size
        self.samples = []

        # Class mapping
        self.class_names = ["normal", "benign", "malignant"]
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        # Load samples from each category
        for class_name in self.class_names:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue

            # Group images with their masks
            files = os.listdir(class_dir)
            image_files = [f for f in files if not '_mask' in f and f.endswith('.png')]

            for img_file in image_files:
                img_path = os.path.join(class_dir, img_file)

                # Find corresponding mask(s)
                base_name = img_file.replace('.png', '')
                mask_files = [f for f in files if f.startswith(base_name + '_mask')]
                mask_paths = [os.path.join(class_dir, m) for m in mask_files]

                self.samples.append({
                    "image_path": img_path,
                    "mask_paths": mask_paths,  # May have multiple masks
                    "class_name": class_name,
                    "class_idx": self.class_to_idx[class_name]
                })

        # Shuffle and split (80/20 for train/test, but we use test for blind evaluation)
        np.random.seed(42)  # Reproducible
        indices = np.random.permutation(len(self.samples))
        cutoff = int(len(self.samples) * 0.8)

        if split == "train":
            self.samples = [self.samples[i] for i in indices[:cutoff]]
        else:  # test/val for blind testing
            self.samples = [self.samples[i] for i in indices[cutoff:]]

        # Count per class
        class_counts = {}
        for s in self.samples:
            c = s["class_name"]
            class_counts[c] = class_counts.get(c, 0) + 1

        print(f"[BUSI {split.upper()}] Loaded {len(self.samples)} samples: {class_counts}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample["image_path"]).convert("RGB")
        orig_size = image.size  # (W, H)

        # Load and combine masks (some images have multiple mask annotations)
        combined_mask = None
        if sample["mask_paths"]:
            for mask_path in sample["mask_paths"]:
                mask = Image.open(mask_path).convert("L")
                mask_array = np.array(mask)
                if combined_mask is None:
                    combined_mask = mask_array
                else:
                    combined_mask = np.maximum(combined_mask, mask_array)
            combined_mask = (combined_mask > 0).astype(np.float32)
        else:
            # Normal images may not have masks
            combined_mask = np.zeros((image.size[1], image.size[0]), dtype=np.float32)

        # Resize for segmentation task
        if self.task in ["segmentation", "both"]:
            image_resized = image.resize((self.image_size, self.image_size))
            mask_pil = Image.fromarray((combined_mask * 255).astype(np.uint8))
            mask_resized = mask_pil.resize((self.image_size, self.image_size), resample=Image.NEAREST)
            combined_mask = np.array(mask_resized) / 255.0

        # Apply transformer if provided
        if self.transformer:
            image = self.transformer(image_resized if self.task in ["segmentation", "both"] else image)

        result = {
            "image": image if self.transformer else (image_resized if self.task in ["segmentation", "both"] else image),
            "class_name": sample["class_name"],
            "class_idx": sample["class_idx"],
            "mask": torch.from_numpy(combined_mask).unsqueeze(0) if combined_mask is not None else None,
            "image_path": sample["image_path"],
            "orig_size": orig_size
        }

        return result
