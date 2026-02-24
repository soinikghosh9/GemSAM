import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from hydra.utils import instantiate
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from src.data.factory import MedicalDatasetFactory
import argparse
import numpy as np
from tqdm import tqdm

class SAM2Trainer:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Load SAM 2 manually (Bypassing build_sam2 to avoid Hydra config path issues)
        print(f"Loading SAM 2 Model...")
        config_path = "sam2_hiera_t.yaml"
        checkpoint_path = os.path.join("checkpoints", "sam2_hiera_tiny.pt")
        
        if not os.path.exists(config_path):
             raise FileNotFoundError(f"Config not found at {config_path}")
        if not os.path.exists(checkpoint_path):
             # Try falling back to root if not in checkpoints/
             if os.path.exists("sam2_hiera_tiny.pt"):
                 checkpoint_path = "sam2_hiera_tiny.pt"
             else:
                 raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
             
        # Load Config
        cfg = OmegaConf.load(config_path)
        
        # Instantiate Model
        try:
            self.model = instantiate(cfg.model, _recursive_=True)
        except Exception as e:
            print(f"(!) Failed to instantiate SAM 2 model: {e}")
            raise e
            
        # Load Weights
        print(f"Loading weights from {checkpoint_path}")
        sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if "model" in sd:
            sd = sd["model"]
        
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing:
            print(f"DEBUG: Missing keys (expected for Hiera Tiny if safe): {len(missing)}")
        if unexpected:
            print(f"DEBUG: Unexpected keys: {len(unexpected)}")
            
        self.model.to(self.device)
        
        # 2. Apply LoRA - Target Hiera blocks
        # Hiera uses 'q_proj', 'v_proj' in attention blocks?
        # Let's print modules to be sure if debugging, but standard Transformer naming usually holds.
        # If Hiera uses different names (e.g. 'q', 'v'), we need to adjust.
        # For now, we target standard names.
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["qkv", "proj"], # Hiera uses 'qkv' and 'proj'
            lora_dropout=0.05,
            bias="none"
        )
        try:
            self.model.image_encoder = get_peft_model(self.model.image_encoder, peft_config)
            print("LoRA Adapter attached to Image Encoder.")
            self.model.image_encoder.print_trainable_parameters()
        except Exception as e:
            print(f"(!) Failed to apply LoRA: {e}")
            print("Proceeding with full fine-tuning (fallback) or check target_modules.")
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)

    def compute_loss(self, pred_masks, gt_masks):
        # pred_masks: [B, 1, 256, 256] logs (low res)
        # gt_masks: [B, 1, 1024, 1024]
        
        # Resize GT to 256x256 to match logits for efficiency
        gt_masks_low = F.interpolate(gt_masks, size=(256, 256), mode='nearest')
        
        # Sigmoid on logits
        preds = torch.sigmoid(pred_masks)
        
        # Dice Loss
        intersection = (preds * gt_masks_low).sum()
        union = preds.sum() + gt_masks_low.sum()
        dice_loss = 1 - (2. * intersection + 1e-6) / (union + 1e-6)
        
        # BCE / Focal (Simple BCE here)
        bce_loss = F.binary_cross_entropy_with_logits(pred_masks, gt_masks_low)
        
        return dice_loss + bce_loss

    def train_step(self, batch):
        images = batch['image'].to(self.device)
        masks = batch['mask'].to(self.device) # [B, 1, 1024, 1024]
        B = images.shape[0]
        
        self.optimizer.zero_grad()
        
        # 1. Image Encoder
        with torch.amp.autocast('cuda'):
             backbone_out = self.model.forward_image(images)
             _, vision_feats, _, feat_sizes = self.model._prepare_backbone_features(backbone_out)

             # Prepare High Res Features if needed
             high_res_features = None
             if self.model.use_high_res_features_in_sam:
                  if len(vision_feats) > 1:
                     high_res_features = [
                         x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                         for x, s in zip(vision_feats[:-1], feat_sizes[:-1])
                     ]

             # Prepare Low Res Features (Image Embeddings) [B, C, H, W]
             image_embeddings = vision_feats[-1].permute(1, 2, 0).view(B, -1, *feat_sizes[-1])
            
             # 2. Prompt Generation (Simulate Box Prompt)
             # Extract box from mask
             prompts = []
             for i in range(B):
                 mask = masks[i, 0]
                 rows, cols = torch.where(mask > 0.5)
                 if len(rows) > 0:
                     y1, x1 = rows.min(), cols.min()
                     y2, x2 = rows.max(), cols.max()
                     # Add noise
                     noise = torch.randint(-5, 5, (4,), device=self.device)
                     box = torch.tensor([x1, y1, x2, y2], device=self.device).float() + noise
                     prompts.append(box)
                 else:
                      # No mask, use full image or dummy
                      prompts.append(torch.tensor([0, 0, 100, 100], device=self.device).float())
            
             boxes = torch.stack(prompts).unsqueeze(1) # [B, 1, 4]
            
             # 3. Prompt Encoder & Mask Decoder
            
             point_coords = None
             point_labels = None
            
             sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
                 points=None,
                 boxes=boxes,
                 masks=None,
             )
            
             # SAM 2 inputs features from 'backbone_out['vision_features']' + 'backbone_out['backbone_fpn']'
             # The decoder call is complex. Assumed simplified signature:
             low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder(
                 image_embeddings=image_embeddings, 
                 image_pe=self.model.sam_prompt_encoder.get_dense_pe(), 
                 sparse_prompt_embeddings=sparse_embeddings,
                 dense_prompt_embeddings=dense_embeddings,
                 multimask_output=False,
                 repeat_image=False,
                 high_res_features=high_res_features
             )
            
             loss = self.compute_loss(low_res_masks, masks)
             
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def train(self):
        # Use our standard preprocessing (SquarePad + Resize + Normalize)
        from src.preprocess import get_training_transform
        transform = get_training_transform(size=1024)
        
        factory = MedicalDatasetFactory(base_data_dir=self.args.data_dir)
        loader = factory.get_loader("sam2", "segmentation", batch_size=2, split=self.args.split, transformer=transform)
        
        # Debug/Demo: Subset
        if self.args.max_samples > 0:
            print(f"DEBUG: Truncating SAM 2 dataset to {self.args.max_samples} samples.")
            dataset = loader.dataset
            indices = list(range(min(len(dataset), self.args.max_samples)))
            dataset = torch.utils.data.Subset(dataset, indices)
            loader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        print("Starting training...")
        self.model.train()
        
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir, exist_ok=True)
        
        loss_log_path = os.path.join(self.args.output_dir, "loss.csv")
        with open(loss_log_path, "w") as f:
             f.write("epoch,loss\n")
        
        for epoch in range(self.args.epochs):
            pbar = tqdm(loader)
            epoch_loss = 0.0
            steps = 0
            for batch in pbar:
                loss = self.train_step(batch)
                epoch_loss += loss
                steps += 1
                pbar.set_description(f"Epoch {epoch} Loss: {loss:.4f}")
            
            avg_loss = epoch_loss / steps if steps > 0 else 0
            with open(loss_log_path, "a") as f:
                 f.write(f"{epoch},{avg_loss:.4f}\n")
        
        # Save
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)
        
        # Save LoRA adapters
        self.model.image_encoder.save_pretrained(self.args.output_dir)
        print(f"Saved SAM 2 Adapter to {self.args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="medical_data")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="outputs/sam2_adapter")
    parser.add_argument("--split", type=str, default="train") # Add split arg
    parser.add_argument("--max_samples", type=int, default=-1, help="Limit samples")
    args = parser.parse_args()
    
    trainer = SAM2Trainer(args)
    trainer.train()
