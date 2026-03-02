import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

from .config import Config
from .medgemma_wrapper import MedGemmaWrapper
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.factory import MedicalDatasetFactory
from torch.utils.data import WeightedRandomSampler
import numpy as np

def collate_fn_vqa(batch, processor, label_map=None):
    """
    Collates batch for VQA fine-tuning with PROPER PROMPT MASKING.
    Labels are set to -100 for prompt tokens so the model only learns
    to predict the answer portion.
    """
    images = []
    texts = []
    
    # Common prompt for classification
    prompt_text = "detect diseases" 
    image_token = getattr(processor, "boi_token", "<image>")
    
    for img, label in batch:
        images.append(img)
        
        # Convert label to text
        target_str = ""
        if isinstance(label, torch.Tensor):
            if label.numel() > 1: # Multi-hot
                indices = torch.where(label == 1)[0]
                if len(indices) == 0:
                    target_str = "No Finding"
                else:
                    if label_map:
                        names = [label_map[i] for i in indices]
                        target_str = ", ".join(names)
            else: # Single class index
                 if label_map:
                     target_str = label_map[label.item()]
                     
        full_text = f"{image_token} {prompt_text} \n {target_str}"
        texts.append(full_text)

    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    # --- PROMPT MASKING ---
    # Compute prompt length once and cache on processor for efficiency
    prompt_len = getattr(processor, '_cached_prompt_len', None)
    if prompt_len is None:
        # Tokenize just the prompt prefix (without answer) to find its length
        prompt_only = f"{image_token} {prompt_text} \n "
        prompt_tokens = processor.tokenizer(prompt_only, add_special_tokens=True)
        prompt_len = len(prompt_tokens["input_ids"])
        # Cache for future calls (same prompt_text every time)
        processor._cached_prompt_len = prompt_len

    # Create labels: input_ids but with prompt positions masked to -100
    labels = inputs["input_ids"].clone()
    # Mask prompt prefix (model shouldn't learn to predict known prompt)
    labels[:, :prompt_len] = -100
    # Mask padding tokens (pad_token_id → -100)
    if hasattr(processor, 'tokenizer') and hasattr(processor.tokenizer, 'pad_token_id'):
        pad_id = processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
    
    inputs["labels"] = labels
    return inputs

def train_lora(dataset_name, root_dir, output_dir="checkpoints/lora_adapter", epochs=3, batch_size=4, lr=2e-4):
    print(f"--- Starting LoRA Training for {dataset_name} ---")
    
    # 1. Load Model (4-bit)
    wrapper = MedGemmaWrapper()
    wrapper.load() # Loads 4-bit quantized model
    model = wrapper.model
    processor = wrapper.processor
    
    # Prepare for LoRA
    print("Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)
    
    # Target modules: linear layers in attention
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 2. Load Dataset
    print(f"Loading Dataset: {dataset_name}")
    factory = MedicalDatasetFactory(root_dir)
    dataset = factory.get_dataset(dataset_name, task="classification", split="train")
    
    # Get Label Map (Class names)
    label_map = getattr(dataset, "NIH_CLASSES", None)
    if label_map is None and hasattr(dataset, "CLASSES"):
        label_map = dataset.CLASSES
        
    # Implement Class-Balanced Sampling for NIH (Heavy "No Finding" Skew)
    print("Computing class weights for balanced sampling...")
    if hasattr(dataset, "df") and "Finding Labels" in dataset.df.columns:
        labels_series = dataset.df["Finding Labels"]
        is_healthy_mask = labels_series.str.contains("No Finding")
        
        num_healthy = is_healthy_mask.sum()
        num_disease = len(dataset) - num_healthy
        
        weight_healthy = 1.0 / max(num_healthy, 1)
        weight_disease = 1.0 / max(num_disease, 1)
        weight_disease *= 5.0
        
        sample_weights = np.where(is_healthy_mask, weight_healthy, weight_disease)
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True
        
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        sampler=sampler,
        collate_fn=lambda b: collate_fn_vqa(b, processor, label_map)
    )
    
    # 3. Optimizer
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=len(loader)*epochs)
    
    # 4. Training Loop
    model.train()
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        total_loss = 0
        
        progress = tqdm(loader)
        for batch in progress:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            # Labels are already pre-masked in collate_fn_vqa
            # Prompt tokens are -100, padding is -100, only answer tokens have real IDs
            outputs = model(**batch)
            
            loss = outputs.loss
            loss.backward()
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            progress.set_postfix({"loss": loss.item()})
            
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
        # Save Adapter
        save_path = os.path.join(output_dir, f"epoch_{epoch+1}")
        model.save_pretrained(save_path)
        print(f"Saved adapter to {save_path}")

if __name__ == "__main__":
    # Example Usage
    # Need to set absolute path
    nih_path = r"d:\MedGamma\medical_data"
    
    if os.path.exists(nih_path):
        train_lora("nih", nih_path, epochs=1) 
    else:
        print("Dataset path not found.")
