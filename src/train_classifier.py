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
    Collates batch for VQA fine-tuning.
    Converts tensor labels to text targets if needed.
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
                    # We need the class names. 
                    # Assuming we pass label_map (list of names)
                    if label_map:
                        names = [label_map[i] for i in indices]
                        target_str = ", ".join(names)
            else: # Single class index
                 if label_map:
                     target_str = label_map[label.item()]
                     
        # Format: <image> detect diseases <bos> Answer
        # Note: MedGemma/PaliGemma formatting is specific. 
        # Usually: "<image> prompt \n answer"
        full_text = f"{image_token} {prompt_text} \n {target_str}"
        texts.append(full_text)

    # Tokenizer inputs
    # Note: Training VLM usually requires inputs["labels"] matching input_ids
    # We let the processor handle it if possible, or manual.
    # PaliGemma processor handles images and text.
    
    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

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
    # For Gemma/PaliGemma: q_proj, k_proj, v_proj, o_proj
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM # or FEATURE_EXTRACTION? MedGemma is CausalLM usually
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 2. Load Dataset
    print(f"Loading Dataset: {dataset_name}")
    factory = MedicalDatasetFactory(root_dir)
    # Using MedicalDatasetFactory.get_dataset instead of get_loader to build our own balanced loader
    dataset = factory.get_dataset(dataset_name, task="classification", split="train")
    
    # Get Label Map (Class names)
    label_map = getattr(dataset, "NIH_CLASSES", None)
    if label_map is None and hasattr(dataset, "CLASSES"):
        label_map = dataset.CLASSES
        
    # Implement Class-Balanced Sampling for NIH (Heavy "No Finding" Skew)
    print("Computing class weights for balanced sampling...")
    if hasattr(dataset, "df") and "Finding Labels" in dataset.df.columns:
        # NIH Dataset logic
        labels_series = dataset.df["Finding Labels"]
        is_healthy_mask = labels_series.str.contains("No Finding")
        
        # Calculate weights: minority classes get higher weight
        num_healthy = is_healthy_mask.sum()
        num_disease = len(dataset) - num_healthy
        
        weight_healthy = 1.0 / max(num_healthy, 1)
        weight_disease = 1.0 / max(num_disease, 1)
        
        # Give extra weight to diseases
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
            
            # Forward
            # VLM Causal Loss: labels = input_ids (masked user prompt usually)
            # Simple approach: Train on everything (Prompt + Answer)
            # Better approach: Mask prompt.
            # processor usually handles 'labels' generation if requested? No.
            
            # We clone input_ids as labels
            labels = batch["input_ids"].clone()
            
            # TODO: Improve masking. For now, training on full sequence (Prompt+A)
            # Ideally we mask the tokens corresponding to "<image> detect diseases \n"
            
            outputs = model(
                **batch,
                labels=labels
            )
            
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
