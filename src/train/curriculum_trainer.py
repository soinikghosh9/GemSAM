"""
Curriculum Trainer for MedGamma Multi-Task Learning.

Implements a 5-stage curriculum learning approach:
1. Binary Screening (Kaggle + Brain Tumor) - Easiest
2. Modality Detection (Multimodal + SLAKE)
3. Detection with Bboxes (VinDr + NIH) - Most Critical
4. VQA Reasoning (SLAKE + VQA-RAD)
5. SAM2 Segmentation (Pseudo-masks from detection)
"""

import os
import sys
import gc
import torch
from torch.utils.data import DataLoader, ConcatDataset
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import json
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.factory import MedicalDatasetFactory

def no_op_check(*args, **kwargs):
    """No-op validation check for pickling safety."""
    return


class TrainingStage(Enum):
    """Training stages in order of difficulty."""
    SCREENING = 1
    MODALITY = 2
    DETECTION = 3
    VQA = 4
    SEGMENTATION = 5


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping."""
    enabled: bool = True
    patience: int = 5  # Number of evaluation windows without improvement before stopping
    min_delta: float = 0.001  # Minimum improvement to count as progress
    loss_threshold: float = 0.01  # Stop if loss goes below this (model converged)
    eval_interval: int = 100  # Evaluate every N batches


@dataclass
class StageConfig:
    """Configuration for a single training stage."""
    name: str
    datasets: List[str]
    task: str
    epochs: int = 1
    learning_rate: float = 1e-5
    batch_size: int = 4
    prompt_template: str = ""
    max_samples: Optional[int] = None  # Limit for faster experimentation
    early_stopping: Optional[EarlyStoppingConfig] = None  # Per-stage early stopping


@dataclass
class CurriculumConfig:
    """Configuration for the full curriculum."""
    base_data_dir: str = "medical_data"
    data_dir: str = ""  # Alias for base_data_dir (if set, overrides base_data_dir)
    output_dir: str = "checkpoints/curriculum"
    model_id: str = "google/medgemma-1.5-4b-it"  # Updated to correct model name

    # LoRA configuration - optimized for 16GB VRAM
    lora_r: int = 8  # Reduced from 16 to save memory
    lora_alpha: int = 16  # Reduced proportionally (alpha = 2*r is common)
    lora_dropout: float = 0.1

    # Global training settings - optimized for 16GB VRAM
    gradient_accumulation_steps: int = 16  # Increased to compensate for smaller batch
    use_fp16: bool = True
    warmup_ratio: float = 0.1
    max_samples_per_dataset: int = -1  # Global limit per dataset (-1 for unlimited)

    def __post_init__(self):
        # Handle data_dir alias - if set, use it; otherwise use base_data_dir
        if self.data_dir:
            self.base_data_dir = self.data_dir
        else:
            self.data_dir = self.base_data_dir
    
    # Stage configurations with early stopping
    stages: List[StageConfig] = field(default_factory=lambda: [
        StageConfig(
            name="screening",
            datasets=["kaggle_pneumonia", "brain_tumor_mri"],
            task="screening",
            epochs=2,
            learning_rate=2e-5,
            batch_size=1,  # Reduced from 2 to prevent OOM
            prompt_template=(
                "Is this medical image showing signs of disease? "
                "Respond with 'HEALTHY' or 'ABNORMAL'. Then, provide a detailed clinical reasoning explaining your conclusion based on the visible physiological structures and any potential anomalies."
            ),
            early_stopping=EarlyStoppingConfig(
                enabled=True,
                patience=3,
                min_delta=0.005,
                loss_threshold=0.05,  # Simple binary task converges fast
                eval_interval=200  # Increased since batch_size=1 means more batches
            )
        ),
        StageConfig(
            name="modality",
            datasets=["brain_tumor_multimodal", "slake"],
            task="modality",
            epochs=2,
            learning_rate=2e-5,
            batch_size=1,  # Reduced from 2 to prevent OOM
            prompt_template=(
                "What imaging modality was used to capture this medical image? "
                "Respond with one of: X-ray, CT, MRI, Ultrasound."
            ),
            max_samples=2000,
            early_stopping=EarlyStoppingConfig(
                enabled=True,
                patience=3,
                min_delta=0.005,
                loss_threshold=0.05,  # 4-class classification, converges fast
                eval_interval=200
            )
        ),
        StageConfig(
            name="detection",
            datasets=["vindr", "nih"],
            task="detection",
            epochs=3,
            learning_rate=1e-5,
            batch_size=1,  # Reduced from 2 to prevent OOM
            prompt_template=(
                "Analyze this chest X-ray for pathological findings. "
                "For each finding, provide the class name and bounding box coordinates. "
                "Additionally, provide a comprehensive clinical description of the anomaly, its severity, and its exact anatomical location. "
                "Output in JSON format: {\"findings\": [{\"class\": \"...\", \"box\": [x1,y1,x2,y2], \"description\": \"...\"}]}"
            ),
            early_stopping=EarlyStoppingConfig(
                enabled=True,
                patience=5,
                min_delta=0.002,
                loss_threshold=0.02,  # More complex JSON output
                eval_interval=300
            )
        ),
        StageConfig(
            name="vqa",
            datasets=["slake", "vqa_rad"],
            task="vqa",
            epochs=2,
            learning_rate=1e-5,
            batch_size=1,  # Reduced from 2 to prevent OOM
            prompt_template="Answer the following clinical question comprehensively, providing step-by-step medical reasoning to justify your conclusion: ",  # Uses question from dataset
            early_stopping=EarlyStoppingConfig(
                enabled=True,
                patience=5,
                min_delta=0.002,
                loss_threshold=0.03,  # Variable length answers
                eval_interval=200
            )
        ),
        StageConfig(
            name="segmentation",
            datasets=["sam2"],
            task="segmentation",
            epochs=5,
            learning_rate=5e-6,
            batch_size=1,  # Reduced from 2 to prevent OOM
            prompt_template="",  # SAM2 uses box prompts, not text
            early_stopping=EarlyStoppingConfig(
                enabled=True,
                patience=5,
                min_delta=0.005,
                loss_threshold=0.1,  # Segmentation uses different loss scale
                eval_interval=100
            )
        )
    ])


class CurriculumTrainer:
    """
    Multi-stage curriculum trainer for MedGamma.
    
    Progressively trains the model through stages of increasing difficulty,
    allowing it to learn simpler tasks before complex ones.
    """
    
    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.factory = MedicalDatasetFactory(config.base_data_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Training state
        self.current_stage: Optional[TrainingStage] = None
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        
        # Logging
        self.training_log = []
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
    
    def load_model(self):
        """Load MedGemma model with strict config synchronization."""
        from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig, AutoConfig
        from peft import LoraConfig, get_peft_model
        import os
        
        print(f"Loading model: {self.config.model_id}")
        
        # Get HuggingFace token
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        
        # 1. LOAD PROCESSOR & TOKENIZER FIRST
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_id,
            token=hf_token,
            trust_remote_code=True
        )
        
        # 2. PATCH TOKENIZER (Add <image> if missing)
        # We check provided vocab first.
        if "<image>" not in self.processor.tokenizer.get_vocab():
            print("Patching Tokenizer: Adding '<image>' token...")
            self.processor.tokenizer.add_tokens(["<image>"], special_tokens=True)
            
        # CRITICAL FIX: Use the SAME token for model config and full_image_sequence
        # MedGemma uses boi_token (<start_of_image>) natively with ID 255999
        # We must use this token's ID for model_config.image_token_index

        # Check if boi_token exists (MedGemma has this)
        if hasattr(self.processor.tokenizer, "boi_token") and self.processor.tokenizer.boi_token:
            target_image_token = self.processor.tokenizer.boi_token
            image_token_id = self.processor.tokenizer.convert_tokens_to_ids(target_image_token)
            print(f"DEBUG: Using native boi_token '{target_image_token}' with ID {image_token_id}")
        else:
            # Fallback: Use <image> token
            target_image_token = "<image>"
            image_token_id = self.processor.tokenizer.convert_tokens_to_ids(target_image_token)
            print(f"DEBUG: Using fallback '<image>' token with ID {image_token_id}")

        # Store the target token for later use
        self._target_image_token = target_image_token

        # 3. PATCH CONFIG INDEPENDENTLY
        print("Loading Config...")
        model_config = AutoConfig.from_pretrained(self.config.model_id, token=hf_token, trust_remote_code=True)
        
        # Synchronize: Force config to use the ID our tokenizer is using
        if hasattr(model_config, "image_token_index"):
            print(f"DEBUG: Updating config.image_token_index from {model_config.image_token_index} to {image_token_id}")
            model_config.image_token_index = image_token_id
        model_config.ignore_index = -100

        # 4. PATCH IMAGE SEQ LENGTH
        # Get the actual image_seq_length from processor/model config
        # MedGemma typically uses 256 tokens per image, but check what features the model expects
        target_image_token = self._target_image_token

        try:
            # Try to get from processor first, then model config
            image_seq_length = getattr(self.processor, 'image_seq_length', None)
            if image_seq_length is None:
                image_seq_length = getattr(model_config, 'image_seq_length', 256)

            print(f"DEBUG: Using image_seq_length = {image_seq_length}")

            # CRITICAL FIX: Store parameters for direct token injection
            # The text-based expansion approach causes tokenization issues with spaces
            # Instead, we'll inject image token IDs directly into the input_ids tensor
            self._image_token_id = image_token_id
            self._image_seq_length = image_seq_length
            self._use_direct_injection = True

            # Create a simple placeholder for text processing (will be replaced)
            self.processor.full_image_sequence = "<IMAGE>"

            print(f"DEBUG: Using direct token injection strategy")
            print(f"DEBUG: Will inject {image_seq_length} tokens with ID {image_token_id}")

        except Exception as e:
            print(f"Warning: Failed to patch full_image_sequence: {e}")
            import traceback
            traceback.print_exc()

        # Monkey-patch validation check
        if self.processor:
             import types
             self.processor._check_special_mm_tokens = types.MethodType(no_op_check, self.processor)

        # 5. LOAD MODEL WITH UPDATED CONFIG
        quantization_config = None
        if torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

        print("Loading Model with updated config...")
        # CRITICAL FIX: Use AutoModelForImageTextToText (Vision-Language Model)
        # NOT AutoModelForCausalLM (Text-Only Model)
        # This ensures the vision tower is properly loaded and image features are used
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.config.model_id,
            config=model_config,  # INJECT UPDATED CONFIG HERE
            quantization_config=quantization_config,
            torch_dtype=torch.float16 if self.config.use_fp16 else torch.float32,
            device_map="auto",
            trust_remote_code=True,
            token=hf_token
        )

        # Prepare for k-bit training
        from peft import prepare_model_for_kbit_training
        self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=True)
        
        # Resize embeddings (Critical since we added tokens)
        self.model.resize_token_embeddings(len(self.processor.tokenizer))
        
        # Add LoRA with expanded target modules for better VLM fine-tuning
        # Include both attention (q/k/v/o_proj) AND MLP layers (gate/up/down_proj)
        # This allows the model to better adapt its reasoning capabilities
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",  # Attention projections
                "gate_proj", "up_proj", "down_proj"       # MLP projections for better adaptation
            ],
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        return self.model
    
    def get_stage_datasets(self, stage_config: StageConfig) -> List[torch.utils.data.Dataset]:
        """Load datasets for a given stage."""
        datasets = []

        for dataset_name in stage_config.datasets:
            try:
                ds = self.factory.get_dataset(
                    dataset_name,
                    task=stage_config.task,
                    split="train"
                )

                # Determine sample limit (stage-level takes precedence, else use global)
                max_samples = stage_config.max_samples
                if max_samples is None and self.config.max_samples_per_dataset > 0:
                    max_samples = self.config.max_samples_per_dataset

                # CRITICAL FIX: Filter out normal images during detection training
                # The VinDr dataset is 50%+ "No finding" which causes the model
                # to default to "No significant abnormality" too often
                if stage_config.task == "detection":
                    # Filter to only include images WITH pathological findings
                    abnormal_indices = []
                    normal_count = 0
                    max_normal_ratio = 0.1  # Allow only 10% normal images

                    for i in range(len(ds)):
                        try:
                            sample = ds[i]
                            boxes = sample.get("boxes", [])
                            if boxes and len(boxes) > 0:
                                # Has pathology - always include
                                abnormal_indices.append(i)
                            else:
                                # Normal image - include sparingly
                                normal_count += 1
                        except:
                            continue

                    # Add a small portion of normal images (10% of abnormal count)
                    max_normal = int(len(abnormal_indices) * max_normal_ratio)
                    normal_indices = []
                    for i in range(len(ds)):
                        if len(normal_indices) >= max_normal:
                            break
                        try:
                            sample = ds[i]
                            boxes = sample.get("boxes", [])
                            if not boxes or len(boxes) == 0:
                                normal_indices.append(i)
                        except:
                            continue

                    filtered_indices = abnormal_indices + normal_indices
                    print(f"  Detection filter: {len(abnormal_indices)} abnormal + {len(normal_indices)} normal = {len(filtered_indices)} total (was {len(ds)})")
                    ds = torch.utils.data.Subset(ds, filtered_indices)

                # Apply limit if specified
                if max_samples and max_samples > 0 and len(ds) > max_samples:
                    indices = torch.randperm(len(ds))[:max_samples].tolist()
                    ds = torch.utils.data.Subset(ds, indices)

                datasets.append(ds)
                print(f"  Loaded {dataset_name}: {len(ds)} samples")

            except Exception as e:
                print(f"  Warning: Failed to load {dataset_name}: {e}")

        return datasets
    
    def create_prompt(self, sample: Dict, stage_config: StageConfig, image_token: str = "<image>") -> str:
        """Create training prompt with Gemma chat template."""
        task = stage_config.task
        
        # Base user query part
        if task == "screening":
            user_query = f"{image_token} {stage_config.prompt_template}"
            target = "HEALTHY" if sample.get("is_healthy", False) else "ABNORMAL"
            
        elif task == "modality":
            modality = sample.get("modality", "unknown")
            modality_map = {"xray": "X-ray", "ct": "CT", "mri": "MRI", "ultrasound": "Ultrasound"}
            target = modality_map.get(modality.lower(), modality)
            user_query = f"{image_token} {stage_config.prompt_template}"
            
        elif task == "detection":
            boxes = sample.get("boxes", [])
            labels = sample.get("labels", [])
            
            if not boxes:
                target_json = json.dumps({"findings": [{"class": "No significant abnormality", "box": [0,0,0,0], "description": "No acute cardiopulmonary abnormality identified. Heart size and mediastinal contour are within normal limits. Lungs are clear bilaterally without focal consolidation, effusion, or pneumothorax."}]})
            else:
                findings = []
                for box, label in zip(boxes, labels):
                    # Add a robust clinical description based on the label
                    desc = f"Pathological presence of {label} detected in the specified region, indicating an underlying clinical anomaly requiring attention."
                    findings.append({"class": label, "box": [int(b) for b in box], "description": desc})
                target_json = json.dumps({"findings": findings})
            
            user_query = f"{image_token} {stage_config.prompt_template}"
            target = target_json
            
        elif task == "vqa":
            question = sample.get("question", "")
            answer = sample.get("answer", "")
            user_query = f"{image_token} Question: {question}"
            target = answer
            
        else:
            user_query = f"{image_token} {stage_config.prompt_template}"
            target = ""

        # Format with Gemma chat template
        # <start_of_turn>user\n{query}<end_of_turn>\n<start_of_turn>model\n{answer}<end_of_turn>
        full_prompt = f"<start_of_turn>user\n{user_query}<end_of_turn>\n<start_of_turn>model\n{target}<end_of_turn>"
        
        return full_prompt
    
    def train_stage(self, stage: TrainingStage, stage_config: StageConfig):
        """Train a single stage with early stopping support."""
        import time
        import gc

        stage_start_time = time.time()

        print(f"\n{'='*60}")
        print(f"STAGE {stage.value}: {stage_config.name.upper()}")
        print(f"{'='*60}")
        print(f"Datasets: {stage_config.datasets}")
        print(f"Epochs: {stage_config.epochs}, LR: {stage_config.learning_rate}", flush=True)

        # Early stopping configuration
        es_config = stage_config.early_stopping
        if es_config and es_config.enabled:
            print(f"Early Stopping: enabled (patience={es_config.patience}, "
                  f"min_delta={es_config.min_delta}, threshold={es_config.loss_threshold})")
        else:
            print("Early Stopping: disabled")

        # Skip segmentation for now (requires SAM2)
        if stage == TrainingStage.SEGMENTATION:
            print("Skipping segmentation stage (requires separate SAM2 training)")
            return

        # Load datasets
        datasets = self.get_stage_datasets(stage_config)
        if not datasets:
            print("No datasets loaded, skipping stage")
            return

        # Combine datasets
        combined = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
        print(f"Total training samples: {len(combined)}")

        # Create dataloader
        dataloader = DataLoader(
            combined,
            batch_size=stage_config.batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 to fix Windows multiprocessing spawn error
            collate_fn=self._collate_fn
        )

        # Training loop
        self.model.train()

        print(f"Starting training: {len(dataloader)} batches, batch_size={stage_config.batch_size}, "
              f"grad_accum={self.config.gradient_accumulation_steps}", flush=True)

        # Early stopping state
        best_loss = float('inf')
        patience_counter = 0
        early_stop_triggered = False
        window_loss_sum = 0.0
        window_steps = 0
        global_batch_idx = 0  # Track across epochs
        first_loss_check_done = False  # For immediate convergence check

        # Clear cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Ensure GPU is ready
            
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=stage_config.learning_rate
        )
        
        total_loss = 0
        step_count = 0
        
        # Image token - use the same token that model_config.image_token_index was set to
        image_token = getattr(self, "_target_image_token", "<image>")
        
        for epoch in range(stage_config.epochs):
            epoch_loss = 0
            epoch_steps = 0
            
            for batch_idx, batch in enumerate(dataloader):
                try:
                    # Get images directly from batch
                    images_raw = batch["image"]
                    batch_size = len(images_raw)
                    
                    # Ensure images are PIL Images
                    from PIL import Image
                    processed_images = []
                    for img in images_raw:
                        if isinstance(img, str):
                            processed_images.append(Image.open(img).convert("RGB"))
                        elif isinstance(img, Image.Image):
                            processed_images.append(img.convert("RGB"))
                        else:
                            try:
                                # Start with a fallback
                                processed_images.append(img)
                            except:
                                pass
                    
                    # Verify images
                    if len(processed_images) != batch_size:
                         print(f"Warning: Image processing mismatch {len(processed_images)} vs {batch_size}")
                         continue

                    # Create prompts
                    prompts = []
                    for i in range(batch_size):
                        sample = {}
                        for k, v in batch.items():
                            if isinstance(v, list) and len(v) > i:
                                sample[k] = v[i]
                            else:
                                sample[k] = v
                        
                        # Pass image_token to create_prompt
                        prompt = self.create_prompt(sample, stage_config, image_token)
                        prompts.append(prompt)
                    
                    # ROBUST MANUAL BATCHING
                    # Process each item individually to avoid "inconsistent batch" error
                    input_ids_list = []
                    pixel_values_list = []
                    labels_list = []
                    attention_mask_list = []
                    
                    for p, img in zip(prompts, processed_images):
                        # DIRECT TOKEN INJECTION STRATEGY
                        # Instead of text expansion (which causes tokenization issues),
                        # we remove the image placeholder and inject token IDs directly

                        # Remove image_token from prompt for clean tokenization
                        p_clean = p.replace(image_token, "")

                        # 1. Tokenize text WITHOUT image tokens
                        text_inputs = self.processor.tokenizer(
                            p_clean,
                            return_tensors="pt",
                            truncation=True,
                            max_length=300  # Shorter since no image tokens
                        )

                        # 2. Process image
                        image_inputs = self.processor.image_processor(
                            img,
                            return_tensors="pt"
                        )

                        input_ids = text_inputs["input_ids"].squeeze(0)
                        attention_mask = text_inputs["attention_mask"].squeeze(0)

                        # 3. DIRECT TOKEN INJECTION: Insert exactly 256 image tokens
                        img_token_id = getattr(self, '_image_token_id', self.model.config.image_token_index)
                        img_seq_len = getattr(self, '_image_seq_length', 256)

                        # Create image token tensor
                        img_tokens = torch.full((img_seq_len,), img_token_id, dtype=input_ids.dtype)

                        # Find insertion point - after "<bos><start_of_turn>user\n" (typically position 4)
                        # Look for the newline after "user"
                        user_token_ids = self.processor.tokenizer.encode("user\n", add_special_tokens=False)
                        insert_pos = 4  # Default fallback

                        # Search for actual position
                        ids_list = input_ids.tolist()
                        for i in range(min(10, len(ids_list) - len(user_token_ids))):
                            if ids_list[i:i+len(user_token_ids)] == user_token_ids:
                                insert_pos = i + len(user_token_ids)
                                break

                        # Insert image tokens at the correct position
                        input_ids = torch.cat([
                            input_ids[:insert_pos],
                            img_tokens,
                            input_ids[insert_pos:]
                        ])
                        attention_mask = torch.cat([
                            attention_mask[:insert_pos],
                            torch.ones(img_seq_len, dtype=attention_mask.dtype),
                            attention_mask[insert_pos:]
                        ])

                        input_ids_list.append(input_ids)
                        attention_mask_list.append(attention_mask)
                        pixel_values_list.append(image_inputs["pixel_values"].squeeze(0))

                    # Pad and Stack
                    from torch.nn.utils.rnn import pad_sequence
                    
                    # Pad input_ids and masks
                    input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id)
                    attention_mask_padded = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
                    
                    # Stack pixel values (assuming same size)
                    pixel_values_stacked = torch.stack(pixel_values_list)
                    
                    # Create labels with PROPER MASKING for instruction-tuning
                    # CRITICAL: Only compute loss on model response, NOT on user prompt/image tokens
                    labels = input_ids_padded.clone()

                    # Find the position of "<start_of_turn>model\n" to mask everything before it
                    model_turn_marker = "<start_of_turn>model\n"
                    model_turn_ids = self.processor.tokenizer.encode(model_turn_marker, add_special_tokens=False)

                    for batch_i in range(labels.shape[0]):
                        seq = input_ids_padded[batch_i].tolist()
                        # Find where model turn starts
                        model_start_idx = -1
                        for pos in range(len(seq) - len(model_turn_ids) + 1):
                            if seq[pos:pos + len(model_turn_ids)] == model_turn_ids:
                                model_start_idx = pos + len(model_turn_ids)  # Start AFTER the marker
                                break

                        if model_start_idx > 0:
                            # Mask all tokens BEFORE model response (user prompt + image tokens)
                            labels[batch_i, :model_start_idx] = -100

                    # Also mask pad tokens
                    if self.processor.tokenizer.pad_token_id is not None:
                         labels[input_ids_padded == self.processor.tokenizer.pad_token_id] = -100

                    # Create token_type_ids (zeros, same shape as input_ids)
                    token_type_ids = torch.zeros_like(input_ids_padded)

                    inputs = {
                        "input_ids": input_ids_padded.to(self.device),
                        "pixel_values": pixel_values_stacked.to(self.device),
                        "attention_mask": attention_mask_padded.to(self.device),
                        "token_type_ids": token_type_ids.to(self.device),
                        "labels": labels.to(self.device)
                    }

                    # DEBUG DIAGNOSTICS FOR TOKEN MISMATCH
                    if batch_idx == 0 and epoch == 0:
                        try:
                            print("\n--- DEBUG: Pre-Forward Diagnostics ---")
                            print(f"Model Image Token Index: {self.model.config.image_token_index}")
                            
                            # Check first sample
                            sample_ids = inputs["input_ids"][0].cpu().tolist()
                            img_token_count = sample_ids.count(self.model.config.image_token_index)
                            print(f"Input IDs Count of Image Token ({self.model.config.image_token_index}): {img_token_count}")
                            
                            # Print first 20 tokens to see what's actually there
                            print(f"First 20 tokens: {sample_ids[:20]}")
                            
                            # Check if token exists in vocab
                            decoded = self.processor.tokenizer.decode(sample_ids[:20])
                            print(f"Decoded first 20: {decoded}")
                            
                            # Check pixel values
                            print(f"Pixel Values Shape: {inputs['pixel_values'].shape}")
                            print("----------------------------------------\n")
                        except Exception as e:
                            print(f"Debug Print Failed: {e}")

                    # Forward pass
                    outputs = self.model(**inputs)
                    loss = outputs.loss / self.config.gradient_accumulation_steps

                    # Check for NaN loss
                    if torch.isnan(loss):
                        print(f"  WARNING: NaN loss at batch {batch_idx}, skipping", flush=True)
                        optimizer.zero_grad()
                        continue

                    # Backward pass
                    loss.backward()

                    # Optimizer step with gradient accumulation
                    if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                        # Gradient clipping for stability
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        step_count += 1

                    epoch_loss += loss.item() * self.config.gradient_accumulation_steps
                    epoch_steps += 1

                    # Clear CUDA cache frequently to prevent memory buildup (critical for 16GB GPU)
                    if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()  # Also run garbage collection

                    # Track loss for early stopping window
                    window_loss_sum += loss.item() * self.config.gradient_accumulation_steps
                    window_steps += 1
                    global_batch_idx += 1

                    # Log progress more frequently (every 10 batches or last batch) with memory info
                    is_last_batch = (batch_idx == len(dataloader) - 1)
                    if batch_idx % 10 == 0 or is_last_batch:
                        avg_loss = epoch_loss / max(epoch_steps, 1)
                        mem_info = ""
                        if torch.cuda.is_available():
                            mem_used = torch.cuda.memory_allocated() / 1024**3
                            mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                            mem_info = f", GPU: {mem_used:.1f}/{mem_total:.1f}GB"
                        print(f"  Epoch {epoch+1}/{stage_config.epochs}, "
                              f"Batch {batch_idx}/{len(dataloader)}, "
                              f"Loss: {avg_loss:.4f}{mem_info}", flush=True)

                    # Early stopping check at evaluation intervals
                    if es_config and es_config.enabled:
                        # Immediate convergence check after first 10 batches (catch already-converged models)
                        if not first_loss_check_done and global_batch_idx >= 10:
                            first_loss_check_done = True
                            current_loss = epoch_loss / max(epoch_steps, 1)
                            if current_loss < es_config.loss_threshold:
                                print(f"\n  [EARLY STOP] Initial loss {current_loss:.6f} already below threshold {es_config.loss_threshold}")
                                print(f"  Model appears already converged. Stopping training early.")
                                early_stop_triggered = True
                                break

                        # Regular interval check
                        if global_batch_idx % es_config.eval_interval == 0 and window_steps > 0:
                            window_avg_loss = window_loss_sum / window_steps

                            # Check absolute threshold (model has converged)
                            if window_avg_loss < es_config.loss_threshold:
                                print(f"\n  [EARLY STOP] Loss {window_avg_loss:.6f} < threshold {es_config.loss_threshold}")
                                print(f"  Model has converged. Stopping training early.")
                                early_stop_triggered = True
                                break

                            # Check for improvement
                            improvement = best_loss - window_avg_loss
                            if improvement > es_config.min_delta:
                                is_first_best = (best_loss == float('inf'))
                                best_loss = window_avg_loss
                                patience_counter = 0
                                if is_first_best:
                                    print(f"  [ES] Initial best loss: {best_loss:.6f}")
                                else:
                                    print(f"  [ES] New best loss: {best_loss:.6f} (improved by {improvement:.6f})")
                            else:
                                patience_counter += 1
                                print(f"  [ES] No improvement. Patience: {patience_counter}/{es_config.patience}")

                                if patience_counter >= es_config.patience:
                                    print(f"\n  [EARLY STOP] No improvement for {es_config.patience} evaluations")
                                    print(f"  Best loss: {best_loss:.6f}. Stopping training early.")
                                    early_stop_triggered = True
                                    break

                            # Reset window
                            window_loss_sum = 0.0
                            window_steps = 0

                except Exception as e:
                    print(f"  Error in batch {batch_idx}: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    continue

            # Final optimizer step for remaining gradients at end of epoch (if not early stopped)
            if not early_stop_triggered and epoch_steps % self.config.gradient_accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            # Epoch complete (or stopped early) - ALWAYS compute and track loss
            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            epoch_time = time.time() - stage_start_time
            if early_stop_triggered:
                print(f"  Epoch {epoch+1} stopped early. Avg Loss: {avg_epoch_loss:.4f} (took {epoch_time:.1f}s)", flush=True)
            else:
                print(f"  Epoch {epoch+1} complete. Avg Loss: {avg_epoch_loss:.4f} (took {epoch_time:.1f}s)", flush=True)
            total_loss += avg_epoch_loss  # Track loss even for partial epochs

            # Check if early stop was triggered - break AFTER tracking loss
            if early_stop_triggered:
                break

        # Save stage checkpoint
        print(f"  Saving checkpoint to {stage_config.name}...", flush=True)
        stage_dir = os.path.join(self.config.output_dir, stage_config.name)
        os.makedirs(stage_dir, exist_ok=True)
        self.model.save_pretrained(stage_dir)
        print(f"  Checkpoint saved.", flush=True)

        # Log results with early stopping info
        epochs_completed = (epoch + 1) if 'epoch' in dir() else 0
        log_entry = {
            "stage": stage_config.name,
            "epochs_planned": stage_config.epochs,
            "epochs_completed": epochs_completed,
            "samples": len(combined),
            "avg_loss": total_loss / max(epochs_completed, 1),  # Divide by ACTUAL epochs, not planned
            "best_loss": best_loss if best_loss != float('inf') else None,
            "early_stopped": early_stop_triggered,
            "timestamp": datetime.now().isoformat()
        }
        self.training_log.append(log_entry)

        if early_stop_triggered:
            print(f"  [SUMMARY] Stage completed via early stopping. Best loss: {best_loss:.6f}")
        
        # Stage timing and cleanup
        stage_duration = time.time() - stage_start_time
        print(f"Stage {stage_config.name} complete in {stage_duration:.1f}s. Checkpoint saved to {stage_dir}", flush=True)

        # Force garbage collection and CUDA cache clear between stages
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def train_all(self, start_stage: TrainingStage = TrainingStage.SCREENING):
        """Train all stages sequentially."""
        print("\n" + "="*80)
        print("CURRICULUM TRAINING - MedGamma Multi-Task")
        print("="*80)
        
        # Load model if not already loaded
        if self.model is None:
            self.load_model()
        
        # Get stages to train
        stages_to_train = [s for s in TrainingStage if s.value >= start_stage.value]
        print(f"Stages to train: {[s.name for s in stages_to_train]}", flush=True)

        for idx, stage in enumerate(stages_to_train):
            print(f"\n>>> Starting stage {idx+1}/{len(stages_to_train)}: {stage.name}", flush=True)

            # Find matching config
            stage_config = None
            for cfg in self.config.stages:
                if cfg.name.upper() == stage.name:
                    stage_config = cfg
                    break

            if stage_config is None:
                print(f"No config found for stage {stage.name}, skipping", flush=True)
                continue

            self.current_stage = stage
            self.train_stage(stage, stage_config)
            print(f"<<< Finished stage {idx+1}/{len(stages_to_train)}: {stage.name}\n", flush=True)
        
        # Save final checkpoint and log
        print("\n>>> Saving final checkpoint...", flush=True)
        final_dir = os.path.join(self.config.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        self.model.save_pretrained(final_dir)
        print(f"Final checkpoint saved to {final_dir}", flush=True)

        log_path = os.path.join(self.config.output_dir, "training_log.json")
        with open(log_path, "w") as f:
            json.dump(self.training_log, f, indent=2)
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print(f"Final checkpoint: {final_dir}")
        print(f"Training log: {log_path}")
        print("="*80)
    
    def _collate_fn(self, batch):
        """Collate function that handles variable-length data and mixed dataset keys."""
        if not batch:
            return {}

        # Get intersection of all keys (keys that exist in ALL items)
        all_keys = [set(d.keys()) for d in batch]
        common_keys = set.intersection(*all_keys) if all_keys else set()

        # Always include 'image' - it's required
        if 'image' not in common_keys and any('image' in d for d in batch):
            common_keys.add('image')

        result = {}
        for key in common_keys:
            result[key] = [d.get(key) for d in batch]

        # For non-common keys, include them with None for missing items
        # This allows the training loop to handle different dataset types
        all_possible_keys = set.union(*all_keys) if all_keys else set()
        for key in all_possible_keys - common_keys:
            result[key] = [d.get(key) for d in batch]

        return result


def main():
    """Main entry point for curriculum training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Curriculum Training for MedGamma")
    parser.add_argument("--data-dir", default="medical_data", help="Base data directory")
    parser.add_argument("--output-dir", default="checkpoints/curriculum", help="Output directory")
    parser.add_argument("--start-stage", default="screening", 
                        choices=["screening", "modality", "detection", "vqa", "segmentation"],
                        help="Stage to start from")
    parser.add_argument("--quick", action="store_true", help="Quick test with limited samples")
    
    args = parser.parse_args()
    
    # Create config
    config = CurriculumConfig(
        base_data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    # Limit samples for quick testing
    if args.quick:
        for stage in config.stages:
            stage.max_samples = 100
            stage.epochs = 1
    
    # Map stage name to enum
    stage_map = {
        "screening": TrainingStage.SCREENING,
        "modality": TrainingStage.MODALITY,
        "detection": TrainingStage.DETECTION,
        "vqa": TrainingStage.VQA,
        "segmentation": TrainingStage.SEGMENTATION
    }
    
    start_stage = stage_map[args.start_stage]
    
    # Create trainer and run
    trainer = CurriculumTrainer(config)
    trainer.train_all(start_stage=start_stage)


if __name__ == "__main__":
    main()
