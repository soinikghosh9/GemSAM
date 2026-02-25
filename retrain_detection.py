"""
Focused Detection Retraining for MedGamma.

Retrains ONLY the detection adapter with class-balanced sampling to fix
the zero-score detection issue caused by VinDr's 70% "No finding" imbalance.

Key changes from original training:
- Class-balanced sampling: 30% normal, 70% abnormal images
- Higher LoRA rank (r=16) for more capacity
- 5 epochs with delayed early stopping (disabled for first 3 epochs)
- Higher learning rate (2e-5)

Usage:
    python retrain_detection.py                    # Full retraining
    python retrain_detection.py --quick            # Quick test (100 samples)
    python retrain_detection.py --epochs 3         # Custom epochs
"""

import os
import sys
import gc
import json
import time
import argparse
import random
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from transformers import get_cosine_schedule_with_warmup


def collate_fn(batch):
    """Custom collate for medical datasets."""
    result = {}
    for key in batch[0].keys():
        result[key] = [item[key] for item in batch]
    return result

def main():
    parser = argparse.ArgumentParser(description="Retrain Detection Adapter (Class-Balanced)")
    parser.add_argument("--data_dir", default="medical_data", help="Base data directory")
    parser.add_argument("--output_dir", default="checkpoints/production/medgemma/detection",
                        help="Output directory for retrained adapter")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--max_normal", type=int, default=1000,
                        help="Max normal (no-finding) samples to keep")
    # SAFE DEFAULTS for 16GB VRAM (4-bit quantization)
    # Batch size 8 OOM'd. Batch size 4 uses ~8-9GB and is safe.
    # We use grad_accum 4 to reach effective batch size 16.
    # CRITICAL: We also force 448x448 resolution which is 4x faster than default 896x896
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (4 is safe for 16GB)")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps (Target effective batch ~16)")
    parser.add_argument("--quick", action="store_true", help="Quick test mode (100 samples)")
    parser.add_argument("--resume_from", default=None,
                        help="Resume from existing adapter (fine-tune further)")
    args = parser.parse_args()

    # Memory Optimization
    # NOTE: expandable_segments not supported on all Windows setups, removing to avoid warning
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # CRITICAL: Sanitize sys.argv so imported modules don't crash
    sys.argv = [sys.argv[0]]

    print("=" * 60)
    print("  MedGamma Detection Retraining (Class-Balanced)")
    print("=" * 60)
    print(f"  Output: {args.output_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  LR: {args.lr}")
    print(f"  LoRA r: {args.lora_r}")
    print(f"  Max Normal: {args.max_normal}")
    print(f"  Quick: {args.quick}")
    print("=" * 60)

    # =========================================================================
    # Step 1: Load and balance dataset
    # =========================================================================
    print("\n[1/4] Loading VinDr dataset with class-balanced sampling...")

    from data.factory import MedicalDatasetFactory
    factory = MedicalDatasetFactory(args.data_dir)
    dataset = factory.get_dataset("vindr", task="detection", split="train")

    total_len = len(dataset)
    print(f"  Total VinDr training samples: {total_len}")

    # Separate normal vs abnormal indices
    normal_indices = []
    abnormal_indices = []

    print("  Scanning for normal vs abnormal samples...")
    for idx in range(total_len):
        try:
            sample = dataset[idx]
            boxes = sample.get("boxes", [])
            labels = sample.get("labels", [])

            # Check if this is a "No finding" sample
            if not boxes and not labels:
                normal_indices.append(idx)
            elif isinstance(labels, list) and len(labels) == 1 and \
                 isinstance(labels[0], str) and labels[0].lower() == "no finding":
                normal_indices.append(idx)
            else:
                abnormal_indices.append(idx)
        except Exception:
            continue

        if idx % 2000 == 0 and idx > 0:
            print(f"    Scanned {idx}/{total_len}... "
                  f"({len(normal_indices)} normal, {len(abnormal_indices)} abnormal)")

    print(f"  Normal: {len(normal_indices)}, Abnormal: {len(abnormal_indices)}")

    # Cap normal samples
    random.seed(42)
    if len(normal_indices) > args.max_normal:
        random.shuffle(normal_indices)
        normal_indices = normal_indices[:args.max_normal]
        print(f"  Capped normal to {args.max_normal}")

    # Quick mode
    if args.quick:
        abnormal_indices = abnormal_indices[:80]
        normal_indices = normal_indices[:20]
        print(f"  [QUICK MODE] Using {len(abnormal_indices)} abnormal + {len(normal_indices)} normal")

    # Combine and create balanced subset
    balanced_indices = abnormal_indices + normal_indices
    random.shuffle(balanced_indices)

    balanced_dataset = Subset(dataset, balanced_indices)
    pct_abnormal = len(abnormal_indices) / max(len(balanced_indices), 1) * 100

    print(f"  Balanced dataset: {len(balanced_dataset)} samples "
          f"({pct_abnormal:.1f}% abnormal, {100 - pct_abnormal:.1f}% normal)")

    # =========================================================================
    # Step 2: Load model with LoRA
    # =========================================================================
    print(f"\n[2/4] Loading MedGemma with LoRA r={args.lora_r}...")

    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig, AutoConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
    import types

    model_id = "google/medgemma-1.5-4b-it"
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    # Load processor
    processor = AutoProcessor.from_pretrained(model_id, token=hf_token, trust_remote_code=True)

    # CRITICAL FIX: Use native image token to avoid embedding mismatch
    # Do NOT add new tokens or resize embeddings if we want LORA to work without saving the full model
    
    # Check for boi_token (MedGemma native)
    # NOTE: boi_token might be an empty string but still exist! Check attribute existence.
    if hasattr(processor.tokenizer, "boi_token") and processor.tokenizer.boi_token is not None:
        # The boi_token might be an empty string depending on version, but check ID
        target_image_token = processor.tokenizer.boi_token
        print(f"  Using native boi_token: '{target_image_token}'")
    else:
        # Fallback only if absolutely necessary, but print WARNING
        print("  WARNING: boi_token not found. Using <image> (May cause embedding mismatch if not saved)")
        target_image_token = "<image>"
        if target_image_token not in processor.tokenizer.get_vocab():
            processor.tokenizer.add_tokens([target_image_token], special_tokens=True)
            # We must resize if we added a token, but this is dangerous for LoRA
            model.resize_token_embeddings(len(processor.tokenizer)) 
    
    # Verify ID is within bounds
    image_token_id = processor.tokenizer.convert_tokens_to_ids(target_image_token)
    print(f"  Image token ID: {image_token_id}")

    # Load config
    model_config = AutoConfig.from_pretrained(model_id, token=hf_token, trust_remote_code=True)
    if hasattr(model_config, "image_token_index"):
        model_config.image_token_index = image_token_id
    model_config.ignore_index = -100

    # Get image_seq_length
    image_seq_length = getattr(processor, 'image_seq_length', None)
    if image_seq_length is None:
        image_seq_length = getattr(model_config, 'image_seq_length', 256)
    print(f"  Image seq length: {image_seq_length}")

    # Monkey-patch
    def no_op_check(*args, **kwargs):
        return
    processor._check_special_mm_tokens = types.MethodType(no_op_check, processor)

    # Quantization
    quantization_config = None
    if torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

    # Load model
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        config=model_config,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token
    )

    model = prepare_model_for_kbit_training(
        model, 
        use_gradient_checkpointing=True, 
        gradient_checkpointing_kwargs={'use_reentrant': False}
    )
    
    # CRITICAL: Do NOT resize embeddings if we are using native token
    # model.resize_token_embeddings(len(processor.tokenizer))

    # Apply LoRA - if resuming, load existing adapter first
    if args.resume_from and os.path.exists(os.path.join(args.resume_from, "adapter_config.json")):
        print(f"  Resuming from existing adapter: {args.resume_from}")
        model = PeftModel.from_pretrained(model, args.resume_from, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,  # alpha = 2*r
            lora_dropout=0.1,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Memory Check
    if torch.cuda.is_available():
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU VRAM: {mem_total:.1f} GB")
        if mem_total < 10 and args.batch_size > 2:
            print("  [WARNING] Low VRAM (<10GB) detected. If OOM occurs, try --batch_size 2 --grad_accum 8")
        elif mem_total >= 16 and args.batch_size < 4:
            print("  [TIP] You have 16GB+ VRAM. You can increase speed with --batch_size 8 --grad_accum 2")
            
    # =========================================================================
    # Step 3: Training Loop
    # =========================================================================
    print(f"\n[3/4] Training for {args.epochs} epochs...")

    detection_prompt = (
        "Analyze this chest X-ray for pathological findings. "
        "For each finding, provide the class name and bounding box coordinates. "
        'Output in JSON format: {"findings": [{"class": "...", "box": [x1,y1,x2,y2]}]}'
    )

    from PIL import Image
    from torch.nn.utils.rnn import pad_sequence

    
    # Windows-safe worker count
    # Increasing workers to prevent GPU starvation
    num_workers = 4 
    if os.name != 'nt':
        num_workers = 8

    dataloader = DataLoader(
        balanced_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False
    )

    from transformers import get_cosine_schedule_with_warmup

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Cosine Scheduler with 10% warmup
    num_training_steps = len(dataloader) * args.epochs
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )
    
    model.train()

    # Training state
    best_loss = float('inf')
    patience_counter = 0
    # RELAXED EARLY STOPPING: 5 epochs is short, don't stop easily
    early_stop_patience = 5   # Allow full training if needed
    no_early_stop_before_epoch = 3  # Allow warm-up for first 3 epochs

    training_log = []
    total_batches = len(dataloader)
    best_adapter_dir = args.output_dir + "_best"

    for epoch in range(args.epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_steps = 0
        skipped = 0

        for batch_idx, batch in enumerate(dataloader):
            try:
                images_raw = batch["image"]
                batch_size = len(images_raw)

                # Ensure PIL Images
                processed_images = []
                for img in images_raw:
                    if isinstance(img, str):
                        processed_images.append(Image.open(img).convert("RGB"))
                    elif isinstance(img, Image.Image):
                        processed_images.append(img.convert("RGB"))
                    else:
                        processed_images.append(img)

                if len(processed_images) != batch_size:
                    skipped += 1
                    continue

                # Create prompts for each sample
                input_ids_list = []
                attention_mask_list = []
                pixel_values_list = []

                for i in range(batch_size):
                    # Get target JSON for this sample
                    boxes = batch.get("boxes", [[]])[i]
                    labels = batch.get("labels", [[]])[i]

                    # IMPORTANT: Normalize findings
                    findings = []
                    if boxes:
                        for box, label in zip(boxes, labels):
                            findings.append({
                                "class": label,
                                "box": [int(b) for b in box]
                            })
                    
                    target_json = json.dumps({"findings": findings})

                    # Build full prompt with chat template
                    # Use specific format for training
                    full_prompt = (
                        f"<start_of_turn>user\n{target_image_token} {detection_prompt}"
                        f"<end_of_turn>\n<start_of_turn>model\n{target_json}<end_of_turn>"
                    )
                    
                    # Tokenize (without image token)
                    p_clean = full_prompt.replace(target_image_token, "")
                    text_inputs = processor.tokenizer(
                        p_clean,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512 # Reduced from 768 for speed, still enough for most samples
                    )

                    # Process image (Default resolution 896x896 to avoid mismatch error)
                    image_inputs = processor.image_processor(
                        processed_images[i],
                        return_tensors="pt"
                    )

                    input_ids = text_inputs["input_ids"].squeeze(0)
                    attention_mask = text_inputs["attention_mask"].squeeze(0)

                    # Direct token injection
                    img_tokens = torch.full(
                        (image_seq_length,), image_token_id, dtype=input_ids.dtype
                    )

                    # Find insertion point
                    user_token_ids = processor.tokenizer.encode("user\n", add_special_tokens=False)
                    insert_pos = 4
                    ids_list = input_ids.tolist()
                    for pos in range(min(10, len(ids_list) - len(user_token_ids))):
                        if ids_list[pos:pos + len(user_token_ids)] == user_token_ids:
                            insert_pos = pos + len(user_token_ids)
                            break

                    input_ids = torch.cat([
                        input_ids[:insert_pos], img_tokens, input_ids[insert_pos:]
                    ])
                    attention_mask = torch.cat([
                        attention_mask[:insert_pos],
                        torch.ones(image_seq_length, dtype=attention_mask.dtype),
                        attention_mask[insert_pos:]
                    ])

                    input_ids_list.append(input_ids)
                    attention_mask_list.append(attention_mask)
                    pixel_values_list.append(image_inputs["pixel_values"].squeeze(0))

                # Pad and stack
                pad_id = processor.tokenizer.pad_token_id or 0
                input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
                attention_mask_padded = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
                pixel_values_stacked = torch.stack(pixel_values_list)

                # Create labels with proper masking
                labels = input_ids_padded.clone()

                # Mask everything before model response
                model_turn_marker = "<start_of_turn>model\n"
                model_turn_ids = processor.tokenizer.encode(model_turn_marker, add_special_tokens=False)

                for bi in range(labels.shape[0]):
                    seq = input_ids_padded[bi].tolist()
                    model_start_idx = -1
                    for pos in range(len(seq) - len(model_turn_ids) + 1):
                        if seq[pos:pos + len(model_turn_ids)] == model_turn_ids:
                            model_start_idx = pos + len(model_turn_ids)
                            break
                    if model_start_idx > 0:
                        labels[bi, :model_start_idx] = -100

                # Mask pad tokens
                if pad_id is not None:
                    labels[input_ids_padded == pad_id] = -100

                token_type_ids = torch.zeros_like(input_ids_padded)

                inputs = {
                    "input_ids": input_ids_padded.to(device),
                    "pixel_values": pixel_values_stacked.to(device),
                    "attention_mask": attention_mask_padded.to(device),
                    "token_type_ids": token_type_ids.to(device),
                    "labels": labels.to(device)
                }

                # Debug first batch
                if batch_idx == 0 and epoch == 0:
                    print(f"\n--- DEBUG: First batch ---")
                    sample_ids = inputs["input_ids"][0].cpu().tolist()
                    img_count = sample_ids.count(image_token_id)
                    print(f"  Input length: {len(sample_ids)}")
                    print(f"  Image tokens: {img_count}")
                    print(f"  Pixel values shape: {inputs['pixel_values'].shape}")

                    # Show target text
                    target_start = (labels[0] != -100).nonzero(as_tuple=True)[0]
                    if len(target_start) > 0:
                        target_ids = labels[0, target_start[0]:].tolist()
                        target_ids = [t for t in target_ids if t != -100]
                        target_text = processor.tokenizer.decode(target_ids)
                        print(f"  Target text: {target_text[:200]}")
                    print(f"--- END DEBUG ---\n")

                # Forward pass
                outputs = model(**inputs)
                loss = outputs.loss / args.grad_accum

                if torch.isnan(loss):
                    print(f"  WARNING: NaN loss at batch {batch_idx}, skipping")
                    optimizer.zero_grad()
                    skipped += 1
                    continue

                loss.backward()

                # Optimizer step with gradient accumulation
                if (batch_idx + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item() * args.grad_accum
                epoch_steps += 1

                # Clear cache periodically
                if (batch_idx + 1) % args.grad_accum == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

            except Exception as e:
                if batch_idx < 3:
                    print(f"  Error batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                skipped += 1
                continue

            # Log progress (OUTSIDE try/except so logging errors don't skip batches)
            if batch_idx % 50 == 0 or batch_idx == total_batches - 1:
                avg_loss = epoch_loss / max(epoch_steps, 1)
                mem_info = ""
                try:
                    if torch.cuda.is_available():
                        mem_used = torch.cuda.memory_allocated() / 1024**3
                        mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                        mem_info = f", GPU: {mem_used:.1f}/{mem_total:.1f}GB"
                except Exception:
                    pass
                print(f"  Epoch {epoch+1}/{args.epochs} | "
                      f"Batch {batch_idx}/{total_batches} | "
                      f"Loss: {avg_loss:.4f}{mem_info}", flush=True)

        # Final optimizer step for remaining gradients
        if epoch_steps % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        # Epoch summary
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        epoch_time = time.time() - epoch_start
        print(f"\n  >>> Epoch {epoch+1}/{args.epochs} complete: "
              f"Loss={avg_epoch_loss:.4f} "
              f"Time={epoch_time:.1f}s "
              f"Steps={epoch_steps} "
              f"Skipped={skipped}")

        training_log.append({
            "epoch": epoch + 1,
            "avg_loss": round(avg_epoch_loss, 6),
            "time_seconds": round(epoch_time, 1),
            "steps": epoch_steps,
            "skipped": skipped
        })

        # Early stopping (start checking after warm-up epoch)
        if epoch >= no_early_stop_before_epoch:
            if avg_epoch_loss < best_loss - 0.001:
                best_loss = avg_epoch_loss
                patience_counter = 0
                # Save best adapter checkpoint
                os.makedirs(best_adapter_dir, exist_ok=True)
                model.save_pretrained(best_adapter_dir)
                print(f"  [ES] New best loss: {best_loss:.6f} — saved best adapter to {best_adapter_dir}")
            else:
                patience_counter += 1
                print(f"  [ES] No improvement (best={best_loss:.6f}). Patience: {patience_counter}/{early_stop_patience}")
                if patience_counter >= early_stop_patience:
                    print(f"  [EARLY STOP] Loss converged. Stopping after epoch {epoch+1}")
                    print(f"  [EARLY STOP] Best adapter saved at: {best_adapter_dir}")
                    break
        else:
            best_loss = min(best_loss, avg_epoch_loss)
            print(f"  [ES] Epoch {epoch+1} (warm-up, no early stopping). Loss: {avg_epoch_loss:.6f}")

    # =========================================================================
    # Step 4: Save adapter
    # =========================================================================
    print(f"\n[4/4] Saving retrained adapter to {args.output_dir}...")

    os.makedirs(args.output_dir, exist_ok=True)

    # Save adapter
    model.save_pretrained(args.output_dir)
    print(f"  Adapter saved to {args.output_dir}")

    # Save training log
    log = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "epochs": args.epochs,
            "lr": args.lr,
            "lora_r": args.lora_r,
            "max_normal": args.max_normal,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "total_samples": len(balanced_dataset),
            "abnormal_samples": len(abnormal_indices),
            "normal_samples": len(normal_indices),
            "pct_abnormal": round(pct_abnormal, 1)
        },
        "training_log": training_log
    }

    log_path = os.path.join(args.output_dir, "retraining_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  Training log saved to {log_path}")

    print("\n" + "=" * 60)
    print("  Detection Retraining Complete!")
    print("  Next step: Run evaluation:")
    print("  python -m src.eval.evaluate_all_stages --checkpoint checkpoints/production --stages detection --max_samples 50")
    print("=" * 60)


if __name__ == "__main__":
    main()

