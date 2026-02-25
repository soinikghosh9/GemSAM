with open('src/medgemma_wrapper.py', 'r') as f:
    t = f.read()

import re
idx = t.find('def _setup_processor')
if idx != -1:
    good_t = t[:idx]
    
    # We also need to fix _extract_json which I corrupted
    extract_idx = good_t.find('def _extract_json')
    if extract_idx != -1:
        good_t = good_t[:extract_idx] + '''    def _extract_json(self, decoded: str) -> str:
        # basic extraction logic
        json_str = ""
        start = decoded.find('{')
        end = decoded.rfind('}')
        if start != -1 and end != -1 and end >= start:
            json_str = decoded[start:end+1]
        return json_str

'''
    
    with open('src/medgemma_wrapper.py', 'w') as f:
        f.write(good_t)
        # Write setup processor
        f.write('''    def _setup_processor(self, model_path: str, use_base_model: bool = False):
        print("Setting up processor and tokens for inference...")
        base_model = "google/gemma-3-4b-it" 
        try:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(model_path)
            # Patch processor if not default
            target_image_size = 448 # Default to 448
            if target_image_size:
                if hasattr(self.processor, "image_processor"):
                     self.processor.image_processor.size = {"height": target_image_size, "width": target_image_size}

            # Setup tokenizer
            self.tokenizer = self.processor.tokenizer

            # Monkey-patch image token validation
            def _check_special_mm_tokens(sequence): pass
            self.processor._check_special_mm_tokens = _check_special_mm_tokens

        except Exception as e:
            print(f"Error setting up processor: {e}")
            raise

        if use_base_model:
            print("  > Base Model Requested (Zero-Shot Mode).")
            self.model = base_model
        else:
            lora_path = "checkpoints/production/medgemma/final"
            if os.path.exists(lora_path):
                print(f"  > Loading MedGemma LoRA Adapter from {lora_path}...")
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(
                    base_model,
                    lora_path,
                    is_trainable=False
                )
                print("  > LoRA Adapter Loaded Successfully. Model in eval mode.")
            else:
                self.model = base_model
''')
print('Fixed!')
