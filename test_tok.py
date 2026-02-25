import sys
sys.path.append('d:/MedGamma/src')
from transformers import AutoProcessor
p = AutoProcessor.from_pretrained('google/gemma-3-4b-it', local_files_only=False)
t = p.tokenizer
print("Tokenizer loaded. user\\n tokens:", t.encode("user\n", add_special_tokens=False))
prompt = "Analyze this chest X-ray"
full_text = f"<start_of_turn>user\n{prompt}\n<end_of_turn>\n<start_of_turn>model\n"
seq = t.encode(full_text)
print("Full text tokens:", seq[:15])
user_turn_ids = t.encode("user\n", add_special_tokens=False)
insert_pos_train = -1
for i in range(len(seq) - len(user_turn_ids) + 1):
    if seq[i:i+len(user_turn_ids)] == user_turn_ids:
        insert_pos_train = i + len(user_turn_ids)
        break
print("Train insert pos:", insert_pos_train)

prompt_clean = f"<start_of_turn>user\n{prompt}\n<end_of_turn>\n<start_of_turn>model\n"
seq_clean = t.encode(prompt_clean)
insert_pos_inf = -1
for i in range(len(seq_clean) - len(user_turn_ids) + 1):
    if seq_clean[i:i+len(user_turn_ids)] == user_turn_ids:
        insert_pos_inf = i + len(user_turn_ids)
        break
print("Inference insert pos:", insert_pos_inf)

