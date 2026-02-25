with open('src/medgemma_wrapper.py', 'r') as f:
    text = f.read()
text = text.replace('"input_ids": text_inputs["input_ids"],', '"input_ids": input_ids,')
text = text.replace('"attention_mask": text_inputs["attention_mask"],', '"attention_mask": attention_mask,')
with open('src/medgemma_wrapper.py', 'w') as f:
    f.write(text)
print('Patched successfully!')
