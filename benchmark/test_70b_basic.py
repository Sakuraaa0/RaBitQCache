"""Minimal test: load 70B model on 2 GPUs and run a simple inference.
No monkey-patching, no sparse attention - just vanilla HuggingFace.
Usage: CUDA_VISIBLE_DEVICES=6,7 python test_70b_basic.py
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/mnt/user-ssd/your_user/rabitq/models/Llama-3.1-70B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

print("Loading model with device_map='auto'...")
model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto",
    attn_implementation="flash_attention_2",
)
model.eval()

print(f"Model device map (first 5 layers):")
for k, v in list(model.hf_device_map.items())[:10]:
    print(f"  {k}: {v}")

print("\nTest 1: Short generation...")
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
print(f"Output: {tokenizer.decode(output[0], skip_special_tokens=True)}")

print("\nTest 2: Longer prefill (1000 tokens)...")
long_text = "The quick brown fox jumps over the lazy dog. " * 100
inputs = tokenizer(long_text, return_tensors="pt", truncation=True, max_length=1000).to(model.device)
print(f"Input length: {inputs.input_ids.shape[1]} tokens")
with torch.no_grad():
    output = model(input_ids=inputs.input_ids, past_key_values=None, use_cache=True)
print(f"Prefill OK. Logits shape: {output.logits.shape}")

print("\nAll tests passed!")
