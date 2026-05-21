import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"

print(f"Loading tokenizer from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

print(f"Loading model from {MODEL_PATH}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

print("Model loaded successfully!")
print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# 简单推理测试
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is in this image? (assume it's a cat)"}
]

text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
print(f"\nInput prompt:\n{text}\n")

# 生成
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Response:\n{response}")

print("\n=== Inference verified successfully! ===")