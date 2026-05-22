## Day 1：环境搭建 + 推理验证

### 1.1 创建 conda 环境

# 创建环境（Python 3.10 兼容性最好）
conda create -n multimodal-agent python=3.10 -y
conda activate multimodal-agent

### 1.2 安装核心依赖

# PyTorch（选和你服务器 CUDA 版本匹配的）
# 先查 CUDA 版本：nvidia-smi
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 大模型核心库
pip install transformers==5.5.0
pip install peft==0.13.2
pip install accelerate==1.2.0
pip install bitsandbytes==0.44.1
pip install sentencepiece
pip install protobuf

# LoRA 微调
pip install unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git

# 推理加速
pip install vllm==0.21.0

# 评估工具
pip install clip
pip install open-clip-torch
pip install nltk
pip install rouge-score
pip install sentence-transformers

# 服务部署
pip install fastapi uvicorn streamlit python-multipart

# 其他
pip install pyyaml requests tqdm

### 1.3 验证安装

# 检查 CUDA 是否可用
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
# 应该输出：CUDA: True, Device: NVIDIA A100-SXM4-40GB

# 检查关键库
python -c "import transformers, peft, accelerate, bitsandbytes, vllm; print('All imports OK')"

### 1.4 跑通一次推理（验证模型能加载）

创建 scripts/test_inference.py：

"""验证 Qwen2.5-VL-7B 能正常加载并推理"""
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

运行：

python scripts/test_inference.py

预期结果：

- 模型从 HuggingFace 下载到本地缓存（约 14GB，首次需要时间）
- 显存占用约 14-16GB（bf16 精度）
- 能正常输出推理结果

如果遇到问题：

- 下载慢：配置 HuggingFace 镜像 export HF_ENDPOINT=https://hf-mirror.com（Linux）或在代码中加 cache_dir="./hf_cache"
- 显存不够：加 load_in_4bit=True 或 load_in_8bit=True 到 from_pretrained