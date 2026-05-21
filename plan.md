# 微博多模态理解与 ReAct Agent 系统 — 完整项目方案

> 目标岗位：微博大模型与多模态 AI 技术研发实习
> 候选人画像：了解大模型基础，无项目经验，Python/C++/PyTorch，2 周时间，2×A100 40GB，local-full-run
> 方案日期：2026-05-21

---

## 一、JD 拆解 vs 你的匹配度

| JD 要求 | 你的现状 | 匹配策略 |
|---------|---------|---------|
| 大模型训练/调优 | 了解基础 | 项目覆盖 LoRA 微调 + 推理部署 |
| 多模态模型应用 | 了解少 | 项目以多模态为核心，边做边学 |
| Agent 系统构建 | 了解少 | 项目包含 ReAct Agent + Tool Calling |
| Python 扎实 | ✅ | 直接用 Python 栈 |
| PyTorch | ✅ | 项目基于 PyTorch |
| 2 周时间 | 紧张 | 选已有模型微调，不从头训练 |
| 2×A100 40GB | 充裕 | 可以跑 Qwen2.5-7B LoRA 微调 |

---

## 二、推荐项目

**项目名称：** 基于 Qwen2.5-VL 的微博图文多模态理解与 ReAct Agent 系统

**一句话简历标题：** 基于 Qwen2.5-VL 的微博图文多模态理解与 ReAct Agent 系统

**核心思路：**
1. 用开源的 Qwen2.5-VL-7B（多模态大模型，支持图文理解）在模拟微博图文数据上做 LoRA 微调
2. 构建 ReAct Agent：模型通过 Reason → Act → Observe 循环自主决定调用工具
3. 多维度评估体系 + Ablation Study
4. Docker 一键部署 + Streamlit Web Demo

**为什么选这个：**
- 完美覆盖 JD 的 4 个核心关键词：大模型、多模态、Agent、AIGC
- Qwen2.5-VL 是开源可商用模型，GitHub 活跃，上手路径清晰
- LoRA 微调在单张 A100 40GB 上就能跑，2 周时间充裕
- ReAct Agent 是当前 Agent 主流范式，面试高频考点

---

## 三、项目架构全景

```
┌─────────────────────────────────────────────────┐
│         Streamlit Web Demo (port 8501)           │
│    上传图文 → 选择 Agent 模式 → 查看结果          │
├─────────────────────────────────────────────────┤
│              FastAPI Service (port 8000)          │
│         /chat /generate /health /docs             │
├─────────────────────────────────────────────────┤
│           ReAct Agent 调度器                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐         │
│  │Image     │ │Text      │ │Sentiment │         │
│  │Analyzer  │ │Generator │ │Analyzer  │         │
│  │(CLIP)    │ │(Qwen-VL) │ │(规则+LLM)│         │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘         │
│       └────────────┼────────────┘                 │
│                    ↓                              │
│         Qwen2.5-VL-7B (LoRA fine-tuned)          │
├─────────────────────────────────────────────────┤
│  评估模块: CLIPScore / ROUGE / BERTScore / 延迟  │
│  Ablation: LoRA rank 对比 / 数据量对比           │
├─────────────────────────────────────────────────┤
│  数据集: LLaVA-Instruct-150K + COCO Captions     │
└─────────────────────────────────────────────────┘
```

---

## 四、2 周详细执行计划

### Week 1：Baseline + 微调 + 理解核心链路

#### Day 1-2：环境搭建 + 数据准备

**环境搭建：**
- 在 A100 上创建 conda 环境
- 安装：transformers, peft, accelerate, bitsandbytes, vllm, unsloth, clip, sentence-transformers, fastapi, uvicorn, streamlit, docker
- 验证：能加载 Qwen2.5-VL-7B 并跑通一次推理

**数据准备（核心）：**
- 下载 LLaVA-Instruct-150K，抽取图文对
- 下载 COCO Captions，提取 caption 数据
- 合并去重，采样 2000 条作为训练集（1600 训练 + 200 验证 + 200 测试）
- 转换为 Qwen2.5-VL 的 training format：

```jsonl
{"messages": [
  {"role": "user", "content": [{"type": "image", "image": "url"}, {"type": "text", "text": "请描述这张图片的内容，并给出3个标签"}]},
  {"role": "assistant", "content": [{"type": "text", "text": "这是一张海滩日落照片，标签：海滩、日落、风景"}]}
]}
```

**产出：**
- `data/` 目录下训练/验证/测试三个 JSONL 文件
- `configs/env_setup.sh` 环境安装脚本
- 验证推理脚本 `scripts/test_inference.py`

---

#### Day 3-4：LoRA 微调

**微调配置：**
```yaml
# configs/lora_config.yaml
model_name: "Qwen/Qwen2.5-VL-7B-Instruct"
lora_rank: 64
lora_alpha: 128
lora_dropout: 0.05
target_modules: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
batch_size: 4
gradient_accumulation: 4
learning_rate: 2e-4
epochs: 3
max_seq_length: 2048
bf16: true
```

**训练流程：**
1. 用 unsloth 或 peft 加载模型（4-bit 量化节省显存）
2. 加载训练数据，做 tokenize
3. 启动训练，监控 loss 曲线和显存
4. 保存 LoRA adapter weights

**关键代码文件：**
- `train/run_lora.py` — 训练入口
- `train/data_loader.py` — 数据加载和预处理
- `train/callbacks.py` — 训练回调（日志、checkpoint 保存）

**产出：**
- LoRA checkpoint（约 100-200MB）
- 训练日志（loss 曲线、显存占用、耗时）

---

#### Day 5：Baseline 推理 + 初步验证

- 加载 LoRA 权重，在测试集上跑 inference
- 生成结果人工抽查 20 条
- 跑通 BLEU/ROUGE 基线指标
- 记录 baseline 结果，作为后续对比的基准

**产出：**
- `results/baseline_predictions.jsonl`
- `results/baseline_metrics.json`

---

### Week 2：三个增量模块

#### Day 6-8：ReAct Agent + Tool Calling

**Agent 架构设计：**

```python
# agent/agent.py — 核心 ReAct Agent
class MultimodalReActAgent:
    """
    ReAct Agent: 模型通过 Reason -> Act -> Observe 循环自主决策
    
    工具列表：
    - image_analyzer: 分析图片内容，提取关键实体和场景
    - text_generator: 生成文案（评论/摘要/标签）
    - sentiment_analyzer: 分析内容情感倾向
    - content_filter: 检测敏感/违规内容
    """
    
    def __init__(self, model, tools):
        self.model = model
        self.tools = tools
        self.max_steps = 5
    
    def run(self, user_input):
        """ReAct 循环: Reason -> Act -> Observe -> ... -> Final Answer"""
        thoughts = []
        for step in range(self.max_steps):
            # 1. Reason: 模型根据当前状态决定下一步
            thought = self.model.generate_reasoning(thoughts)
            
            # 2. Act: 如果决定调用工具，执行对应 tool
            if needs_tool_call(thought):
                tool_name, tool_args = parse_tool_call(thought)
                observation = self.tools[tool_name].call(**tool_args)
                thoughts.append({
                    "step": step,
                    "thought": thought,
                    "action": tool_name,
                    "observation": observation
                })
            else:
                # 3. Final Answer: 不需要工具，直接输出答案
                final_answer = self.model.generate_answer(thoughts)
                return {"thoughts": thoughts, "answer": final_answer}
        
        return {"thoughts": thoughts, "answer": self.model.generate_answer(thoughts)}
```

**四个工具的实现：**

| 工具 | 功能 | 实现方式 | 输入 | 输出 |
|------|------|---------|------|------|
| `image_analyzer` | 分析图片内容 | CLIP + Qwen-VL | 图片 URL | 场景、实体、关键信息列表 |
| `text_generator` | 生成文案 | Qwen-VL LoRA 微调模型 | 图片 + 用户指令 | 评论/摘要/标签文本 |
| `sentiment_analyzer` | 情感分析 | 规则 + 轻量 LLM | 文本 | 情感极性 + 置信度 |
| `content_filter` | 敏感内容检测 | 关键词匹配 + 规则 | 文本 | 是否违规 + 原因 |

**ReAct Prompt 设计：**
```
你是一个微博内容理解与生成助手。你可以使用以下工具：
- image_analyzer: 分析图片内容
- text_generator: 生成文案
- sentiment_analyzer: 分析情感
- content_filter: 检测敏感内容

使用格式：
思考：我需要分析这张图片的内容...
操作：image_analyzer
操作参数：{"image_url": "xxx"}
观察：[工具返回结果]
思考：根据图片分析结果，我需要生成一条评论...
操作：text_generator
操作参数：{"image_url": "xxx", "style": "friendly"}
观察：[工具返回结果]
思考：现在我需要检查内容是否合规...
操作：content_filter
操作参数：{"text": "生成的文案"}
观察：[工具返回结果]
回答：[最终回复]
```

**产出：**
- `agent/agent.py` — ReAct Agent 核心
- `agent/tools/` — 四个工具模块
- `agent/prompts.py` — ReAct prompt 模板
- `agent/test_agent.py` — Agent 测试脚本
- Agent 运行日志（展示 Reason-Act-Observe 循环）

---

#### Day 9-10：多维度评估 + Ablation Study

**评估指标体系：**

| 维度 | 指标 | 含义 | 工具 |
|------|------|------|------|
| 文本质量 | BLEU-4 | n-gram 精确匹配 | `nltk` |
| 文本质量 | ROUGE-L | 最长公共子序列 | `rouge-score` |
| 文本质量 | METEOR | 同义词+词形还原 | `nltk` |
| 图文一致性 | CLIPScore | 图文语义匹配度 | `open_clip_torch` |
| 语义相似度 | BERTScore | 基于预训练语义 | `sentence-transformers` |
| 效率 | 首 token 延迟 | 第一个 token 生成时间 | 计时 |
| 效率 | 端到端延迟 | 完整生成耗时 | 计时 |
| 效率 | 吞吐量 | 每秒生成 token 数 | 计时 |

**Ablation 实验设计：**

```
实验 1: LoRA rank 对比
  - rank=16, rank=64, rank=128
  - 对比指标：ROUGE-L, CLIPScore, 训练时间, 显存占用

实验 2: 数据量对比
  - 500条, 1000条, 2000条
  - 对比指标：ROUGE-L, CLIPScore, BLEU-4

实验 3: 模型对比
  - Qwen2.5-VL-7B (多模态, LoRA)
  - Qwen2.5-7B (纯文本, LoRA)
  - Qwen2.5-VL-7B (多模态, 零样本 prompt)
  - 对比指标：所有上述指标
```

**产出：**
- `eval/evaluate.py` — 评估脚本
- `eval/ablation.py` — Ablation 实验脚本
- `results/ablation_table.md` — 对比表格
- `results/charts/` — 可视化图表（折线图、柱状图）

---

#### Day 11-12：Docker 部署 + Web Demo

**Docker 结构：**
```
docker-compose.yml
├── Dockerfile.api      # FastAPI 服务
├── Dockerfile.streamlit # Streamlit Web UI
├── .env.example        # 环境变量模板
└── scripts/
    ├── start.sh        # 一键启动
    └── health_check.sh # 健康检查
```

**docker-compose.yml：**
```yaml
version: '3.8'
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    environment:
      - MODEL_PATH=/app/models/qwen2.5-vl-lora
      - LOG_LEVEL=info
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
  
  web:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    ports:
      - "8501:8501"
    depends_on:
      - api
    environment:
      - API_URL=http://api:8000
```

**Streamlit Web Demo 功能：**
- 上传图片 + 输入文字
- 选择 Agent 模式（自动工具调用 / 直接生成）
- 查看 ReAct 推理过程（Thought → Action → Observation）
- 查看评估指标和生成结果对比

**产出：**
- `Dockerfile.api`, `Dockerfile.streamlit`
- `docker-compose.yml`
- `app/web_demo.py` — Streamlit 前端
- `app/main.py` — FastAPI 入口
- `scripts/start.sh` — 一键启动脚本
- 启动文档 `docs/DEPLOYMENT.md`

---

#### Day 13-14：面试材料 + 查漏补缺

**面试材料清单：**
- STAR 简历版本（4-5 行）
- 核心代码讲解稿（按 8 步讲解法）
- 面试官追问 Q&A（15 个高频问题）
- PPT 提示词（8-10 页结构）
- 项目 README（GitHub 标准格式）

**查漏补缺：**
- 所有脚本能否一键跑通
- 日志是否完整、可追溯
- 代码注释和文档是否清晰

---

## 五、最终文件结构

```
project/
├── data/
│   ├── build_dataset.py          # 数据构建
│   ├── train.jsonl               # 训练数据
│   ├── val.jsonl                 # 验证数据
│   └── test.jsonl                # 测试数据
├── configs/
│   ├── lora_config.yaml          # LoRA 配置
│   └── env_setup.sh              # 环境安装
├── train/
│   ├── run_lora.py               # 训练入口
│   ├── data_loader.py            # 数据加载
│   └── callbacks.py              # 训练回调
├── agent/
│   ├── agent.py                  # ReAct Agent 核心
│   ├── tools/
│   │   ├── image_analyzer.py     # 图像分析工具
│   │   ├── text_generator.py     # 文案生成工具
│   │   ├── sentiment_analyzer.py # 情感分析工具
│   │   └── content_filter.py     # 内容过滤工具
│   ├── prompts.py                # ReAct prompt 模板
│   └── test_agent.py             # Agent 测试
├── eval/
│   ├── evaluate.py               # 评估脚本
│   └── ablation.py               # Ablation 实验
├── app/
│   ├── main.py                   # FastAPI 入口
│   └── web_demo.py               # Streamlit Web UI
├── results/
│   ├── baseline_predictions.jsonl
│   ├── baseline_metrics.json
│   ├── ablation_table.md
│   └── charts/
├── scripts/
│   ├── start.sh                  # 一键启动
│   └── health_check.sh
├── docs/
│   └── DEPLOYMENT.md
├── Dockerfile.api
├── Dockerfile.streamlit
├── docker-compose.yml
├── .env.example
├── requirements.txt
└── README.md
```

---

## 六、STAR 简历版本

**基于 Qwen2.5-VL 的微博图文多模态理解与 ReAct Agent 系统**

- 针对微博社交场景的图文理解与生成需求，选取 Qwen2.5-VL-7B 在 2000 条多模态指令数据上完成 LoRA 微调（rank=64），实现图文联合理解与文案生成，覆盖 BLEU-4/ROUGE-L/CLIPScore 多维度评估
- 设计并实现 ReAct Agent 系统，包含图像分析、文案生成、情感分析、内容过滤四个工具模块，模型通过 Reason-Act-Observe 循环自主决策调用路径，支持单条/批量推理
- 完成三组 ablation 实验（LoRA rank 对比、数据量对比、多模态 vs 纯文本对比），定位最优配置并输出实验报告
- 通过 FastAPI + Streamlit 构建可演示 Web 服务，使用 Docker Compose 实现一键部署，完整覆盖大模型调优、多模态 Agent、工程化部署三个 JD 核心方向

---

## 七、面试官追问 Q&A

### 基础问题

**Q1：为什么选 Qwen2.5-VL 而不是 LLaVA 或 BLIP？**
> Qwen2.5-VL 在中文场景表现更好，对中文图文的理解能力更强，且开源社区活跃、文档完善。微博场景以中文内容为主，所以选择了这个模型。

**Q2：LoRA 微调的具体配置和为什么这样选？**
> rank=64, alpha=128, dropout=0.05。rank 选择 64 是在参数效率和效果之间的平衡——太小（如 8/16）学习能力不足，太大（如 128+）容易过拟合且显存占用高。alpha=128 是 rank 的 2 倍，这是 LoRA 的常见比例。dropout 用 0.05 防止过拟合。

**Q3：你的 Agent 系统和一个简单的 API 调用有什么区别？**
> 核心区别在于多步骤推理和模块化解耦。Agent pipeline 包含理解→生成→过滤三个独立模块，每个模块可以独立替换和优化。比如 QualityFilter 可以后续接入大模型做二次校验，ContentGenerator 可以接多个生成策略。而简单 API 是单点调用，没有这种可扩展性。

### Agent 相关

**Q4：ReAct Agent 和传统 pipeline 有什么区别？**
> Pipeline 是固定的步骤顺序，每个环节必须执行。ReAct Agent 让模型根据当前状态自主决定下一步——可能调用工具，可能直接回答，可能循环多次。比如用户问"这张图里有什么，帮我写一条朋友圈文案"，Agent 会先调用 image_analyzer 理解图片，再调用 text_generator 生成文案，而不是不管用户需求直接生成。

**Q5：工具调用出错了怎么办？**
> 设计了两层容错：1）工具层有 try-except，返回结构化错误信息而不是崩溃；2）Agent 层在 Observe 阶段收到错误后，模型可以决定重试、换工具、或者跳过该步骤。这体现了 Agent 的鲁棒性。

**Q6：怎么保证 Agent 不会无限循环调用工具？**
> 设置了 max_steps=5 的上限，超过后强制输出最终答案。同时每个 tool call 都有明确的输入输出 schema，避免模型调用参数错误。

### 多模态相关

**Q7：多模态对齐是怎么处理的？**
> 在微调阶段，模型通过图文对的联合训练学习对齐关系。具体来说，图像通过视觉编码器转换为视觉 token，和文本 token 一起输入模型。在数据层面，我确保每条微博图文都有准确的文本标注（标签、摘要），帮助模型学习正确的图文对应关系。

**Q8：如果线上部署，怎么考虑延迟和成本？**
> 当前阶段使用 vLLM 做推理加速，支持 PagedAttention 和连续批处理。后续可以考虑：1）量化到 INT8/INT4 降低显存和延迟；2）对高频请求做缓存；3）根据请求类型路由到不同大小的模型。

### 实验设计

**Q9：你的 ablation 实验发现了什么？**
> （根据实际实验结果回答）例如：rank=64 在 ROUGE-L 上比 rank=16 高 3.2%，但比 rank=128 低 0.8%。考虑到训练时间和显存，rank=64 是性价比最高的选择。数据量方面，1000 条到 2000 条的提升幅度开始递减，说明 2000 条已经接近饱和。

**Q10：为什么不做从头训练，只做 LoRA？**
> 从头训练 7B 模型需要大量数据和算力（数百张 GPU 卡，数周时间）。LoRA 只训练约 1% 的参数，在单张 A100 上 1-2 天就能完成，适合实习项目的时间约束。而且 LoRA 在下游任务上的效果通常接近全量微调。

### 不足与展望

**Q11：你的项目最大的限制是什么？下一步怎么做？**
> 最大限制是数据量较小（2000 条），主要是公开数据集的采样。下一步计划：1）接入真实微博数据（通过 API 或合规爬虫）；2）加入 DPO（直接偏好优化）进一步提升生成质量；3）扩展多模态能力，支持视频理解。

**Q12：多模态只覆盖了"图文"，JD 还提到了视频、语音**
> 当前版本聚焦图文理解，后续计划扩展视频帧序列理解和语音-文本对齐。调研了 Video-LLaMA 和 Qwen-Audio 的架构思路，视频理解的核心挑战在于时序建模和计算效率。

---

## 八、核心技术链路（面试时能讲清）

```
输入：微博图文 (image + text)
    ↓
MultimodalParser → 解析输入，提取图像和文本
    ↓
ReAct Agent 调度
    ↓
Qwen2.5-VL-7B (LoRA fine-tuned)
    ├── 视觉编码器：处理图像 → 视觉 token
    ├── 文本编码器：处理文本 → 文本 token
    └── 多模态融合：联合生成理解结果
    ↓
工具调用（按需）：
    ├── image_analyzer → 场景/实体识别
    ├── text_generator → 文案生成
    ├── sentiment_analyzer → 情感分析
    └── content_filter → 合规检测
    ↓
输出：结构化内容（标签、摘要、生成文案、情感）
```

**讲解顺序（8 步法）：**
1. 入口命令：`docker-compose up` 或 `python train/run_lora.py`
2. 配置/参数：LoRA 配置、模型路径、数据路径
3. 核心输入输出：HTTP request / dataset / tensor shape
4. 核心模块：agent/eval/train 三大模块职责
5. 关键状态变化：数据流 → 模型推理 → 工具调用 → 输出
6. 测试/评估：多维度指标 + ablation
7. 我的改动：Agent 设计、评估体系、Docker 部署
8. 一个失败 case：比如 LoRA rank=128 过拟合，通过早停解决

---

## 九、PPT 提示词（8-10 页）

1. **封面**：项目名称、个人信息、目标岗位
2. **项目背景**：为什么做这个、JD 匹配点
3. **项目来源**：Qwen2.5-VL 开源项目 + LLaVA/COCO 数据集
4. **架构图**：请求流图 + 数据流图
5. **Baseline**：训练命令、smoke test 结果、初始指标
6. **我的修改**：ReAct Agent + 评估体系 + Docker 部署
7. **实验结果**：ablation 对比表 + 图表
8. **限制与后续**：数据量、多模态扩展、DPO/RLHF
9. **演示**：Web Demo 截图 / 录屏
10. **Q&A**

---

## 十、运行深度检查清单

- [ ] 环境搭建完成，Qwen2.5-VL 推理验证通过
- [ ] 数据集下载、清洗、格式转换完成
- [ ] LoRA 微调完成，checkpoint 保存
- [ ] Baseline 推理 + 指标计算完成
- [ ] ReAct Agent 实现 + 测试通过
- [ ] 多维度评估体系实现 + 实验完成
- [ ] Docker 部署 + Web Demo 跑通
- [ ] 面试材料准备完成
- [ ] 所有脚本可一键复现
