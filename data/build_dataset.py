"""
将 LLaVA-Instruct-150K + COCO Captions 合并为 Qwen2.5-VL 训练格式
输出：train.jsonl, val.jsonl, test.jsonl
"""
import json
import random
import os
from pathlib import Path

random.seed(42)

def load_llava_data(filepath):
    """加载 LLaVA 指令数据，提取图文对"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []
    for item in data:
        conversations = item.get('conversations', [])
        if len(conversations) < 2:
            continue

        # 提取 user 和 assistant 轮次
        user_msg = None
        assistant_msg = None
        for i, conv in enumerate(conversations):
            if conv['from'] == 'human' and user_msg is None:
                user_msg = conv['value']
            elif conv['from'] == 'gpt' and assistant_msg is None:
                assistant_msg = conv['value']

        if user_msg and assistant_msg:
            # LLaVA 数据中的图片 URL
            image_url = item.get('image', '')
            if image_url:
                samples.append({
                    'image_url': image_url,
                    'instruction': user_msg,
                    'output': assistant_msg,
                    'source': 'llava'
                })

    return samples


def load_coco_captions(caption_path, image_dir):
    """加载 COCO captions"""
    with open(caption_path, 'r') as f:
        annotations = json.load(f)

    # 构建 image_id -> image_file 的映射
    images = {img['id']: img['file_name'] for img in annotations['images']}

    samples = []
    for ann in annotations['annotations']:
        img_id = ann['image_id']
        if img_id not in images:
            continue

        image_file = images[img_id]
        caption = ann['caption']

        # COCO 图片 URL（使用 MS COCO 公开 URL）
        image_url = f"http://images.cocodataset.org/train2014/{image_file}"

        samples.append({
            'image_url': image_url,
            'instruction': '请描述这张图片的内容，并给出3个标签',
            'output': caption,
            'source': 'coco'
        })

    return samples


def convert_to_qwen_format(samples):
    """转换为 Qwen2.5-VL 训练格式"""
    converted = []
    for s in samples:
        message = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": s['image_url']},
                        {"type": "text", "text": s['instruction']}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": s['output']}
                    ]
                }
            ],
            "metadata": {
                "source": s['source'],
                "instruction": s['instruction'],
                "output": s['output']
            }
        }
        converted.append(message)
    return converted


def main():
    # 路径配置
    llava_json = "raw/LLaVA/llava_instruct_150k.json"  # 根据实际路径修改
    coco_annotations = "raw/coco/annotations/captions_train2014.json"  # 根据实际路径修改
    coco_image_dir = "raw/coco/train2014"

    output_dir = Path("processed")
    output_dir.mkdir(exist_ok=True)

    # 加载数据
    print("Loading LLaVA data...")
    llava_samples = load_llava_data(llava_json)
    print(f"  LLaVA samples: {len(llava_samples)}")

    print("Loading COCO captions...")
    coco_samples = load_coco_captions(coco_annotations, coco_image_dir)
    print(f"  COCO samples: {len(coco_samples)}")

    # 合并
    all_samples = llava_samples + coco_samples
    print(f"  Total: {len(all_samples)}")

    # 采样：LLaVA 取 1500 条，COCO 取 500 条
    llava_subset = random.sample(llava_samples, min(1500, len(llava_samples)))
    coco_subset = random.sample(coco_samples, min(500, len(coco_samples)))
    subset = llava_subset + coco_subset

    # 打乱
    random.shuffle(subset)

    # 划分：80% 训练，10% 验证，10% 测试
    n = len(subset)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    train_data = subset[:train_end]
    val_data = subset[train_end:val_end]
    test_data = subset[val_end:]

    # 转换格式
    train_qwen = convert_to_qwen_format(train_data)
    val_qwen = convert_to_qwen_format(val_data)
    test_qwen = convert_to_qwen_format(test_data)

    # 保存
    def save_jsonl(data, path):
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"  Saved {len(data)} samples to {path}")

    save_jsonl(train_qwen, output_dir / "train.jsonl")
    save_jsonl(val_qwen, output_dir / "val.jsonl")
    save_jsonl(test_qwen, output_dir / "test.jsonl")

    print("\n=== Dataset preparation complete! ===")
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")


if __name__ == "__main__":
    main()