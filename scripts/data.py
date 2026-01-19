import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from datasets import load_dataset


def _map_gsm8k() -> List[Dict[str, str]]:
    print("Loading GSM8K...")
    try:
        ds = load_dataset("gsm8k", "main", split="train")
        return [
            {
                "instruction": sample["question"].strip(),
                "output": sample["answer"].strip(),
                "source": "gsm8k",
            }
            for sample in ds
        ]
    except Exception as e:
        print(f"Error loading GSM8K: {e}")
        return []


def _map_commonsense_qa() -> List[Dict[str, str]]:
    print("Loading CommonsenseQA...")
    try:
        # 官方 test split 没有答案，这里使用 train + validation
        train = load_dataset("commonsense_qa", split="train")
        val = load_dataset("commonsense_qa", split="validation")
        merged = list(train) + list(val)

        mapped = []
        for sample in merged:
            choices = sample.get("choices", {})
            labels = choices.get("label", [])
            texts = choices.get("text", [])
            options = [f"{lbl}. {txt}" for lbl, txt in zip(labels, texts)]
            option_block = "\n".join(options)
            answer_key = sample.get("answerKey")
            answer_text = ""
            if answer_key in labels:
                idx = labels.index(answer_key)
                answer_text = texts[idx]

            instruction = f"{sample['question'].strip()}\nOptions:\n{option_block.strip()}".strip()
            mapped.append(
                {
                    "instruction": instruction,
                    "output": answer_text.strip() if answer_text else answer_key,
                    "source": "commonsense_qa",
                }
            )
        return mapped
    except Exception as e:
        print(f"Error loading CommonsenseQA: {e}")
        return []


def _map_strategy_qa() -> List[Dict[str, str]]:
    # 优先尝试 ChilleD/StrategyQA (Parquet格式，更稳定)，其次是 wics
    candidates = [
        ("ChilleD/StrategyQA", None),    # 推荐：Parquet 格式，无需脚本
        ("wics/strategy-qa", None),      # 官方：可能需要 trust_remote_code
        ("tasksource/strategy-qa", None),
    ]

    ds = None
    last_err = None
    print("Loading StrategyQA...")
    
    for name, cfg in candidates:
        try:
            print(f"  Trying {name}...")
            # trust_remote_code=True 对某些含脚本的数据集是必须的
            ds = load_dataset(name, cfg, split="train", trust_remote_code=True)
            print(f"  Success: Loaded StrategyQA from {name}")
            break
        except Exception as exc:
            last_err = exc
            print(f"  Failed {name}: {str(exc)[:100]}...")
            continue

    if ds is None:
        print("Warning: StrategyQA not loaded; proceeding without it.")
        return []

    mapped = []
    for sample in ds:
        # 不同源的字段可能不同，做一下兼容
        question = sample.get("question", "")
        answer = sample.get("answer", None)
        facts = sample.get("facts", [])
        
        # 兼容 ChilleD 格式 (answer 是 boolean)
        if isinstance(answer, bool):
            answer_str = "Yes" if answer else "No"
        else:
            answer_str = str(answer)

        context = "\n".join(facts).strip() if isinstance(facts, list) else str(facts)
        instruction = question.strip()
        # StrategyQA 有时 context 是空的或者不重要，为了保持 prompt 简洁，可以选填
        # 这里如果不为空，我们加进去
        if context:
             instruction += f"\nContext: {context}"

        mapped.append(
            {
                "instruction": instruction,
                "output": answer_str,
                "source": "strategy_qa",
            }
        )
    return mapped


def _upsample_list(data: List[Dict], target_size: int, seed: int) -> List[Dict]:
    """上采样数据列表以达到目标大小"""
    if not data:
        return []
    
    rng = random.Random(seed)
    current_len = len(data)
    if current_len >= target_size:
        # 如果已经够大，截断到 target_size 以保持平衡
        return rng.sample(data, target_size)
    
    # 需要上采样
    full_repeats = target_size // current_len
    upsampled = data * full_repeats
    # 补齐剩余部分
    remainder = target_size % current_len
    upsampled += rng.sample(data, remainder)
    
    rng.shuffle(upsampled)
    return upsampled


def _stratified_dirichlet_split(
    data: List[Dict], 
    num_experts: int, 
    alpha: float, 
    seed: int,
    min_samples_per_expert: int = 50
) -> List[List[Dict]]:
    """
    对单个任务的数据进行 Dirichlet 划分。
    引入重试机制，防止某个 Expert 分到的数据过少。
    """
    if not data:
        return [[] for _ in range(num_experts)]

    # 使用 seed 初始化，但如果分布太差，我们会改变 seed 重试
    current_seed = seed
    best_counts = None
    best_slices = None
    
    # 最多尝试 20 次，避免死循环
    for attempt in range(20):
        rng = np.random.default_rng(current_seed)
        proportions = rng.dirichlet(np.repeat(alpha, num_experts))
        
        # 计算每个 Expert 分到的数量
        counts = (proportions * len(data)).astype(int)
        # 修正舍入误差
        diff = len(data) - counts.sum()
        counts[0] += diff
        
        # 检查是否满足最小样本要求
        if counts.min() >= min_samples_per_expert:
            best_counts = counts
            break
        
        # 如果不满足，打印一下并重试
        # print(f"    [Retry {attempt}] Min count {counts.min()} < {min_samples_per_expert}, reshuffling...")
        current_seed += 1
        
        # 如果是最后一次尝试，就只好接受了
        if attempt == 19:
            best_counts = counts
            print(f"    Warning: Could not satisfy min_samples requirement after 20 attempts. Min count: {counts.min()}")

    print(f"    Final Task Split Dist: {best_counts.tolist()}")

    slices = []
    offset = 0
    for count in best_counts:
        slices.append(data[offset : offset + count])
        offset += count
    return slices


def prepare_data(
    output_dir: Path,
    num_experts: int = 8,
    alpha: float = 0.5,
    min_size: int = 50, # 提高最小阈值
    train_ratio: float = 0.8,
    router_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    # 1. 分别加载数据
    data_gsm = _map_gsm8k()
    data_cqa = _map_commonsense_qa()
    data_sqa = _map_strategy_qa()

    # 2. 计算目标大小并上采样 (Auto-Balancing)
    lens = [len(d) for d in [data_gsm, data_cqa, data_sqa] if d]
    if not lens:
        raise RuntimeError("No data loaded!")
        
    target_size = max(lens)
    print(f"Original Sizes -> GSM: {len(data_gsm)}, CQA: {len(data_cqa)}, StrategyQA: {len(data_sqa)}")
    print(f"Target Balancing Size: {target_size}")

    # 使用不同的 seed 上采样，增加随机性
    data_gsm = _upsample_list(data_gsm, target_size, seed)
    data_cqa = _upsample_list(data_cqa, target_size, seed + 1)
    data_sqa = _upsample_list(data_sqa, target_size, seed + 2)

    # 3. 初始化全局容器
    global_router_train = []
    global_test = []
    expert_datasets = [[] for _ in range(num_experts)]

    # 4. 对每个任务独立进行切分和分配 (Stratified Split)
    task_idx = 0
    for task_name, task_data in [("GSM8K", data_gsm), ("CommonsenseQA", data_cqa), ("StrategyQA", data_sqa)]:
        if not task_data:
            continue
            
        print(f"\nProcessing Task: {task_name}")
        
        # 4.1 Split Train/Router/Test
        n = len(task_data)
        n_train = int(n * train_ratio)
        n_router = int(n * router_ratio)
        
        task_train = task_data[:n_train]
        task_router = task_data[n_train : n_train + n_router]
        task_test = task_data[n_train + n_router :]

        global_router_train.extend(task_router)
        global_test.extend(task_test)

        # 4.2 对 Train Set 进行 Dirichlet 划分
        # [CRITICAL FIX] 使用 seed + task_idx 确保每个任务的分布形状不同！
        task_expert_splits = _stratified_dirichlet_split(
            task_train, 
            num_experts, 
            alpha, 
            seed=seed + task_idx * 100, # 加上偏移量
            min_samples_per_expert=min_size
        )
        task_idx += 1
        
        # 4.3 分发给 Expert
        for i in range(num_experts):
            expert_datasets[i].extend(task_expert_splits[i])

    # 5. 打乱并保存
    print("\nSaving datasets...")
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    # 保存 Experts
    experts_dir = output_dir / "experts"
    experts_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, dataset in enumerate(expert_datasets):
        rng.shuffle(dataset) 
        path = experts_dir / f"expert_{idx}.json"
        with path.open("w") as f:
            json.dump(dataset, f, indent=2)
        print(f"  Expert {idx}: {len(dataset)} samples saved.")

    # 保存 Router Train
    rng.shuffle(global_router_train)
    with (output_dir / "router_train.json").open("w") as f:
        json.dump(global_router_train, f, indent=2)
    print(f"  Router Train: {len(global_router_train)} samples saved.")

    # 保存 Test
    rng.shuffle(global_test)
    with (output_dir / "test.json").open("w") as f:
        json.dump(global_test, f, indent=2)
    print(f"  Test Set: {len(global_test)} samples saved.")

    print(f"\nAll data prepared in {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Stratified & Balanced data.")
    parser.add_argument("--output-dir", type=Path, default=Path("data"), help="Directory to store processed JSON files.")
    parser.add_argument("--num-experts", type=int, default=8, help="Number of experts.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha.")
    parser.add_argument("--min-size", type=int, default=100, help="Min samples per expert per task.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Fraction of data for expert training.")
    parser.add_argument("--router-ratio", type=float, default=0.1, help="Fraction of data for router training.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_data(
        output_dir=args.output_dir,
        num_experts=args.num_experts,
        alpha=args.alpha,
        min_size=args.min_size,
        train_ratio=args.train_ratio,
        router_ratio=args.router_ratio,
        seed=args.seed,
    )