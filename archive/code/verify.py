"""
DP Comparison Script: Local DP (先加噪再聚合) vs Central DP (先聚合再加噪)

比较在相同隐私预算 (ε) 下，不同方案的性能表现：
1. Global LoRA - 全局训练的单一模型 (with DP)
2. Single Expert (Avg) - 遍历所有专家，每个加噪后取平均
3. Aggregated_A - 先加噪再聚合 (Local DP)
4. Aggregated_B - 先聚合再加噪 (Central DP)
"""

import os
import sys
import json
import torch
import numpy as np
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# === 环境变量 ===
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# ============ 配置 ============
BASE_MODEL = "/root/gpufree-data/hf/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
LORA_DIR = "/root/gpufree-data/lapped-lora/model/subset_8"
NUM_EXPERTS = 8

# 测试配置
TEST_PROMPT = "Question: Janet has 3 times as many marbles as Arnold. If Arnold has 12 marbles, how many marbles do they have together?\nAnswer:"
TARGET_TEXT = " 48"

# 隐私配置
EPSILONS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, float('inf')]  # inf 表示无隐私保护
DELTA = 1e-5
CLIP_THRESHOLD = 1.0  # 灵敏度上界 C

N_SAMPLES = 20  # 蒙特卡洛采样次数
TOP_P = 0.95  # Top-p 采样阈值

# 输出配置
OUTPUT_DIR = "results"


# ============ 隐私计算 ============
def compute_sigma(epsilon: float, delta: float, sensitivity: float) -> float:
    """
    高斯机制：给定 ε, δ, Δf，计算所需的噪声标准差 σ
    
    公式: σ = Δf × √(2 ln(1.25/δ)) / ε
    
    Args:
        epsilon: 隐私预算
        delta: 隐私参数 δ
        sensitivity: 灵敏度 Δf (通常等于 clip threshold)
    
    Returns:
        噪声标准差 σ
    """
    if epsilon == float('inf') or epsilon <= 0:
        return 0.0
    factor = np.sqrt(2 * np.log(1.25 / delta))
    return sensitivity * factor / epsilon


def epsilon_to_str(epsilon: float) -> str:
    """将 epsilon 转换为字符串表示"""
    if epsilon == float('inf'):
        return "inf"
    return f"{epsilon}"


# ============ 评估函数 ============
def is_in_top_p(logits_1d: torch.Tensor, target_id: int, top_p: float = TOP_P) -> bool:
    """
    检查目标 token 是否在 top-p 采样集合中
    
    Args:
        logits_1d: 1D logits tensor [vocab_size]
        target_id: 目标 token 的 ID
        top_p: 累积概率阈值 (默认 0.95)
    
    Returns:
        bool: 目标 token 是否在 top-p 集合中
    """
    # 计算 softmax 概率
    probs = torch.softmax(logits_1d, dim=-1)
    
    # 按概率降序排序
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    # 计算累积概率
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 找到累积概率首次超过 top_p 的位置
    # 包含该位置的所有 token 构成 top-p 集合
    cutoff_idx = (cumulative_probs >= top_p).nonzero(as_tuple=True)[0]
    if len(cutoff_idx) > 0:
        cutoff_idx = cutoff_idx[0].item() + 1  # +1 因为我们要包含这个位置
    else:
        cutoff_idx = len(sorted_indices)
    
    # 获取 top-p 集合中的 token IDs
    top_p_indices = sorted_indices[:cutoff_idx].tolist()
    
    return target_id in top_p_indices


def measure_metrics(logits: torch.Tensor, target_id: int, sigma: float, n_samples: int) -> dict:
    """
    对 logits 加噪并计算指标
    
    Args:
        logits: 模型输出的 logits [1, vocab_size]
        target_id: 目标 token 的 ID
        sigma: 噪声标准差
        n_samples: 蒙特卡洛采样次数
    
    Returns:
        dict: {logit, rank, top1, top_p}
    """
    logit_sum, rank_sum, top1_sum, top_p_sum = 0, 0, 0, 0
    clean_logits = logits.float()
    
    for _ in range(n_samples):
        if sigma > 0:
            noise = torch.randn_like(clean_logits) * sigma
            noisy_logits = clean_logits + noise
        else:
            noisy_logits = clean_logits
        
        target_logit = noisy_logits[0, target_id].item()
        logit_sum += target_logit
        
        # Rank: 1 表示最好
        rank = (noisy_logits[0] > target_logit).sum().item() + 1
        rank_sum += rank
        
        # Top-1 准确率
        top1_sum += int(noisy_logits[0].argmax().item() == target_id)
        
        # Top-p 准确率 (p=0.95)
        top_p_sum += int(is_in_top_p(noisy_logits[0], target_id, TOP_P))
    
    return {
        "logit": logit_sum / n_samples,
        "rank": rank_sum / n_samples,
        "top1": top1_sum / n_samples * 100,  # 转为百分比
        "top_p": top_p_sum / n_samples * 100
    }


def evaluate_aggregated_A(expert_logits_list: list, target_id: int, sigma: float, n_samples: int) -> dict:
    """
    方案 A: 先加噪再聚合 (Local DP)
    
    每个专家的 logits 先独立加噪，然后聚合（取平均）
    """
    logit_sum, rank_sum, top1_sum, top_p_sum = 0, 0, 0, 0
    num_experts = len(expert_logits_list)
    
    for _ in range(n_samples):
        # 每个专家加噪后聚合
        agg_logits = torch.zeros_like(expert_logits_list[0])
        for exp_logits in expert_logits_list:
            if sigma > 0:
                noise = torch.randn_like(exp_logits) * sigma
                agg_logits += (exp_logits + noise)
            else:
                agg_logits += exp_logits
        agg_logits /= num_experts
        
        # 计算指标
        target_logit = agg_logits[0, target_id].item()
        logit_sum += target_logit
        rank_sum += (agg_logits[0] > target_logit).sum().item() + 1
        top1_sum += int(agg_logits[0].argmax().item() == target_id)
        top_p_sum += int(is_in_top_p(agg_logits[0], target_id, TOP_P))
    
    return {
        "logit": logit_sum / n_samples,
        "rank": rank_sum / n_samples,
        "top1": top1_sum / n_samples * 100,
        "top_p": top_p_sum / n_samples * 100
    }


def evaluate_aggregated_B(expert_logits_list: list, target_id: int, sigma_B: float, n_samples: int) -> dict:
    """
    方案 B: 先聚合再加噪 (Central DP)
    
    先将所有专家的 logits 聚合（取平均），然后统一加噪
    灵敏度降低为 C/N，所以噪声也相应减小
    """
    # 先干净聚合
    clean_agg_logits = sum(expert_logits_list) / len(expert_logits_list)
    
    # 在聚合后的 logits 上加噪
    return measure_metrics(clean_agg_logits, target_id, sigma_B, n_samples)


# ============ 打印函数 ============
def print_header(epsilon: float, sigma: float, sigma_B: float):
    """打印表头"""
    eps_str = epsilon_to_str(epsilon)
    print(f"\n{'='*100}")
    print(f"Epsilon: {eps_str:<6} | Delta: {DELTA} | Clip: {CLIP_THRESHOLD}")
    print(f"Sigma_A (Local DP): {sigma:.4f} | Sigma_B (Central DP): {sigma_B:.4f}")
    print(f"{'='*100}")
    print(f"{'Model':<28} | {'Logit':<12} | {'Rank':<10} | {'Top-1':<10} | {'Top-p':<10}")
    print(f"{'-'*100}")


def print_metrics(name: str, metrics: dict):
    """打印单行指标"""
    print(f"{name:<28} | {metrics['logit']:<12.4f} | {metrics['rank']:<10.2f} | {metrics['top1']:<9.1f}% | {metrics['top_p']:<9.1f}%")


# ============ 主函数 ============
def main():
    print("="*100)
    print("DP Comparison: Local DP (先加噪再聚合) vs Central DP (先聚合再加噪)")
    print("="*100)
    
    # 1. 加载模型
    print("\n>>> [1/4] Loading Base Model (4bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map={"": 0},
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    # 准备输入
    inputs = tokenizer(TEST_PROMPT, return_tensors="pt").to(base_model.device)
    target_id = tokenizer.encode(TARGET_TEXT, add_special_tokens=False)[0]
    
    print(f">>> Prompt: {TEST_PROMPT[:50]}...")
    print(f">>> Target: '{TARGET_TEXT}' (ID: {target_id})")
    
    # 2. 预先计算所有专家的 logits（避免重复加载）
    print("\n>>> [2/4] Pre-computing expert logits...")
    expert_logits_list = []
    for i in range(NUM_EXPERTS):
        adapter_path = f"{LORA_DIR}/lora_{i}"
        try:
            base_model.load_adapter(adapter_path, adapter_name=f"expert_{i}")
            base_model.set_adapter(f"expert_{i}")
            with torch.no_grad():
                out = base_model(**inputs)
                expert_logits_list.append(out.logits[:, -1, :].float().clone())
            base_model.delete_adapter(f"expert_{i}")
            print(f"    Expert {i}: OK")
        except Exception as e:
            print(f"    Expert {i}: Failed ({e})")
    
    if len(expert_logits_list) == 0:
        print("!!! Error: No experts loaded. Exiting.")
        return
    
    print(f">>> Successfully loaded {len(expert_logits_list)} experts.")
    
    # 3. 预先计算 Global LoRA 的 logits
    print("\n>>> [3/4] Loading Global LoRA...")
    global_logits = None
    try:
        global_path = f"{LORA_DIR}/lora_global"
        base_model.load_adapter(global_path, adapter_name="global")
        base_model.set_adapter("global")
        with torch.no_grad():
            out = base_model(**inputs)
            global_logits = out.logits[:, -1, :].float().clone()
        base_model.delete_adapter("global")
        print("    Global LoRA: OK")
    except Exception as e:
        print(f"    Global LoRA: Not found or failed ({e})")
    
    # 4. 开始评估
    print("\n>>> [4/4] Starting evaluation...")
    
    results = {
        "config": {
            "epsilons": [epsilon_to_str(e) for e in EPSILONS],
            "delta": DELTA,
            "clip_threshold": CLIP_THRESHOLD,
            "num_experts": len(expert_logits_list),
            "n_samples": N_SAMPLES,
            "test_prompt": TEST_PROMPT,
            "target_text": TARGET_TEXT,
            "timestamp": datetime.now().isoformat()
        },
        "results": {}
    }
    
    for epsilon in EPSILONS:
        eps_str = epsilon_to_str(epsilon)
        
        # 计算噪声标准差
        sigma = compute_sigma(epsilon, DELTA, CLIP_THRESHOLD)
        sigma_B = compute_sigma(epsilon, DELTA, CLIP_THRESHOLD / len(expert_logits_list))
        
        # 打印表头
        print_header(epsilon, sigma, sigma_B)
        
        eps_results = {
            "sigma": sigma,
            "sigma_B": sigma_B
        }
        
        # === 方案 1: Global LoRA (with DP) ===
        if global_logits is not None:
            metrics = measure_metrics(global_logits, target_id, sigma, N_SAMPLES)
            eps_results["global"] = metrics
            print_metrics("Global LoRA", metrics)
        
        # === 方案 2: Single Expert Average (with DP) ===
        # 每个专家单独加噪，分别计算指标后取平均
        all_expert_metrics = []
        for exp_logits in expert_logits_list:
            m = measure_metrics(exp_logits, target_id, sigma, N_SAMPLES)
            all_expert_metrics.append(m)
        
        # 对各指标取平均
        avg_metrics = {
            "logit": np.mean([m["logit"] for m in all_expert_metrics]),
            "rank": np.mean([m["rank"] for m in all_expert_metrics]),
            "top1": np.mean([m["top1"] for m in all_expert_metrics]),
            "top_p": np.mean([m["top_p"] for m in all_expert_metrics])
        }
        eps_results["single_expert_avg"] = avg_metrics
        print_metrics("Single Expert (Avg)", avg_metrics)
        
        # === 方案 3: Aggregated_A - 先加噪再聚合 (Local DP) ===
        agg_A_metrics = evaluate_aggregated_A(expert_logits_list, target_id, sigma, N_SAMPLES)
        eps_results["aggregated_A"] = agg_A_metrics
        print_metrics("Aggregated_A (Local DP)", agg_A_metrics)
        
        # === 方案 4: Aggregated_B - 先聚合再加噪 (Central DP) ===
        agg_B_metrics = evaluate_aggregated_B(expert_logits_list, target_id, sigma_B, N_SAMPLES)
        eps_results["aggregated_B"] = agg_B_metrics
        print_metrics("Aggregated_B (Central DP)", agg_B_metrics)
        
        results["results"][eps_str] = eps_results
    
    # 5. 保存结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "dp_comparison.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*100}")
    print(f">>> Results saved to {output_path}")
    print(f"{'='*100}")
    
    # 6. 打印总结
    print("\n>>> Summary:")
    print("  - Local DP (Aggregated_A): 每个专家独立加噪后聚合")
    print("  - Central DP (Aggregated_B): 先聚合再加噪，需要可信聚合器")
    print(f"  - 噪声比例: Sigma_A / Sigma_B = {len(expert_logits_list)} (专家数量)")


if __name__ == "__main__":
    main()
