"""
DP Comparison Benchmark Script
比较在相同隐私预算 (ε) 下，不同方案的性能表现：

1. Single Expert - 只使用 expert_0 进行评估
2. Aggregated_A - 先加噪再聚合 (Local DP)
3. Aggregated_B - 先聚合再加噪 (Central DP) - 使用 CoMixModel Router

隐私预算计算：
- 使用 Rényi DP (RDP) 组合定理精确计算多次查询的总隐私预算
- 每生成一个 token 视为一次独立查询
- 给定目标总隐私预算 (ε, δ) 和最大 token 数，反推每次查询所需噪声

测试集: data/test.json 前100条
- gsm8k: 数学题 (数字提取匹配)
- commonsense_qa: 常识选择题 (完整答案匹配)
- strategy_qa: Yes/No 推理题 (完整答案匹配)
"""

import os
import sys
import re
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# RDP 隐私计算
from dp_accounting.rdp import rdp_privacy_accountant
from dp_accounting import dp_event

# === 添加当前目录到路径 ===
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from custom_router import CoMixModel

# === 环境变量 ===
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# ============================================================================
#                              配置部分
# ============================================================================

# 模型路径配置
BASE_MODEL = "/root/gpufree-data/hf/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
LORA_DIR = "/root/gpufree-data/OverlappedLoRA/model"
EXPERT_DIR = "/root/gpufree-data/OverlappedLoRA/model"  # CoMixModel 专家目录
ROUTER_CHECKPOINT = "/root/gpufree-data/OverlappedLoRA/model/router/routers.pt"

NUM_EXPERTS = 8

# 测试配置
TEST_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "test.json")
N_TEST_SAMPLES = 100  # 测试样本数量
MAX_NEW_TOKENS = 128  # 生成最大 token 数 (也是最大查询次数)

# 隐私配置
EPSILONS = [0.1, 1.0, 5.0, 10.0, float('inf')]  # inf 表示无隐私保护
DELTA = 1e-5
CLIP_THRESHOLD = 1.0  # 灵敏度上界 C

# 输出配置
OUTPUT_DIR = "results"


# ============================================================================
#                          隐私计算函数 (RDP 组合)
# ============================================================================

def compute_epsilon_for_sigma(sigma: float, sensitivity: float, num_queries: int, delta: float) -> float:
    """
    给定 sigma 和查询次数，使用 RDP 组合计算总 epsilon
    
    Args:
        sigma: 高斯噪声标准差
        sensitivity: 查询灵敏度 (Δf)
        num_queries: 查询次数 (token 数量)
        delta: 隐私参数 δ
    
    Returns:
        总隐私预算 ε
    """
    if sigma <= 0:
        return float('inf')
    
    accountant = rdp_privacy_accountant.RdpAccountant()
    noise_multiplier = sigma / sensitivity
    
    event = dp_event.GaussianDpEvent(noise_multiplier=noise_multiplier)
    composed_event = dp_event.SelfComposedDpEvent(event, count=num_queries)
    
    accountant.compose(composed_event)
    return accountant.get_epsilon(delta)


def compute_sigma_with_rdp(target_epsilon: float, delta: float, sensitivity: float, num_queries: int) -> float:
    """
    使用 RDP 组合定理计算所需的噪声标准差
    
    给定目标总隐私预算 (ε, δ) 和查询次数，使用二分搜索反推所需的 sigma
    
    Args:
        target_epsilon: 目标总隐私预算 ε
        delta: 隐私参数 δ
        sensitivity: 单次查询灵敏度 (Δf)
        num_queries: 总查询次数 (最大 token 数)
    
    Returns:
        所需的噪声标准差 σ
    """
    if target_epsilon == float('inf') or target_epsilon <= 0:
        return 0.0
    
    # 二分搜索范围
    sigma_low, sigma_high = 0.01, 10000.0
    
    # 先检查边界：sigma_high 是否足够大
    eps_high = compute_epsilon_for_sigma(sigma_high, sensitivity, num_queries, delta)
    if eps_high > target_epsilon:
        print(f"Warning: Even sigma={sigma_high} cannot achieve eps={target_epsilon} with {num_queries} queries")
        return sigma_high
    
    # 二分搜索找到满足条件的最小 sigma
    for _ in range(100):  # 最多 100 次迭代
        sigma_mid = (sigma_low + sigma_high) / 2
        eps_mid = compute_epsilon_for_sigma(sigma_mid, sensitivity, num_queries, delta)
        
        if abs(eps_mid - target_epsilon) < 0.001:  # 精度足够
            return sigma_mid
        
        if eps_mid > target_epsilon:
            # epsilon 太大，需要更大的 sigma (更多噪声)
            sigma_low = sigma_mid
        else:
            # epsilon 太小，可以减小 sigma
            sigma_high = sigma_mid
    
    return sigma_mid


def compute_sigma(epsilon: float, delta: float, sensitivity: float, num_queries: int = 1) -> float:
    """
    计算高斯机制所需的噪声标准差 (支持 RDP 组合)
    
    Args:
        epsilon: 目标隐私预算 ε
        delta: 隐私参数 δ
        sensitivity: 查询灵敏度 Δf
        num_queries: 查询次数 (默认 1，即单次查询)
    
    Returns:
        噪声标准差 σ
    """
    if num_queries <= 1:
        # 单次查询：使用经典公式 σ = Δf × √(2 ln(1.25/δ)) / ε
        if epsilon == float('inf') or epsilon <= 0:
            return 0.0
        factor = np.sqrt(2 * np.log(1.25 / delta))
        return sensitivity * factor / epsilon
    else:
        # 多次查询：使用 RDP 组合
        return compute_sigma_with_rdp(epsilon, delta, sensitivity, num_queries)


def epsilon_to_str(epsilon: float) -> str:
    """将 epsilon 转换为字符串表示"""
    if epsilon == float('inf'):
        return "inf"
    return f"{epsilon}"


# ============================================================================
#                         数据加载与评估模块
# ============================================================================

def load_test_data(path: str, n: int = 100) -> list:
    """
    加载测试数据，取前 n 条
    返回: [{"instruction": ..., "output": ..., "source": ...}, ...]
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data[:n]


def extract_gsm8k_answer(output: str) -> float:
    """
    从 gsm8k 输出中提取最终答案
    格式: "... #### 1600" -> 1600
    """
    # 尝试匹配 #### 后的数字
    match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', output)
    if match:
        return float(match.group(1).replace(',', ''))
    
    # Fallback: 提取最后一个数字
    numbers = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', output)
    if numbers:
        return float(numbers[-1].replace(',', ''))
    return None


def evaluate_answer(pred: str, gold: str, source: str) -> bool:
    """
    根据数据源评估答案正确性
    
    Args:
        pred: 模型预测的答案
        gold: 标准答案
        source: 数据来源 (gsm8k, commonsense_qa, strategy_qa)
    
    Returns:
        bool: 是否正确
    """
    if source == 'gsm8k':
        # 数学题: 提取数字比较
        pred_num = extract_gsm8k_answer(pred)
        gold_num = extract_gsm8k_answer(gold)
        if pred_num is None or gold_num is None:
            return False
        return abs(pred_num - gold_num) < 1e-6
    else:
        # 选择题/推理题: 完整答案匹配 (忽略大小写和空白)
        pred_clean = pred.strip().lower()
        gold_clean = gold.strip().lower()
        # 检查预测是否包含正确答案，或正确答案是否在预测开头
        return gold_clean in pred_clean or pred_clean.startswith(gold_clean)


def format_prompt(instruction: str, source: str) -> str:
    """
    根据数据源格式化 prompt
    """
    if source == 'gsm8k':
        return f"Question: {instruction}\nAnswer:"
    elif source == 'commonsense_qa':
        return f"{instruction}\nThe answer is:"
    elif source == 'strategy_qa':
        return f"{instruction}\nAnswer (Yes or No):"
    else:
        return f"{instruction}\nAnswer:"


# ============================================================================
#                         ForcedRouter 类 (单专家测试)
# ============================================================================

class ForcedRouter(nn.Module):
    """强制路由到单一专家的 Router"""
    
    def __init__(self, expert_idx: int, num_experts: int):
        super().__init__()
        self.expert_idx = expert_idx
        self.num_experts = num_experts
    
    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        probs = torch.zeros(
            batch_size, seq_len, self.num_experts,
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        probs[:, :, self.expert_idx] = 1.0
        return probs


# ============================================================================
#                         模型生成函数
# ============================================================================

def generate_with_base_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    sigma: float = 0.0
) -> str:
    """
    使用基础模型 (带 LoRA adapter) 生成回答
    
    Args:
        model: 模型
        tokenizer: tokenizer
        prompt: 输入 prompt
        max_new_tokens: 最大生成 token 数
        sigma: 噪声标准差 (在 logits 上添加)
    
    Returns:
        生成的文本
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 自定义生成逻辑以支持加噪
    generated_ids = inputs.input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(generated_ids)
            logits = outputs.logits[:, -1, :].float()
            
            # 添加噪声
            if sigma > 0:
                noise = torch.randn_like(logits) * sigma
                logits = logits + noise
            
            # Greedy decoding
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # 检查是否生成了结束符
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # 解码生成的部分
    response = tokenizer.decode(
        generated_ids[0, inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    return response


def generate_with_comix(
    model: CoMixModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    sigma: float = 0.0
) -> str:
    """
    使用 CoMixModel 生成回答 (支持 Central DP)
    
    噪声在 Router 聚合后的 logits 上添加
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generated_ids = inputs.input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(generated_ids)
            logits = outputs.logits[:, -1, :].float()
            
            # Central DP: 在聚合后添加噪声
            if sigma > 0:
                noise = torch.randn_like(logits) * sigma
                logits = logits + noise
            
            # Greedy decoding
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    response = tokenizer.decode(
        generated_ids[0, inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    return response


def generate_aggregated_A(
    expert_models: list,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    sigma: float = 0.0
) -> str:
    """
    Aggregated_A: 先加噪再聚合 (Local DP)
    
    每个专家独立生成 logits 并加噪，然后平均聚合
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(expert_models[0].device)
    generated_ids = inputs.input_ids.clone()
    num_experts = len(expert_models)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 收集所有专家的 logits
            all_logits = []
            for model in expert_models:
                outputs = model(generated_ids)
                logits = outputs.logits[:, -1, :].float()
                
                # Local DP: 每个专家独立加噪
                if sigma > 0:
                    noise = torch.randn_like(logits) * sigma
                    logits = logits + noise
                
                all_logits.append(logits)
            
            # 聚合 (平均)
            agg_logits = sum(all_logits) / num_experts
            
            # Greedy decoding
            next_token = agg_logits.argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    response = tokenizer.decode(
        generated_ids[0, inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    return response


# ============================================================================
#                         评估方案实现
# ============================================================================

class BenchmarkEvaluator:
    """Benchmark 评估器"""
    
    def __init__(self, base_model_path: str, lora_dir: str, expert_dir: str,
                 router_checkpoint: str, num_experts: int = 8):
        self.base_model_path = base_model_path
        self.lora_dir = lora_dir
        self.expert_dir = expert_dir
        self.router_checkpoint = router_checkpoint
        self.num_experts = num_experts
        
        self.tokenizer = None
        self.base_model = None
        self.comix_model = None
        self.comix_model_local_dp = None
        self.expert_adapters_loaded = False
        
    def load_tokenizer(self):
        """加载 tokenizer"""
        if self.tokenizer is None:
            print(">>> Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path, trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer
    
    def load_base_model(self):
        """加载基础模型 (半精度 bfloat16)"""
        if self.base_model is None:
            print(">>> Loading base model (bfloat16)...")
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                device_map={"": 0},
                torch_dtype=torch.bfloat16
            )
        return self.base_model
    
    def load_comix_model(self):
        """加载 CoMixModel (用于 Central DP)"""
        if self.comix_model is None:
            print(">>> Loading CoMixModel...")
            self.comix_model = CoMixModel(
                base_model_path=self.base_model_path,
                num_experts=self.num_experts,
                r=16,
                alpha=64,
                device="cuda"
            )
            self.comix_model.load_experts(self.expert_dir)
            
            # 加载训练好的 Router
            if os.path.exists(self.router_checkpoint):
                print(f">>> Loading Router from {self.router_checkpoint}...")
                router_state = torch.load(self.router_checkpoint, map_location="cuda")
                self.comix_model.routers.load_state_dict(router_state)
            else:
                print("!!! Warning: Router checkpoint not found, using random initialization.")
        
        return self.comix_model
    
    def evaluate_single_expert_avg(self, test_data: list, sigma: float) -> dict:
        """
        评估方案 2: Single Expert (只评估 expert_0 以加速)
        """
        model = self.load_base_model()
        tokenizer = self.load_tokenizer()
        
        # 只评估 expert_0
        exp_idx = 0
        adapter_path = f"{self.lora_dir}/expert_{exp_idx}"
        try:
            model.load_adapter(adapter_path, adapter_name=f"expert_{exp_idx}")
            model.set_adapter(f"expert_{exp_idx}")
        except Exception as e:
            print(f"!!! Failed to load expert {exp_idx}: {e}")
            return None
        
        expert_results = {"correct": 0, "total": 0, "by_source": {}}
        
        for sample in tqdm(test_data, desc=f"Single Expert (expert_0)", leave=False):
            prompt = format_prompt(sample["instruction"], sample["source"])
            pred = generate_with_base_model(model, tokenizer, prompt, sigma=sigma)
            
            is_correct = evaluate_answer(pred, sample["output"], sample["source"])
            expert_results["correct"] += int(is_correct)
            expert_results["total"] += 1
            
            source = sample["source"]
            if source not in expert_results["by_source"]:
                expert_results["by_source"][source] = {"correct": 0, "total": 0}
            expert_results["by_source"][source]["correct"] += int(is_correct)
            expert_results["by_source"][source]["total"] += 1
        
        try:
            model.delete_adapter(f"expert_{exp_idx}")
        except:
            pass
        
        return expert_results
    
    def load_comix_model_local_dp(self):
        """
        加载 CoMixModel 用于 Local DP 评估
        使用均匀权重路由器
        """
        if not hasattr(self, 'comix_model_local_dp') or self.comix_model_local_dp is None:
            print(">>> Loading CoMixModel for Local DP...")
            self.comix_model_local_dp = CoMixModel(
                base_model_path=self.base_model_path,
                num_experts=self.num_experts,
                r=16,
                alpha=64,
                device="cuda"
            )
            self.comix_model_local_dp.load_experts(self.expert_dir)
            # Local DP 使用均匀权重 (Router 已默认初始化为均匀分布)
            # 不加载训练好的 Router，保持均匀
            print(">>> Using uniform routing for Local DP")
        
        return self.comix_model_local_dp
    
    def evaluate_aggregated_A(self, test_data: list, sigma: float) -> dict:
        """
        评估方案 3: Aggregated_A (Local DP)
        先加噪再聚合 - 使用 CoMixModel，在 LoRA 层级每个专家独立加噪后聚合
        优化版本：复用 CoMixModel 架构，一次前向计算所有专家
        
        注意：噪声加在 LoRA 输出层 (CoMixLoRALayer)，使用 clip_threshold 和 noise_sigma
        """
        comix_model = self.load_comix_model_local_dp()
        tokenizer = self.load_tokenizer()
        
        # 设置 Local DP 噪声参数 (噪声会在 CoMixLoRALayer 中对每个专家独立加)
        comix_model.noise_sigma = sigma
        comix_model.clip_threshold = CLIP_THRESHOLD
        
        results = {"correct": 0, "total": 0, "by_source": {}}
        
        for sample in tqdm(test_data, desc="Aggregated_A (Local DP)"):
            prompt = format_prompt(sample["instruction"], sample["source"])
            # 噪声已在 CoMixLoRALayer 内部对每个专家独立添加
            pred = generate_with_comix(comix_model, tokenizer, prompt, sigma=0)
            
            is_correct = evaluate_answer(pred, sample["output"], sample["source"])
            results["correct"] += int(is_correct)
            results["total"] += 1
            
            source = sample["source"]
            if source not in results["by_source"]:
                results["by_source"][source] = {"correct": 0, "total": 0}
            results["by_source"][source]["correct"] += int(is_correct)
            results["by_source"][source]["total"] += 1
        
        # 重置噪声参数
        comix_model.noise_sigma = 0.0
        comix_model.clip_threshold = -1.0
        
        return results
    
    def evaluate_aggregated_B(self, test_data: list, sigma_B: float) -> dict:
        """
        评估方案 4: Aggregated_B (Central DP)
        先聚合再加噪 - 使用 CoMixModel Router 进行加权聚合，然后统一加噪
        灵敏度从 C 降低为 C/N
        """
        comix_model = self.load_comix_model()
        tokenizer = self.load_tokenizer()
        
        results = {"correct": 0, "total": 0, "by_source": {}}
        
        for sample in tqdm(test_data, desc="Aggregated_B (Central DP)"):
            prompt = format_prompt(sample["instruction"], sample["source"])
            pred = generate_with_comix(comix_model, tokenizer, prompt, sigma=sigma_B)
            
            is_correct = evaluate_answer(pred, sample["output"], sample["source"])
            results["correct"] += int(is_correct)
            results["total"] += 1
            
            source = sample["source"]
            if source not in results["by_source"]:
                results["by_source"][source] = {"correct": 0, "total": 0}
            results["by_source"][source]["correct"] += int(is_correct)
            results["by_source"][source]["total"] += 1
        
        return results


# ============================================================================
#                         结果处理与可视化
# ============================================================================

def compute_accuracy(results: dict) -> dict:
    """计算准确率"""
    if results is None:
        return None
    
    acc = {
        "overall": results["correct"] / results["total"] * 100 if results["total"] > 0 else 0
    }
    
    for source, data in results.get("by_source", {}).items():
        if isinstance(data, dict) and data.get("total", 0) > 0:
            acc[source] = data["correct"] / data["total"] * 100
        else:
            acc[source] = 0.0
    
    return acc


def print_results_table(epsilon: float, sigma: float, sigma_B: float, all_results: dict):
    """打印结果表格"""
    eps_str = epsilon_to_str(epsilon)
    
    print(f"\n{'='*100}")
    print(f"Epsilon: {eps_str:<6} | Delta: {DELTA} | Clip: {CLIP_THRESHOLD} | Queries: {MAX_NEW_TOKENS}")
    print(f"Sigma_A (Local DP, RDP): {sigma:.4f} | Sigma_B (Central DP, RDP): {sigma_B:.4f}")
    print(f"{'='*100}")
    print(f"{'Method':<28} | {'Overall':<10} | {'GSM8K':<10} | {'CSQA':<10} | {'StrategyQA':<10}")
    print(f"{'-'*100}")
    
    method_names = {
        "single_expert_avg": "Single Expert (expert_0)",
        "aggregated_A": "Aggregated_A (Local DP)",
        "aggregated_B": "Aggregated_B (Central DP)"
    }
    
    for method_key, method_name in method_names.items():
        acc = all_results.get(method_key)
        if acc is None:
            print(f"{method_name:<28} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10}")
        else:
            overall = f"{acc.get('overall', 0):.1f}%"
            gsm8k = f"{acc.get('gsm8k', 0):.1f}%"
            csqa = f"{acc.get('commonsense_qa', 0):.1f}%"
            strategy = f"{acc.get('strategy_qa', 0):.1f}%"
            print(f"{method_name:<28} | {overall:<10} | {gsm8k:<10} | {csqa:<10} | {strategy:<10}")
    
    print(f"{'-'*100}")


def plot_results(all_epsilon_results: dict, output_path: str):
    """生成可视化图表"""
    epsilons = []
    methods = ["single_expert_avg", "aggregated_A", "aggregated_B"]
    method_labels = ["Single Expert (expert_0)", "Aggregated_A (LDP)", "Aggregated_B (CDP)"]
    colors = ["orange", "blue", "red"]
    
    # 收集数据
    data = {method: {"overall": [], "gsm8k": [], "commonsense_qa": [], "strategy_qa": []} 
            for method in methods}
    
    for eps_str, results in all_epsilon_results.items():
        epsilons.append(eps_str)
        for method in methods:
            acc = results.get(method)
            if acc:
                data[method]["overall"].append(acc.get("overall", 0))
                data[method]["gsm8k"].append(acc.get("gsm8k", 0))
                data[method]["commonsense_qa"].append(acc.get("commonsense_qa", 0))
                data[method]["strategy_qa"].append(acc.get("strategy_qa", 0))
            else:
                for key in data[method]:
                    data[method][key].append(0)
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    titles = ["Overall Accuracy", "GSM8K Accuracy", "CommonsenseQA Accuracy", "StrategyQA Accuracy"]
    keys = ["overall", "gsm8k", "commonsense_qa", "strategy_qa"]
    
    x = np.arange(len(epsilons))
    width = 0.2
    
    for ax, title, key in zip(axes.flatten(), titles, keys):
        for i, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
            values = data[method][key]
            ax.bar(x + i * width, values, width, label=label, color=color, alpha=0.8)
        
        ax.set_xlabel("Epsilon")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(title)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(epsilons)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\n>>> Plot saved to {output_path}")
    
    # 创建趋势线图
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    for method, label, color in zip(methods, method_labels, colors):
        values = data[method]["overall"]
        ax2.plot(epsilons, values, 'o-', label=label, color=color, linewidth=2, markersize=8)
    
    ax2.set_xlabel("Epsilon (Privacy Budget)", fontsize=12)
    ax2.set_ylabel("Overall Accuracy (%)", fontsize=12)
    ax2.set_title("Privacy-Accuracy Tradeoff: Local DP vs Central DP", fontsize=14)
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    trend_path = output_path.replace(".png", "_trend.png")
    plt.savefig(trend_path, dpi=150)
    print(f">>> Trend plot saved to {trend_path}")


# ============================================================================
#                              主函数
# ============================================================================

def main():
    print("="*100)
    print("DP Comparison Benchmark")
    print("Local DP (先加噪再聚合) vs Central DP (先聚合再加噪)")
    print("="*100)
    
    # 1. 加载测试数据
    print(f"\n>>> [1/5] Loading test data from {TEST_DATA_PATH}...")
    test_data = load_test_data(TEST_DATA_PATH, N_TEST_SAMPLES)
    
    # 统计数据分布
    source_counts = defaultdict(int)
    for sample in test_data:
        source_counts[sample["source"]] += 1
    
    print(f">>> Loaded {len(test_data)} samples:")
    for source, count in source_counts.items():
        print(f"    - {source}: {count}")
    
    # 2. 初始化评估器
    print("\n>>> [2/5] Initializing evaluator...")
    evaluator = BenchmarkEvaluator(
        base_model_path=BASE_MODEL,
        lora_dir=LORA_DIR,
        expert_dir=EXPERT_DIR,
        router_checkpoint=ROUTER_CHECKPOINT,
        num_experts=NUM_EXPERTS
    )
    evaluator.load_tokenizer()
    
    # 3. 运行评估
    print("\n>>> [3/5] Running benchmark...")
    
    all_results = {
        "config": {
            "n_samples": N_TEST_SAMPLES,
            "max_new_tokens": MAX_NEW_TOKENS,
            "num_queries": MAX_NEW_TOKENS,  # 查询次数 = token 数
            "privacy_accounting": "RDP (Rényi DP) Composition",
            "epsilons": [epsilon_to_str(e) for e in EPSILONS],
            "delta": DELTA,
            "clip_threshold": CLIP_THRESHOLD,
            "num_experts": NUM_EXPERTS,
            "source_distribution": dict(source_counts),
            "timestamp": datetime.now().isoformat()
        },
        "results": {}
    }
    
    for epsilon in EPSILONS:
        eps_str = epsilon_to_str(epsilon)
        print(f"\n{'='*50}")
        print(f">>> Evaluating with epsilon = {eps_str}")
        print(f"{'='*50}")
        
        # 计算噪声标准差 (考虑 RDP 组合：MAX_NEW_TOKENS 次查询)
        # Single Expert 和 Aggregated_A 使用相同的 sigma (基于灵敏度 C)
        # Aggregated_B (Central DP) 的灵敏度降低为 C/N，因此噪声更小
        sigma = compute_sigma(epsilon, DELTA, CLIP_THRESHOLD, num_queries=MAX_NEW_TOKENS)
        sigma_B = compute_sigma(epsilon, DELTA, CLIP_THRESHOLD / NUM_EXPERTS, num_queries=MAX_NEW_TOKENS)
        
        print(f">>> Sigma_A (Local DP, {MAX_NEW_TOKENS} queries): {sigma:.4f}")
        print(f">>> Sigma_B (Central DP, {MAX_NEW_TOKENS} queries): {sigma_B:.4f}")
        
        eps_results = {
            "sigma": sigma,
            "sigma_B": sigma_B
        }
        
        # 方案 1: Single Expert (expert_0 only)
        print("\n--- Evaluating Single Expert (expert_0 only) ---")
        single_results = evaluator.evaluate_single_expert_avg(test_data, sigma)
        eps_results["single_expert_avg"] = compute_accuracy(single_results)
        
        # 方案 2: Aggregated_A (Local DP)
        print("\n--- Evaluating Aggregated_A (Local DP) ---")
        agg_A_results = evaluator.evaluate_aggregated_A(test_data, sigma)
        eps_results["aggregated_A"] = compute_accuracy(agg_A_results)
        
        # 方案 3: Aggregated_B (Central DP)
        print("\n--- Evaluating Aggregated_B (Central DP) ---")
        agg_B_results = evaluator.evaluate_aggregated_B(test_data, sigma_B)
        eps_results["aggregated_B"] = compute_accuracy(agg_B_results)
        
        # 打印结果表格
        print_results_table(epsilon, sigma, sigma_B, eps_results)
        
        all_results["results"][eps_str] = eps_results
    
    # 4. 保存结果
    print("\n>>> [4/5] Saving results...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    output_json = os.path.join(OUTPUT_DIR, "benchmark_results.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f">>> Results saved to {output_json}")
    
    # 5. 生成可视化
    print("\n>>> [5/5] Generating plots...")
    output_plot = os.path.join(OUTPUT_DIR, "benchmark_comparison.png")
    plot_results(all_results["results"], output_plot)
    
    # 打印总结
    print("\n" + "="*100)
    print(">>> Benchmark Complete!")
    print("="*100)
    print("\n>>> Summary:")
    print("  - Local DP (Aggregated_A): 每个专家独立加噪后聚合")
    print("  - Central DP (Aggregated_B): 使用 Router 加权聚合后统一加噪")
    print(f"  - 噪声比例: Sigma_A / Sigma_B = {NUM_EXPERTS} (专家数量)")
    print(f"\n>>> Output files:")
    print(f"    - {output_json}")
    print(f"    - {output_plot}")


if __name__ == "__main__":
    main()
