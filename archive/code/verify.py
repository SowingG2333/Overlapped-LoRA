import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# === 1. 环境变量 ===
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# === 配置 ===
BASE_MODEL = "/root/gpufree-data/hf/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
LORA_DIR = "/root/gpufree-data/lapped-lora/model/subset_8"
NUM_SUBSETS = 8

# 目标问题
TEST_PROMPT = "Question: Janet has 3 times as many marbles as Arnold. If Arnold has 12 marbles, how many marbles do they have together?\nAnswer:"

# === 关键修正：手动指定正确的 Token ID ===
# 根据你的诊断结果：' Janet' (id=53665)
TARGET_ID = 53665 

# 噪声等级 (建议多测几个点以便画图)
NOISE_LEVELS = [0.0, 2.0, 4.0, 5.0, 6.0, 8.0]
N_SAMPLES = 20 # 蒙特卡洛采样次数

def measure_confidence(logits, target_id, sigma, n_samples=1):
    """计算目标 Token 的 Logit 和 Rank"""
    logit_sums = 0
    rank_sums = 0
    clean_logits = logits.float()
    
    for _ in range(n_samples):
        if sigma <= 0:
            noisy_logits = clean_logits
        else:
            noise = torch.normal(mean=0, std=sigma, size=clean_logits.size(), device=clean_logits.device)
            noisy_logits = clean_logits + noise
        
        target_logit = noisy_logits[0, target_id].item()
        logit_sums += target_logit
        
        # 计算排名 (数值 1 代表最好)
        rank = (noisy_logits[0] > target_logit).sum().item() + 1
        rank_sums += rank

    return logit_sums / n_samples, rank_sums / n_samples

def main():
    print("Loading Base Model (BF16)...")
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
        dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    inputs = tokenizer(TEST_PROMPT, return_tensors="pt").to(base_model.device)
    
    print(f"\nPrompt: {TEST_PROMPT.strip()}")
    print(f"Target Token ID: {TARGET_ID} (Manually Set)")
    print("=" * 95)
    print(f"{'Noise':<6} | {'Model':<12} | {'Avg Logit (High Better)':<25} | {'Avg Rank (1.0 is Best)':<25}")
    print("-" * 95)

    for sigma in NOISE_LEVELS:
        # 1. Global Baseline
        try:
            base_model.load_adapter(f"{LORA_DIR}/lora_global", adapter_name="global")
            base_model.set_adapter("global")
            with torch.no_grad():
                outputs = base_model(**inputs)
                logits = outputs.logits[:, -1, :]
                avg_logit, avg_rank = measure_confidence(logits, TARGET_ID, sigma, N_SAMPLES)
            print(f"{sigma:<6} | {'Global':<12} | {avg_logit:<25.4f} | {avg_rank:<25.2f}")
            base_model.delete_adapter("global")
        except:
            print(f"{sigma:<6} | {'Global':<12} | Error")

        # 2. Local Baseline
        try:
            base_model.load_adapter(f"{LORA_DIR}/lora_0", adapter_name="local_0")
            base_model.set_adapter("local_0")
            with torch.no_grad():
                outputs = base_model(**inputs)
                logits = outputs.logits[:, -1, :]
                avg_logit, avg_rank = measure_confidence(logits, TARGET_ID, sigma, N_SAMPLES)
            print(f"{sigma:<6} | {'Local_0':<12} | {avg_logit:<25.4f} | {avg_rank:<25.2f}")
            base_model.delete_adapter("local_0")
        except:
             print(f"{sigma:<6} | {'Local_0':<12} | Error")

        # 3. Aggregated (Ours)
        clean_logits_list = []
        for i in range(NUM_SUBSETS):
            try:
                adapter_name = f"lora_{i}"
                base_model.load_adapter(f"{LORA_DIR}/{adapter_name}", adapter_name=adapter_name)
                base_model.set_adapter(adapter_name)
                with torch.no_grad():
                    out = base_model(**inputs)
                    clean_logits_list.append(out.logits[:, -1, :].float())
                base_model.delete_adapter(adapter_name)
            except:
                continue
        
        if clean_logits_list:
            agg_logit_sum = 0
            agg_rank_sum = 0
            for _ in range(N_SAMPLES):
                current_round_sum = 0
                for l_logits in clean_logits_list:
                    if sigma > 0:
                        noise = torch.normal(mean=0, std=sigma, size=l_logits.size(), device=l_logits.device)
                        current_round_sum += (l_logits + noise)
                    else:
                        current_round_sum += l_logits
                
                avg_noisy_logits = current_round_sum / len(clean_logits_list)
                agg_logit_sum += avg_noisy_logits[0, TARGET_ID].item()
                agg_rank_sum += (avg_noisy_logits[0] > avg_noisy_logits[0, TARGET_ID]).sum().item() + 1

            final_avg_logit = agg_logit_sum / N_SAMPLES
            final_avg_rank = agg_rank_sum / N_SAMPLES
            print(f"{sigma:<6} | {'Aggregated':<12} | {final_avg_logit:<25.4f} | {final_avg_rank:<25.2f}")
        else:
            print(f"{sigma:<6} | {'Aggregated':<12} | Error")
            
        print("-" * 95)

if __name__ == "__main__":
    main()