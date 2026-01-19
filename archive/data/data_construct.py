import json
import os
from datasets import load_dataset

# === 配置 ===
WINDOW_SIZE = 1000
OVERLAP_RATIO = 0.5
STRIDE = int(WINDOW_SIZE * (1 - OVERLAP_RATIO))
DATA_DIR = "/root/gpufree-data/lapped-lora/data"
os.makedirs(DATA_DIR, exist_ok=True)

# 加载足够大的数据集
def load_raw_data():
    print("Loading GSM8K...")
    ds = load_dataset("gsm8k", "main")
    # 合并 train 和 test
    all_data = list(ds['train']) + list(ds['test']) 
    # 提取 question + answer
    formatted_data = []
    for item in all_data:
        text = f"Question: {item['question']}\nAnswer: {item['answer']}"
        formatted_data.append({"text": text})
    print(f"Total available samples: {len(formatted_data)}")
    return formatted_data

def generate_for_n(all_data, n_loras):
    print(f"\n--- Generating Data for N={n_loras} ---")
    required_total = WINDOW_SIZE + (n_loras - 1) * STRIDE
    
    if required_total > len(all_data):
        print(f"⚠️ Warning: N={n_loras} requires {required_total} samples, but only {len(all_data)} available.")
        return

    # 1. 截取当前 N 对应的总数据
    current_global_data = all_data[:required_total]
    global_path = f"{DATA_DIR}/subset_global.json"
    with open(global_path, 'w') as f:
        json.dump(current_global_data, f, indent=2)
    print(f"Generated Global Baseline: {len(current_global_data)} samples")

    # 2. 生成 N 个 Local Subsets
    for i in range(n_loras):
        start_idx = i * STRIDE
        end_idx = start_idx + WINDOW_SIZE
        subset_data = current_global_data[start_idx:end_idx]
        
        subset_path = f"{DATA_DIR}/subset_{i}.json"
        with open(subset_path, 'w') as f:
            json.dump(subset_data, f, indent=2)
        print(f"  - Subset {i}: [{start_idx}, {end_idx}) - {len(subset_data)} samples")

if __name__ == "__main__":
    raw_data = load_raw_data()
    generate_for_n(raw_data, 8)