import os
import torch
import gc # 引入垃圾回收模块

# === 1. 强制设置环境变量，解决 RTX 4090 等显卡的 NCCL 通信报错 ===
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from trl import SFTTrainer

# === 配置 ===
MODEL_ID = "/root/gpufree-data/hf/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
DATA_DIR = "/root/gpufree-data/lapped-lora/data/subset_8"
OUTPUT_DIR = "/root/gpufree-data/lapped-lora/model/subset_8"
NUM_SUBSETS = 8

def train_one_lora(subset_id):
    # === 根据 subset_id 判断是训练局部子集还是全局子集 ===
    if subset_id == "global":
        print(f"\n=== Training LoRA Global ===")
        data_file = f"{DATA_DIR}/subset_global.json"
        output_path = f"{OUTPUT_DIR}/lora_global"
    else:
        print(f"\n=== Training LoRA {subset_id} ===")
        data_file = f"{DATA_DIR}/subset_{subset_id}.json"
        output_path = f"{OUTPUT_DIR}/lora_{subset_id}"

    # === 量化计算类型改为 bfloat16 ===
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # === 模型加载 ===
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map={"": 0}, 
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token 

    # === LoRA 配置 ===
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"] 
    )
    model = get_peft_model(model, peft_config)

    # === 加载对应的数据集文件 ===
    print(f"Loading data from: {data_file}")
    dataset = load_dataset("json", data_files=data_file, split="train")

    # === 训练参数 ===
    training_args = TrainingArguments(
        output_dir=output_path, # 使用动态生成的输出路径
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=3,
        save_strategy="no",
        
        fp16=False,
        bf16=True,
        
        ddp_find_unused_parameters=False,
        report_to="none",
    )

    def formatting_prompts_func(example):
        return example['text']

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,               
        formatting_func=formatting_prompts_func, 
        args=training_args,
    )

    trainer.train()
    
    # 保存模型
    model.save_pretrained(output_path)
    print(f"LoRA saved to {output_path}.")
    
    # === 显存清理 ===
    del model, trainer
    gc.collect() # 显式进行垃圾回收
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # 1. 先训练所有的局部编号子集 (0, 1, 2...)
    for i in range(NUM_SUBSETS):
        train_one_lora(i)
        
    # 2. 最后训练全局子集
    train_one_lora("global")