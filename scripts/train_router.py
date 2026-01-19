import os
import sys
import torch
import logging
from dataclasses import dataclass, field
from typing import Dict
# 添加 safetensors 导入
from safetensors.torch import load_file 

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset

# === 关键：导入 MixLoRA 模块 ===
# 假设你的目录结构是 OverlappedLoRA/MixLoRA
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
mixlora_path = os.path.join(root_dir, "MixLoRA")
sys.path.append(mixlora_path)

from mixlora import MixLoraConfig

# === 关键修复：MixLoRA 尚未支持 qwen3，手动注册兼容模式 ===
import mixlora.model
# Qwen3 的 MLP 结构与 Llama 一致，复用其 Forward 逻辑
mixlora.model._compatible_model_types["qwen3"] = "_llama_forward"
# ==========================================================

from mixlora.model import inject_adapter_in_model  # 直接使用注入函数

logger = logging.getLogger(__name__)

@dataclass
class Arguments:
    base_model: str = "/root/gpufree-data/hf/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
    expert_dir: str = "/root/gpufree-data/OverlappedLoRA/model"
    router_data: str = "/root/gpufree-data/OverlappedLoRA/data/router_train.json"
    num_experts: int = 8
    noise_sigma: float = 4.0
    clip_threshold: float = 1.0

def train():
    parser = transformers.HfArgumentParser((Arguments, TrainingArguments))
    
    # === 修改开始：添加默认参数逻辑 ===
    # 如果运行脚本没传命令行参数，则使用这些默认值
    if len(sys.argv) == 1:
        print(">>> No args provided, using default arguments...")
        args, training_args = parser.parse_args_into_dataclasses(args=[
            "--output_dir", "/root/gpufree-data/OverlappedLoRA/model/router",  #在此处定义默认输出目录
            "--num_train_epochs", "3",
            "--per_device_train_batch_size", "1",   # <--- 修改：降到 1 (原为 4)
            "--gradient_accumulation_steps", "16",  # <--- 修改：增加到 16 (原为 4)，保持等效 Batch Size = 16
            "--learning_rate", "1e-4",
            "--logging_steps", "10",
            "--save_steps", "100",
            "--save_total_limit", "2",
            "--overwrite_output_dir",
            "--gradient_checkpointing", "True"      # <--- 新增：开启梯度检查点，节省大量显存
        ])
    else:
        args, training_args = parser.parse_args_into_dataclasses()
    # === 修改结束 ===

    # 1. 初始化 Config
    print(">>> [1/5] Initializing Config...")
    config_dict = {
        "base_model_name_or_path": args.base_model,
        "task_type": "CAUSAL_LM",
        "peft_type": "MIXLORA",
        "routing_strategy": "mixlora",
        "num_experts": args.num_experts,
        "top_k": 2,
        # Privacy Params
        "noise_sigma": args.noise_sigma,
        "clip_threshold": args.clip_threshold,
        # Router Params
        "router_loss": True,
        "router_aux_loss_coef": 0.01,
        "router_init_range": 0.02,
        # LoRA Params (必须与 Expert 训练时一致)
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # === 关键修复：添加激活函数 (Qwen/Llama 使用 silu) ===
        "act_fn": "silu"
    }
    config = MixLoraConfig.from_config(config_dict)
    config.dtype_ = torch.bfloat16
    config.adapter_name_ = "default"  # <--- 关键修复：显式设置 adapter_name_，防止 key 为 None

    # 2. 加载 Base Model
    print(f">>> [2/5] Loading Base Model from {args.base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        # 显式关闭 flash attn 以避免潜在安装问题，或者根据你的环境开启
        # attn_implementation="flash_attention_2" 
    )

    # 3. 内存中构建 MixLoRA 权重 (Stitching in Memory)
    print(">>> [3/5] Stitching Experts in Memory...")
    mixlora_weights = {}
    
    # 临时存储 Attention 层的权重用于取平均 (Shared Attention)
    attn_accum = {} 
    attn_count = 0

    # 遍历所有专家
    for i in range(args.num_experts):
        # 修改开始：自动检测权重文件格式 (safetensors 或 bin)
        expert_base_path = os.path.join(args.expert_dir, f"expert_{i}")
        safetensors_path = os.path.join(expert_base_path, "adapter_model.safetensors")
        bin_path = os.path.join(expert_base_path, "adapter_model.bin")

        if os.path.exists(safetensors_path):
            print(f"  - Loading Expert {i} weights from {safetensors_path}...")
            expert_state = load_file(safetensors_path)
        elif os.path.exists(bin_path):
            print(f"  - Loading Expert {i} weights from {bin_path}...")
            expert_state = torch.load(bin_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"Expert checkpoint not found in {expert_base_path}")
        # 修改结束
        
        for k, v in expert_state.items():
            # k 示例: "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight"
            if "lora_" not in k: continue
            
            # 解析 Key
            # key 结构通常是: base_model.model.model.layers.{layer_idx}.{module}.{proj}.lora_{A/B}.weight
            parts = k.split(".")
            try:
                # 找到 layer_idx 的位置 (通常是 'layers' 后面一个)
                if "layers" in parts:
                    idx = parts.index("layers")
                    layer_idx = parts[idx+1]
                    module_part = parts[idx+2] # self_attn 或 mlp
                    proj_part = parts[idx+3]   # q_proj, gate_proj 等
                    suffix = ".".join(parts[idx+4:]) # lora_A.weight
                else:
                    continue
            except IndexError:
                continue

            # === 分支 A: MLP 层 (MoE Experts) ===
            # gate_proj, up_proj, down_proj -> 保持独立，变为 Experts
            if proj_part in ["gate_proj", "up_proj", "down_proj"]:
                new_key = f"mixlora.layers.{layer_idx}.mlp.{proj_part}.experts.{i}.{suffix}"
                mixlora_weights[new_key] = v

            # === 分支 B: Attention 层 (Shared LoRA) ===
            # q_proj, k_proj, v_proj, o_proj -> 需要取平均
            # MixLoRA 架构中 Attention LoRA 是共享的，不是 MoE
            elif proj_part in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                # 构造一个通用的 Key (不带 expert id)
                attn_key = f"mixlora.layers.{layer_idx}.self_attn.{proj_part}.{suffix}"
                if attn_key not in attn_accum:
                    attn_accum[attn_key] = v.clone().float() # 转 float 防止精度溢出
                else:
                    attn_accum[attn_key] += v.float()
    
    # 处理 Attention 权重平均化
    print("  - Averaging Attention LoRA weights (Shared Consensus)...")
    for k, v_sum in attn_accum.items():
        mixlora_weights[k] = (v_sum / args.num_experts).to(dtype=torch.bfloat16)

    # 4. 初始化 Router 权重
    print(">>> [4/5] Initializing Router Gates...")
    hidden_size = model.config.hidden_size
    num_experts = args.num_experts
    num_layers = model.config.num_hidden_layers
    
    for layer_idx in range(num_layers):
        # MixLoRA Router Key: mixlora.layers.{layer}.mlp.moe_gate.weight
        # Shape: (num_experts, hidden_size)
        gate_key = f"mixlora.layers.{layer_idx}.mlp.moe_gate.weight"
        
        # 随机初始化 (Normal distribution with small std)
        gate_weight = torch.randn(num_experts, hidden_size, dtype=torch.bfloat16) * 0.02
        mixlora_weights[gate_key] = gate_weight

    # === 新增：确保注入前的权重字典全是 bfloat16 ===
    for k, v in mixlora_weights.items():
        if v.dtype != torch.bfloat16:
            mixlora_weights[k] = v.to(torch.bfloat16)
    # ==========================================

    # 5. 注入权重到模型
    print(">>> [5/5] Injecting Adapter to Base Model...")
    # 这一步会修改 model 的结构，插入 MixLoraSparseMoe 和 LoraLinear
    inject_adapter_in_model(model, config, mixlora_weights)

    # === 关键修复 2：将 MixLoRA 模块注册为 PyTorch 子模块 ===
    # mixlora 默认实现使用普通 dict 存储 submodule，导致 pytorch 无法识别参数和进行类型转换
    print(">>> Registering MixLoRA modules to PyTorch...")
    
    # 1. 先收集需要修改的模块，避免在遍历中修改引发 RecursionError
    modules_to_patch = []
    for name, module in model.named_modules():
        if hasattr(module, "mixlora_moes") and isinstance(module.mixlora_moes, dict):
            modules_to_patch.append(module)
            
    # 2. 批量修改
    for module in modules_to_patch:
        # 兼容性修复：如果字典里有 None Key，替换为 "default"
        safe_moes = {}
        for k, v in module.mixlora_moes.items():
            if k is None:
                safe_moes["default"] = v
            else:
                safe_moes[str(k)] = v
        
        # === 核心修复：打破循环引用 ===
        # MixLoraSparseMoe.base_layer_ 指向了 Parent Module (module)，导致 nn.ModuleDict 注册后出现递归死循环
        # 我们需要从子模块注册表中移除 base_layer_，但保留对象引用以便 forward 调用
        for moe_instance in safe_moes.values():
            if "base_layer_" in moe_instance._modules:
                # 1. 获取引用
                base_layer = moe_instance._modules["base_layer_"]
                # 2. 从 PyTorch 注册表中删除 (切断 model.to/parameters 的递归路径)
                del moe_instance._modules["base_layer_"]
                # 3. 重新挂载为普通 Python 属性 (绕过 __setattr__ 的自动注册)
                moe_instance.__dict__["base_layer_"] = base_layer
        
        # 将普通 dict 转换为 nn.ModuleDict
        module.mixlora_moes = torch.nn.ModuleDict(safe_moes)
        
    print(f">>> Patched {len(modules_to_patch)} modules.")

    # === 新增：强制模型整体为 bfloat16 ===
    # 现在 patch 完成了，这里的一次调用就会递归处理所有 mixlora_moes
    model.to(dtype=torch.bfloat16) 
    
    # === 关键：开启 Input Gradients 以支持 Gradient Checkpointing ===
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
    # ==================================

    # 冻结除 Router 外的所有参数
    print(">>> Setting up freeze/unfreeze...")
    trainable_count = 0
    for name, param in model.named_parameters():
        if "moe_gate" in name:
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False
    
    print(f"  Trainable Parameters (Router Only): {trainable_count}")

    # 6. 开始训练
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files=args.router_data, split="train")
    
    def preprocess(examples):
        inputs = [f"Below is an instruction that describes a task.\n\n### Instruction:\n{i}\n\n### Response:\n{o}{tokenizer.eos_token}" 
                  for i, o in zip(examples['instruction'], examples['output'])]
        
        # 获取 tokenized 输出
        model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)
        
        # === 关键修复：添加 labels ===
        # 对于 Causal LM 训练，labels 通常等于 input_ids
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        
        # 可选：将 pad_token 的 label 设为 -100 以忽略 loss 计算
        # pad_token_id = tokenizer.pad_token_id
        # model_inputs["labels"] = [
        #     [(l if l != pad_token_id else -100) for l in label] for label in model_inputs["labels"]
        # ]
        
        return model_inputs

    train_ds = dataset.map(preprocess, batched=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt"),
    )

    print(">>> Start Training...")
    trainer.train()
    
    # 保存时，我们需要把 mixlora config 也保存进去
    model.save_pretrained(training_args.output_dir)
    print(f">>> Model saved to {training_args.output_dir}")

if __name__ == "__main__":
    train()