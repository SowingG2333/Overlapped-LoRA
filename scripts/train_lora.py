import os
import torch
import logging
from dataclasses import dataclass, field
from datasets import load_dataset
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="/root/gpufree-data/hf/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4")
    use_flash_attn: bool = field(default=True, metadata={"help": "Use Flash Attention 2"})
    use_4bit: bool = field(default=True, metadata={"help": "Load model with 4-bit quantization (bitsandbytes)."})

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data json file"})
    max_seq_length: int = field(default=1024)

@dataclass
class LoraArguments:
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma separated list of target modules"}
    )

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    # 1. 设置随机种子
    set_seed(training_args.seed)

    # 2. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right", # SFT 通常 padding 在右边（除非使用 FlashAttn 的特定库，但右侧通用）
        trust_remote_code=True,
        use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # Qwen 等模型有时需要手动指定 pad_token_id
        # tokenizer.pad_token_id = tokenizer.eos_token_id 

    # 3. 加载数据集 (适配 data.py 生成的 json 格式)
    # data format: [{"instruction": "...", "output": "...", "source": "..."}]
    if not os.path.exists(data_args.data_path):
        raise FileNotFoundError(f"Data file not found: {data_args.data_path}")
        
    dataset = load_dataset("json", data_files=data_args.data_path, split="train")

    # 4. 数据预处理函数
    def preprocess_function(examples):
        inputs = examples["instruction"]
        targets = examples["output"]
        
        model_inputs = []
        labels = []
        
        for i in range(len(inputs)):
            # 构建 Prompt (Alpaca 风格)
            # 为了简单和通用，我们使用标准的 Instruction 格式
            prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{inputs[i]}\n\n### Response:\n"
            response = f"{targets[i]}" + tokenizer.eos_token
            
            # Tokenize
            full_text = prompt + response
            
            tokenized = tokenizer(
                full_text,
                truncation=True,
                max_length=data_args.max_seq_length,
                padding=False,
                return_tensors=None
            )
            
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            
            # 构建 Labels (Mask 掉 Prompt 部分，只计算 Response 的 Loss)
            # 先 tokenize prompt 算长度
            prompt_tokenized = tokenizer(
                prompt,
                truncation=True,
                max_length=data_args.max_seq_length,
                padding=False,
                return_tensors=None
            )
            prompt_len = len(prompt_tokenized["input_ids"])
            
            # labels: prompt 部分设为 -100，response 部分保留 input_ids
            label = [-100] * prompt_len + input_ids[prompt_len:]
            
            # 处理截断导致的长度不一致
            if len(label) > len(input_ids): 
                label = label[:len(input_ids)]
            elif len(label) < len(input_ids): # 极少情况
                label = label + [-100] * (len(input_ids) - len(label))
                
            model_inputs.append(input_ids)
            labels.append(label)

        return {
            "input_ids": model_inputs,
            "labels": labels,
            "attention_mask": [[1] * len(x) for x in model_inputs] # 简单mask
        }

    # 处理数据集
    with training_args.main_process_first(desc="dataset map pre-processing"):
        train_dataset = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=4,
            remove_columns=dataset.column_names,
            desc="Running tokenizer on dataset",
        )

    # 5. 加载模型
    print(f"Loading base model: {model_args.model_name_or_path}")
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # 自动判断 Attention 实现方式
    attn_implementation = "eager"
    if model_args.use_flash_attn and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            attn_implementation = "flash_attention_2"
    
    print(f"Using attention implementation: {attn_implementation}")

    # 4bit 量化配置（默认为开启）
    quantization_config = None
    if model_args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation=attn_implementation,
        quantization_config=quantization_config,
    )

    # k-bit 训练前置：使量化模型适配 LoRA 训练
    if model_args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    # 6. 配置 LoRA
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules.split(","),
        lora_dropout=lora_args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 7. 训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, 
            pad_to_multiple_of=8, 
            return_tensors="pt", 
            padding=True
        ),
    )

    # 8. 开始训练
    trainer.train()

    # 9. 保存模型
    trainer.save_model(training_args.output_dir)
    
    # 同时保存 tokenizer 和 adapter config
    # 确保 adapter_model.bin 被保存，这对于后续 Stitch 很重要
    model.save_pretrained(training_args.output_dir) 

if __name__ == "__main__":
    train()