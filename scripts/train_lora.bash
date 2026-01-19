#!/bin/bash

# 基础模型路径
BASE_MODEL="/root/gpufree-data/hf/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4"

# 循环训练 8 个专家
# for i in {0..7}
for i in {6..7}
do
    echo "=================================================="
    echo ">>> Training Expert $i ..."
    echo "=================================================="
    
    # 显存优化技巧：
    # 如果显存不够，减小 batch_size，增大 gradient_accumulation_steps
    
    python /root/gpufree-data/OverlappedLoRA/scripts/train_lora.py \
        --model_name_or_path $BASE_MODEL \
        --data_path "/root/gpufree-data/OverlappedLoRA/data/experts/expert_$i.json" \
        --output_dir "/root/gpufree-data/OverlappedLoRA/model/expert_$i" \
        --num_train_epochs 2 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --learning_rate 2e-4 \
        --weight_decay 0.001 \
        --warmup_ratio 0.03 \
        --logging_steps 10 \
        --save_strategy "no" \
        --lora_r 16 \
        --lora_alpha 32 \
        --bf16 True \
        --seed 42 \
        --use_flash_attn False
        
    echo ">>> Expert $i Finished."
done

echo "All experts trained!"