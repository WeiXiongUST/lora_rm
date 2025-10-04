export NCCL_P2P_DISABLE=1

# LoRA rank 实验列表
LORA_RANKS=(1 4 16 64 256)

# 学习率实验列表 - LoRA通常需要2-10倍的学习率
LEARNING_RATES=(5e-5 7e-5 9e-5 1e-4 3e-4 5e-4 7e-4 1e-3)

echo "=========================================="
echo "开始 Llama-3.2-1B LoRA 实验"
echo "=========================================="

# 循环测试不同 LoRA rank 和学习率组合 - 1B模型
for rank in "${LORA_RANKS[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
        echo "=========================================="
        echo "开始训练 - 模型: Llama-3.2-1B, LoRA rank: $rank, 学习率: $lr"
        echo "=========================================="
        
        # 生成实验名称
        lr_name=$(echo $lr | sed 's/e-0/e-/g' | sed 's/\.//g')
        run_name="llama3.2-1b-rm-lora-r${rank}-lr${lr_name}-bs64"
        
        # 运行训练
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch ./train_rm_with_lora.py \
            --wandb_project "huggingface" \
            --wandb_run_name "$run_name" \
            --model_name "meta-llama/Llama-3.2-1B-Instruct" \
            --learning_rate $lr \
            --lora_r $rank \
            --lora_alpha $((rank * 2)) \
            --lora_dropout 0.05 \
            --output_path "./models/llama3_2_1b_rm_lora_r${rank}_lr${lr_name}" \
            --use_lora True
        
        echo "1B模型 LoRA rank $rank, 学习率 $lr 的训练完成"
        echo "=========================================="
    done
done

echo "Llama-3.2-1B 模型所有 LoRA rank 和学习率实验完成！"
echo ""

echo "=========================================="
echo "开始 Llama-3.2-3B LoRA 实验"
echo "=========================================="

# 循环测试不同 LoRA rank 和学习率组合 - 3B模型
for rank in "${LORA_RANKS[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
        echo "=========================================="
        echo "开始训练 - 模型: Llama-3.2-3B, LoRA rank: $rank, 学习率: $lr"
        echo "=========================================="
        
        # 生成实验名称
        lr_name=$(echo $lr | sed 's/e-0/e-/g' | sed 's/\.//g')
        run_name="llama3.2-3b-rm-lora-r${rank}-lr${lr_name}-bs64"
        
        # 运行训练
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch ./train_rm_with_lora.py \
            --wandb_project "huggingface" \
            --wandb_run_name "$run_name" \
            --model_name "meta-llama/Llama-3.2-3B-Instruct" \
            --learning_rate $lr \
            --lora_r $rank \
            --lora_alpha $((rank * 2)) \
            --lora_dropout 0.05 \
            --output_path "./models/llama3_2_3b_rm_lora_r${rank}_lr${lr_name}" \
            --use_lora True
        
        echo "3B模型 LoRA rank $rank, 学习率 $lr 的训练完成"
        echo "=========================================="
    done
done

echo "所有模型、LoRA rank 和学习率实验完成！"
