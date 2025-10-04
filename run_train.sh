export NCCL_P2P_DISABLE=1

# 学习率实验列表
LEARNING_RATES=(5e-6 1e-5 2e-5)

echo "=========================================="
echo "开始 Llama-3.2-1B 模型实验"
echo "=========================================="

# 循环测试不同学习率 - 1B模型
for lr in "${LEARNING_RATES[@]}"; do
    echo "=========================================="
    echo "开始训练 - 模型: Llama-3.2-1B, 学习率: $lr"
    echo "=========================================="
    
    # 生成实验名称
    lr_name=$(echo $lr | sed 's/e-0/e-/g' | sed 's/\.//g')
    run_name="llama3.2-1b-rm-lr${lr_name}-bs64"
    
    # 运行训练
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch ./train_rm.py \
        --wandb_project "huggingface" \
        --wandb_run_name "$run_name" \
        --learning_rate $lr \
        --output_path "./models/llama3_2_1b_rm_lr${lr_name}" \
        --model_name "meta-llama/Llama-3.2-1B-Instruct"
    
    echo "1B模型学习率 $lr 的训练完成"
    echo "=========================================="
done

echo "Llama-3.2-1B 模型所有学习率实验完成！"
echo ""

echo "=========================================="
echo "开始 Llama-3.2-3B 模型实验"
echo "=========================================="
LEARNING_RATES=(2e-6 5e-6 1e-5 2e-5)

# 循环测试不同学习率 - 3B模型
for lr in "${LEARNING_RATES[@]}"; do
    echo "=========================================="
    echo "开始训练 - 模型: Llama-3.2-3B, 学习率: $lr"
    echo "=========================================="
    
    # 生成实验名称
    lr_name=$(echo $lr | sed 's/e-0/e-/g' | sed 's/\.//g')
    run_name="llama3.2-3b-rm-lr${lr_name}-bs64"
    
    # 运行训练
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch ./train_rm.py \
        --wandb_project "huggingface" \
        --wandb_run_name "$run_name" \
        --learning_rate $lr \
        --output_path "./models/llama3_2_3b_rm_lr${lr_name}" \
        --model_name "meta-llama/Llama-3.2-3B-Instruct"
    
    echo "3B模型学习率 $lr 的训练完成"
    echo "=========================================="
done

echo "所有模型和学习率实验完成！" 
