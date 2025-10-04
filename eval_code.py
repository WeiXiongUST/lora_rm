#!/usr/bin/env python3
"""
评估奖励模型在test set上的性能
支持评估单个模型或批量评估多个模型
"""

import argparse
import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils import PaddingStrategy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import json
from tqdm import tqdm

@dataclass
class EvalArguments:
    model_path: str = None
    test_dataset_path: str = "weqweasdas/ultrafeedback_binarized_processed"
    max_length: int = 4096
    batch_size: int = 1
    device: str = "auto"
    output_file: Optional[str] = None

def tokenize_sample(sample, tokenizer, max_length=4096):
    """对单个样本进行tokenization"""
    sample['positive'] = tokenizer.apply_chat_template(
        sample['chosen'], tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
    sample['negative'] = tokenizer.apply_chat_template(
        sample['rejected'], tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
    
    tokenized_pos = tokenizer(sample['positive'], truncation=True, max_length=max_length)
    tokenized_neg = tokenizer(sample['negative'], truncation=True, max_length=max_length)
    
    sample["input_ids_j"] = tokenized_pos["input_ids"]
    sample["attention_mask_j"] = tokenized_pos["attention_mask"]
    sample["input_ids_k"] = tokenized_neg["input_ids"]
    sample["attention_mask_k"] = tokenized_neg["attention_mask"]
    return sample

# 注意：使用逐样本处理，不需要数据整理器

# 注意：指标计算直接在评估循环中进行

def evaluate_model(model, tokenizer, test_dataset, args):
    """评估单个模型"""
    print(f"开始评估模型: {args.model_path}")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    model = model.to(device)
    model.eval()
    
    # 注意：使用逐样本处理，不需要数据整理器
    
    # 准备数据
    print("准备测试数据...")
    test_data = test_dataset.map(
        lambda x: tokenize_sample(x, tokenizer, args.max_length), 
        num_proc=8
    )
    
    # 逐样本评估 (batch_size=1)
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    num_samples = 0
    
    print(f"开始逐样本评估 (batch_size=1，避免padding影响)")
    
    with torch.no_grad():
        for i in tqdm(range(len(test_data)), desc="评估进度"):
            # 获取单个样本
            sample = test_data[i]
            
            # 准备正负样本
            pos_input_ids = torch.tensor(sample["input_ids_j"]).unsqueeze(0).to(device)
            pos_attention_mask = torch.tensor(sample["attention_mask_j"]).unsqueeze(0).to(device)
            neg_input_ids = torch.tensor(sample["input_ids_k"]).unsqueeze(0).to(device)
            neg_attention_mask = torch.tensor(sample["attention_mask_k"]).unsqueeze(0).to(device)
            
            # 分别计算正负样本的得分
            pos_outputs = model(input_ids=pos_input_ids, attention_mask=pos_attention_mask)
            neg_outputs = model(input_ids=neg_input_ids, attention_mask=neg_attention_mask)
            
            pos_score = pos_outputs.logits.squeeze(-1).item()
            neg_score = neg_outputs.logits.squeeze(-1).item()
            
            # 计算loss (与训练时一致)
            loss = -torch.nn.functional.logsigmoid(torch.tensor(pos_score - neg_score)).item()
            total_loss += loss
            num_samples += 1
            
            # 收集预测结果
            all_predictions.append((pos_score, neg_score))
            all_labels.append(1)  # 正样本应该得分更高
    
    # 计算最终指标
    avg_loss = total_loss / num_samples
    
    # 合并所有预测
    pos_scores = np.array([pred[0] for pred in all_predictions])
    neg_scores = np.array([pred[1] for pred in all_predictions])
    
    # 计算准确率
    accuracy = np.sum(pos_scores > neg_scores) / len(pos_scores)
    
    # 计算其他指标
    score_diff = pos_scores - neg_scores
    mean_score_diff = np.mean(score_diff)
    std_score_diff = np.std(score_diff)
    
    results = {
        "model_path": args.model_path,
        "test_samples": len(test_data),
        "avg_loss": avg_loss,
        "accuracy": accuracy,
        "mean_score_difference": mean_score_diff,
        "std_score_difference": std_score_diff,
        "positive_scores_mean": np.mean(pos_scores),
        "negative_scores_mean": np.mean(neg_scores),
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(description="评估奖励模型")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--test_dataset_path", type=str, default="weqweasdas/ultrafeedback_binarized_processed", help="测试数据集路径")
    parser.add_argument("--max_length", type=int, default=4096, help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=1, help="批处理大小 (固定为1以避免padding影响)")
    parser.add_argument("--device", type=str, default="auto", help="设备 (auto/cuda/cpu)")
    parser.add_argument("--output_file", type=str, default=None, help="输出结果文件")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("奖励模型评估脚本")
    print("=" * 60)
    print(f"模型路径: {args.model_path}")
    print(f"测试数据集: {args.test_dataset_path}")
    print(f"最大长度: {args.max_length}")
    print(f"批处理大小: {args.batch_size}")
    print(f"设备: {args.device}")
    print("=" * 60)
    
    # 加载模型和tokenizer
    print("加载模型和tokenizer...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    tokenizer.pad_token = "<|finetune_right_pad_id|>"
    tokenizer.pad_token_id = 128004
    tokenizer.model_max_length = args.max_length
    
    # 加载测试数据
    print("加载测试数据...")
    test_dataset = load_dataset(args.test_dataset_path, split="test").shuffle(seed=42)
    print(f"测试样本数量: {len(test_dataset)}")
    
    # 评估模型
    results = evaluate_model(model, tokenizer, test_dataset, args)
    
    # 打印结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"模型路径: {results['model_path']}")
    print(f"测试样本数: {results['test_samples']}")
    print(f"平均Loss: {results['avg_loss']:.6f}")
    print(f"准确率: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"正样本平均得分: {results['positive_scores_mean']:.4f}")
    print(f"负样本平均得分: {results['negative_scores_mean']:.4f}")
    print(f"得分差异均值: {results['mean_score_difference']:.4f}")
    print(f"得分差异标准差: {results['std_score_difference']:.4f}")
    print("=" * 60)
    
    # 保存结果
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"结果已保存到: {args.output_file}")
    
    return results

if __name__ == "__main__":
    main()
