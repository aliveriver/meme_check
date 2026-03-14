"""
对比实验脚本：MHKE vs 其他模型
统一使用 seed=2026，V4 数据集
用法:
    python run_comparison.py                       # 运行全部模型
    python run_comparison.py --models MHKE clip    # 只运行指定模型
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.Config_base import Config_base
from dataset.dataset import MemeDataset
from train_eval_ import train, eval
from model.clip import *
from model.vit_roberta import *
from model.MHKE import *


# ============================================================
# 可对比的全部模型
# ============================================================
ALL_MODELS = [
    "MHKE",                    # ⭐ 主模型（ViT+RoBERTa + 知识增强）
    "clip",                    # MHKE_CLIP（ChineseCLIP + 知识增强）
    "vit-roberta",             # 纯 ViT+RoBERTa 基线（无知识增强）
    "MHKE_CrossAttention",     # 序列级交叉注意力
    "MHKE_CrossAttention_V2",  # 堆叠交叉注意力 V2
    "MHKE_ISSUES",             # 融合 ISSUES 论文方法
    "roberta",                 # 纯文本基线
]

SEED = 2026


def set_seed(seed):
    """设置全局随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_data_model_name(model_name):
    """
    确定数据加载使用的 model_name。
    CLIP 类模型需要 CLIPProcessor，其他模型使用 BertTokenizer + ViTExtractor。
    """
    if model_name == "clip":
        return "clip"
    elif model_name == "roberta":
        return "roberta"
    else:
        # MHKE, MHKE_CrossAttention 等都用 MHKE pipeline (BertTokenizer + ViT)
        return "MHKE"


def build_config(model_name, task_name="task_1"):
    """为指定模型构建配置"""
    data_mn = get_data_model_name(model_name)
    config = Config_base(data_mn, task_name)

    # 覆盖关键参数
    config.seed = SEED
    config.model_name = model_name

    # CLIP hidden_dim = 512，其余 768
    if model_name == "clip":
        config.hidden_dim = 512
    else:
        config.hidden_dim = 768

    # 为对比实验打标签，避免覆盖其他训练的 checkpoint
    config.exp_tag = f"cmp_s{SEED}"

    return config


def run_single_model(model_name, task_name="task_1"):
    """运行单个模型的训练 + 评估，返回指标字典"""
    print(f"\n{'='*70}")
    print(f"  开始训练模型: {model_name}  (seed={SEED})")
    print(f"{'='*70}\n")

    # 1. 重新设置随机种子（保证每个模型起点一致）
    set_seed(SEED)

    # 2. 构建 Config
    config = build_config(model_name, task_name)

    # 3. 构建数据集
    trn_data = MemeDataset(config, training=True)
    test_data = MemeDataset(config, training=False)
    print(f"训练集大小: {len(trn_data)}  |  测试集大小: {len(test_data)}")

    train_iter = DataLoader(trn_data, batch_size=int(config.batch_size), shuffle=True)
    test_iter = DataLoader(test_data, batch_size=int(config.batch_size), shuffle=False)

    # 4. 训练
    start_time = time.time()
    best_f1, best_epoch = train(config, train_iter, test_iter)
    train_time = time.time() - start_time

    # 5. 加载最佳 checkpoint 做完整评估
    exp_tag = config.exp_tag
    ckp_name = '{}_B-{}_E-{}_Lr-{}_w-{}_{}_add'.format(
        config.model_name, config.batch_size,
        config.num_epochs, config.learning_rate, config.weight, config.task_name
    )
    if exp_tag:
        ckp_name = f'{exp_tag}_{ckp_name}'

    ckp_path = os.path.join(config.checkpoint_path, f'ckp-{ckp_name}-BEST.tar')

    # 重建模型并加载权重
    set_seed(SEED)
    if model_name == "clip":
        model = MHKE_CLIP(config).to(config.device)
    elif model_name == "vit-roberta":
        model = VitRobertaMemesClassifier(config).to(config.device)
    elif model_name == "roberta":
        model = RobertaClassifier(config).to(config.device)
    elif model_name == "MHKE":
        model = MHKE(config).to(config.device)
    elif model_name == "MHKE_CrossAttention":
        model = MHKE_CrossAttention(config).to(config.device)
    elif model_name == "MHKE_CrossAttention_V2":
        model = MHKE_CrossAttention_V2(config).to(config.device)
    elif model_name == "MHKE_ISSUES":
        model = MHKE_ISSUES(config).to(config.device)
    else:
        raise ValueError(f"未知模型: {model_name}")

    if os.path.exists(ckp_path):
        checkpoint = torch.load(ckp_path, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\n✓ 已加载最佳 checkpoint: {ckp_path}")
    else:
        print(f"\n⚠️ 未找到 checkpoint: {ckp_path}，使用训练结束时的模型权重")

    loss_fn = nn.BCEWithLogitsLoss()
    eval_scores, _ = eval(config, model, loss_fn, test_iter, data_name='TEST')

    result = {
        "model_name": model_name,
        "seed": SEED,
        "best_epoch": best_epoch,
        "best_train_f1": round(best_f1, 4),
        "test_F1": round(eval_scores["F1"], 4),
        "test_accuracy": round(eval_scores["accuracy"], 4),
        "test_precision": round(eval_scores["precision"], 4),
        "test_recall": round(eval_scores["recall"], 4),
        "test_all_f1": [round(f, 4) for f in eval_scores["all_f1"]],
        "train_time_min": round(train_time / 60, 2),
    }

    print(f"\n{'='*70}")
    print(f"  模型 {model_name} 完成 | Test F1={result['test_F1']:.4f} | "
          f"耗时 {result['train_time_min']:.1f} min")
    print(f"{'='*70}\n")

    return result


def print_results_table(results):
    """打印可读的结果对比表格"""
    print("\n")
    print("=" * 90)
    print("  📊 对比实验结果汇总  (seed={}, dataset=V4)".format(SEED))
    print("=" * 90)
    print(f"{'模型':<28} {'F1':>8} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'Epoch':>6} {'Time':>8}")
    print("-" * 90)

    # 按 F1 降序排列
    sorted_results = sorted(results, key=lambda x: x["test_F1"], reverse=True)
    best_f1 = sorted_results[0]["test_F1"] if sorted_results else 0

    for r in sorted_results:
        marker = " ⭐" if r["test_F1"] == best_f1 else ""
        print(f"{r['model_name']:<28} "
              f"{r['test_F1']:>7.4f} "
              f"{r['test_accuracy']:>7.4f} "
              f"{r['test_precision']:>7.4f} "
              f"{r['test_recall']:>7.4f} "
              f"{r['best_epoch']:>6} "
              f"{r['train_time_min']:>6.1f}m"
              f"{marker}")

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="对比实验：MHKE vs 其他模型 (seed=2026, V4)")
    parser.add_argument("--models", nargs="+", default=None,
                        help=f"要对比的模型列表，默认全部。可选: {ALL_MODELS}")
    parser.add_argument("--task", default="task_1", choices=["task_1", "task_2"],
                        help="任务类型 (default: task_1)")
    parser.add_argument("--output", default=None,
                        help="结果保存路径 (默认: result/comparison_seed2026_<timestamp>.json)")
    args = parser.parse_args()

    models_to_run = args.models if args.models else ALL_MODELS

    # 验证模型名称
    for m in models_to_run:
        if m not in ALL_MODELS:
            print(f"❌ 错误: 未知模型 '{m}'")
            print(f"   可选模型: {ALL_MODELS}")
            sys.exit(1)

    print(f"\n{'#'*70}")
    print(f"#  对比实验配置")
    print(f"#  Seed:    {SEED}")
    print(f"#  数据集:  V4 (train/test_data_discription_4.0.json)")
    print(f"#  任务:    {args.task}")
    print(f"#  模型数:  {len(models_to_run)}")
    print(f"#  模型:    {models_to_run}")
    print(f"{'#'*70}\n")

    # ============================================================
    # 依次训练每个模型
    # ============================================================
    all_results = []
    failed_models = []

    for i, model_name in enumerate(models_to_run):
        print(f"\n>>> [{i+1}/{len(models_to_run)}] 即将训练: {model_name}")
        try:
            result = run_single_model(model_name, args.task)
            all_results.append(result)
        except Exception as e:
            print(f"\n❌ 模型 {model_name} 训练失败: {e}")
            import traceback
            traceback.print_exc()
            failed_models.append({"model_name": model_name, "error": str(e)})

    # ============================================================
    # 打印结果表格
    # ============================================================
    if all_results:
        print_results_table(all_results)

    # ============================================================
    # 保存结果到 JSON
    # ============================================================
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join("result", f"comparison_seed{SEED}_{timestamp}.json")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output_data = {
        "experiment": "model_comparison",
        "seed": SEED,
        "dataset": "V4 (data_discription_4.0)",
        "task": args.task,
        "timestamp": datetime.now().isoformat(),
        "results": all_results,
        "failed_models": failed_models,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 结果已保存到: {output_path}")

    if failed_models:
        print(f"\n⚠️  {len(failed_models)} 个模型训练失败:")
        for fm in failed_models:
            print(f"   - {fm['model_name']}: {fm['error']}")


if __name__ == "__main__":
    main()
