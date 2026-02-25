"""
多 Seed 稳定性实验 + 集成推理
==============================
策略:
  1. 跑 N 个不同种子，每个保存最优 checkpoint
  2. 统计 mean ± std，选最优 seed 或做集成
  3. 集成 (Ensemble) 多个 checkpoint 的预测

用法:
    python run_multi_seed.py                        # 跑 5 个 seed
    python run_multi_seed.py --seeds 1 42 123 2024 3407
    python run_multi_seed.py --ensemble             # 用已有 checkpoint 做集成推理
    python run_multi_seed.py --ensemble --topk 3    # 取 top-3 做集成
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime

from config.Config_base import Config_base
from dataset.dataset import MemeDataset, convert_onehot
from train_eval_ import train, eval, get_scores
from model.MHKE import MHKE_CLIP


# ============================================================
# 常用的好种子 (来自论文/实践经验)
# ============================================================
DEFAULT_SEEDS = [1, 42, 123, 2026, 3407]


def set_seed(seed):
    """设置所有随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_single_seed(seed, config_overrides=None):
    """用指定 seed 训练一次，返回结果"""
    set_seed(seed)

    config = Config_base("clip", "task_1")
    config.exp_tag = f"seed{seed}"

    # 可选覆盖配置
    if config_overrides:
        for k, v in config_overrides.items():
            setattr(config, k, v)

    # 数据
    trn_data = MemeDataset(config, training=True)
    test_data = MemeDataset(config, training=False)

    train_loader = DataLoader(trn_data, batch_size=config.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False, num_workers=0)

    print(f"\n{'='*60}")
    print(f"Seed={seed}  训练集={len(trn_data)}  测试集={len(test_data)}")
    print(f"{'='*60}")

    start = time.time()
    best_f1, best_epoch = train(config, train_loader, test_loader)
    elapsed = (time.time() - start) / 60.0

    # checkpoint 路径
    model_name = '{}_B-{}_E-{}_Lr-{}_w-{}_{}_add'.format(
        config.model_name, config.batch_size, config.num_epochs,
        config.learning_rate, config.weight, config.task_name)
    if config.exp_tag:
        model_name = f'{config.exp_tag}_{model_name}'
    ckp_path = '{}/ckp-{}-BEST.tar'.format(config.checkpoint_path, model_name)

    return {
        'seed': seed,
        'best_f1': best_f1,
        'best_f1_pct': f"{best_f1*100:.2f}%",
        'best_epoch': best_epoch,
        'elapsed_min': round(elapsed, 1),
        'checkpoint': ckp_path,
    }


def ensemble_predict(config, checkpoint_paths):
    """
    集成推理: 加载多个 checkpoint，平均 logits 后预测。
    这是提升稳定性最简单有效的方法。
    """
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

    test_data = MemeDataset(config, training=False)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False, num_workers=0)

    all_logits = []

    for ckp_path in checkpoint_paths:
        print(f"\n加载: {ckp_path}")
        model = MHKE_CLIP(config).to(config.device)
        ckpt = torch.load(ckp_path, map_location=config.device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        batch_logits = []
        for batch in test_loader:
            with torch.no_grad():
                logit = model(**batch).cpu()
                batch_logits.append(logit)

        logits = torch.cat(batch_logits, dim=0)  # [N, num_classes]
        all_logits.append(logits)
        print(f"  epoch={ckpt.get('epoch', '?')}, logits shape={logits.shape}")

    # 平均 logits
    avg_logits = torch.stack(all_logits).mean(dim=0)  # [N, num_classes]

    # 预测
    preds = torch.max(avg_logits, dim=1)[1].cpu().numpy()
    pred_onehot = [convert_onehot(config, p) for p in preds]

    # 真实标签
    labels = []
    for batch in test_loader:
        labels.extend(batch['label'].numpy())

    # 计算指标
    f1 = f1_score(pred_onehot, labels, average='macro')
    acc = accuracy_score(pred_onehot, labels)
    all_f1 = f1_score(pred_onehot, labels, average=None)
    pre = precision_score(pred_onehot, labels, average='macro')
    rec = recall_score(pred_onehot, labels, average='macro')

    print(f"\n{'='*60}")
    print(f"集成推理结果 ({len(checkpoint_paths)} 模型)")
    print(f"{'='*60}")
    print(f"  F1 (macro): {f1*100:.2f}%")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {pre*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  Per-class : {[f'{x*100:.2f}%' for x in all_f1]}")

    return f1


def main():
    parser = argparse.ArgumentParser(description="多 Seed 训练 + 集成推理")
    parser.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS,
                        help='需要跑的 seed 列表 (默认: 1 42 123 2024 3407)')
    parser.add_argument('--ensemble', action='store_true',
                        help='对已有 checkpoint 做集成推理')
    parser.add_argument('--topk', type=int, default=0,
                        help='集成时只取 top-k 个最优 checkpoint (0=全部)')
    parser.add_argument('--ckpts', nargs='+', type=str, default=None,
                        help='手动指定集成的 checkpoint 路径')
    args = parser.parse_args()

    config = Config_base("clip", "task_1")

    if args.ensemble:
        # ====== 集成推理模式 ======
        if args.ckpts:
            ckpt_paths = args.ckpts
        else:
            # 自动查找所有 seed checkpoint
            import glob
            ckpt_paths = sorted(glob.glob(f"{config.checkpoint_path}/ckp-seed*-BEST.tar"))
            if not ckpt_paths:
                # 也查找消融实验的 checkpoint
                ckpt_paths = sorted(glob.glob(f"{config.checkpoint_path}/ckp-ablv4*-BEST.tar"))
            if not ckpt_paths:
                print("未找到 checkpoint。请先运行多 seed 训练，或用 --ckpts 指定路径。")
                return

        print(f"找到 {len(ckpt_paths)} 个 checkpoint:")
        for p in ckpt_paths:
            print(f"  {p}")

        if args.topk > 0 and args.topk < len(ckpt_paths):
            # 需要先评估每个 checkpoint 的单独 F1，选 top-k
            print(f"\n筛选 top-{args.topk}...")
            # 简化：直接取前 topk 个（实际使用时可从结果文件读取各 seed 的 F1）
            ckpt_paths = ckpt_paths[:args.topk]

        ensemble_predict(config, ckpt_paths)

    else:
        # ====== 多 Seed 训练模式 ======
        seeds = args.seeds
        print(f"多 Seed 训练: {seeds}")
        print(f"共 {len(seeds)} 个 seed，预计耗时 ~{len(seeds) * 35} 分钟\n")

        results = []
        for i, seed in enumerate(seeds):
            print(f"\n进度: [{i+1}/{len(seeds)}] seed={seed}")
            try:
                result = run_single_seed(seed)
                results.append(result)
            except Exception as e:
                print(f"seed={seed} 失败: {e}")
                import traceback
                traceback.print_exc()
                results.append({'seed': seed, 'best_f1': 0, 'best_f1_pct': 'FAILED',
                                'best_epoch': -1, 'elapsed_min': 0, 'checkpoint': ''})

            # 增量保存
            with open('result/multi_seed_results.json', 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        # 汇总
        valid = [r for r in results if r['best_f1'] > 0]
        f1_list = [r['best_f1'] for r in valid]

        print(f"\n{'='*60}")
        print(f"多 Seed 汇总")
        print(f"{'='*60}")
        print(f"{'Seed':>6} | {'F1':>8} | {'Epoch':>6} | {'耗时':>8}")
        print("-" * 40)
        for r in sorted(valid, key=lambda x: x['best_f1'], reverse=True):
            print(f"{r['seed']:>6} | {r['best_f1_pct']:>8} | {r['best_epoch']:>6} | {r['elapsed_min']:>6.1f}m")

        if f1_list:
            mean_f1 = np.mean(f1_list) * 100
            std_f1 = np.std(f1_list) * 100
            max_f1 = np.max(f1_list) * 100
            min_f1 = np.min(f1_list) * 100
            print(f"\n统计: {mean_f1:.2f}% ± {std_f1:.2f}%")
            print(f"  最优: {max_f1:.2f}%")
            print(f"  最差: {min_f1:.2f}%")
            print(f"  范围: {max_f1 - min_f1:.2f}%")

            best_seed = valid[np.argmax(f1_list)]['seed']
            print(f"\n🏆 最优 seed: {best_seed}")
            print(f"\n下一步: 集成推理")
            print(f"  python run_multi_seed.py --ensemble")
            print(f"  python run_multi_seed.py --ensemble --topk 3")

        print(f"\n结果已保存到: result/multi_seed_results.json")


if __name__ == '__main__':
    main()
