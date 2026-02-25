"""
分析 checkpoint (.tar) 文件中的模型参数统计
============================================
用法:
    python analyze_checkpoint.py <path_to_tar>
    python analyze_checkpoint.py saved_dict/ckp-ablv4_grad_clip_rdrop_clip_B-16_E-20_Lr-1e-05_w-0.5_task_1_add-BEST.tar

分析内容:
  1. 基本信息 (epoch, 参数总数)
  2. 各层参数统计 (均值、标准差、范数、最大值)
  3. 分类头权重分析
  4. 参数健康度检查 (NaN/Inf/死神经元)
"""

import sys
import torch
import numpy as np
from collections import defaultdict


def analyze_checkpoint(ckp_path):
    print(f"\n{'='*70}")
    print(f"Checkpoint 分析: {ckp_path}")
    print(f"{'='*70}")

    # 加载 checkpoint
    ckpt = torch.load(ckp_path, map_location='cpu')

    # 基本信息
    print(f"\n📋 基本信息:")
    print(f"  保存的 keys: {list(ckpt.keys())}")
    if 'epoch' in ckpt:
        print(f"  最佳 Epoch: {ckpt['epoch']}")

    state_dict = ckpt.get('model_state_dict', ckpt)

    total_params = sum(p.numel() for p in state_dict.values())
    print(f"  总参数量: {total_params:,}")

    # 按模块分组统计
    print(f"\n📊 各模块参数统计:")
    print(f"{'模块':<45} | {'形状':<25} | {'均值':>10} | {'标准差':>10} | {'L2范数':>10} | {'最大值':>10}")
    print("-" * 44 + "-+-" + "-" * 24 + "-+-" + "-" * 10 + "-+-" + "-" * 10 + "-+-" + "-" * 10 + "-+-" + "-" * 10)

    module_stats = defaultdict(lambda: {'params': 0, 'l2_sum': 0.0})

    classifier_layers = {}
    has_issues = []

    for name, param in state_dict.items():
        p = param.float()
        mean = p.mean().item()
        std = p.std().item()
        l2 = p.norm(2).item()
        max_val = p.abs().max().item()

        # 检查异常
        if torch.isnan(p).any():
            has_issues.append((name, "含 NaN"))
        if torch.isinf(p).any():
            has_issues.append((name, "含 Inf"))
        if p.numel() > 10 and (p == 0).sum().item() / p.numel() > 0.5:
            zero_pct = (p == 0).sum().item() / p.numel() * 100
            has_issues.append((name, f"{zero_pct:.1f}% 为零"))

        # 收集分类头参数
        if 'classifier' in name:
            classifier_layers[name] = p

        # 模块分组
        module = name.split('.')[0]
        module_stats[module]['params'] += p.numel()
        module_stats[module]['l2_sum'] += l2 ** 2

        # 只打印关键层 (分类头 + embeddings + layernorm + projection)
        is_key = any(kw in name for kw in ['classifier', 'projection', 'embeddings.position',
                                            'layernorm', 'layer_norm', 'pooler'])
        if is_key:
            shape_str = str(list(param.shape))
            print(f"{name:<45} | {shape_str:<25} | {mean:>10.6f} | {std:>10.6f} | {l2:>10.4f} | {max_val:>10.6f}")

    # 模块汇总
    print(f"\n📦 模块汇总:")
    print(f"{'模块':<20} | {'参数量':>12} | {'L2范数':>12}")
    print("-" * 19 + "-+-" + "-" * 12 + "-+-" + "-" * 12)
    for module, stats in sorted(module_stats.items()):
        l2_total = np.sqrt(stats['l2_sum'])
        print(f"{module:<20} | {stats['params']:>12,} | {l2_total:>12.4f}")

    # 分类头详细分析
    if classifier_layers:
        print(f"\n🎯 分类头 (Classifier) 详细分析:")
        for name, p in classifier_layers.items():
            print(f"\n  {name}  shape={list(p.shape)}")
            print(f"    均值={p.mean().item():.6f}  标准差={p.std().item():.6f}")
            print(f"    最小={p.min().item():.6f}  最大={p.max().item():.6f}")
            if p.dim() == 2:
                # 权重矩阵：按输出类别分析
                for i in range(p.shape[0]):
                    row = p[i]
                    print(f"    类别{i}: 均值={row.mean().item():.6f}  标准差={row.std().item():.6f}  "
                          f"L2={row.norm(2).item():.4f}  活跃特征={int((row.abs() > 0.01).sum().item())}/{row.numel()}")

    # 健康检查
    print(f"\n🏥 健康检查:")
    if has_issues:
        for name, issue in has_issues:
            print(f"  ⚠ {name}: {issue}")
    else:
        print(f"  ✅ 所有参数正常 (无 NaN/Inf/大量死神经元)")

    # 与 run.py 当前配置对比
    print(f"\n💡 提示:")
    print(f"  此 checkpoint 仅保存了 model_state_dict 和 epoch。")
    print(f"  训练超参数 (lr, rdrop_alpha 等) 未保存在 checkpoint 中。")
    print(f"  根据消融日志，此模型使用的配置为:")
    print(f"    model_name=clip, batch_size=16, lr=1e-5")
    print(f"    use_grad_clip=True, rdrop_alpha=0.5, rdrop_use_sigmoid=True")
    print(f"    freeze_layers=0, weight_decay=0.01")

    print(f"\n{'='*70}")
    print(f"分析完成")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # 自动查找可能的 checkpoint
        import glob
        candidates = glob.glob("saved_dict/ckp-*BEST*.tar") + glob.glob("checkpoints/ckp-*BEST*.tar")
        if candidates:
            print("找到以下 checkpoint:")
            for i, c in enumerate(candidates):
                print(f"  [{i}] {c}")
            ckp_path = candidates[0]
            print(f"\n自动使用: {ckp_path}")
        else:
            print("用法: python analyze_checkpoint.py <path_to_tar>")
            print("\n在 saved_dict/ 和 checkpoints/ 中未找到 checkpoint。")
            print("请指定 .tar 文件路径，例如:")
            print("  python analyze_checkpoint.py saved_dict/ckp-ablv4_grad_clip_rdrop_clip_B-16_E-20_Lr-1e-05_w-0.5_task_1_add-BEST.tar")
            sys.exit(1)
    else:
        ckp_path = sys.argv[1]

    analyze_checkpoint(ckp_path)
