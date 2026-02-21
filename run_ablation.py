"""
消融实验脚本 (Ablation Study)
===============================
基准: main 分支 + 3.0 数据 = 80.27% F1
目标: 逐项测试每种正则化技术的贡献

用法:
    python run_ablation.py --all                   # 运行全部实验
    python run_ablation.py --exp baseline rdrop     # 运行指定实验
    python run_ablation.py --list                   # 列出所有可用实验
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from config.Config_base import Config_base
from dataset.dataset import MemeDataset
from train_eval_ import train


# ============================================================
# 实验定义
# ============================================================
# 每个实验定义为 baseline 上的 config 覆盖项
# baseline = 无正则化 (匹配 main 分支行为)

BASELINE_OVERRIDES = {
    # 关闭所有正则化
    'rdrop_alpha': 0,
    'label_smoothing': 0,
    'use_augmentation': False,
    'use_scheduler': False,
    'classifier_dropout': 0,     # Dropout(0) = 无 dropout
    'use_grad_clip': False,
    'use_ema': False,
    # 匹配 main 分支训练设置
    'num_epochs': 10,
    'patience': 999,             # 不触发 Early Stopping
    'freeze_layers': 0,
    'backbone_lr_scale': 1.0,
    'weight_decay': 0.01,        # AdamW 默认值
}

EXPERIMENTS = {
    'baseline': {
        'desc': '无正则化基准 (10 epochs, 应复现 ~80.27%)',
        'overrides': {},  # 纯 baseline
    },
    'rdrop_softmax': {
        'desc': 'R-Drop (α=0.5, softmax KL)',
        'overrides': {
            'rdrop_alpha': 0.5,
            'rdrop_use_sigmoid': False,
            'num_epochs': 20,
            'patience': 5,
        },
    },
    'rdrop_sigmoid': {
        'desc': 'R-Drop (α=0.5, sigmoid KL, 修复版)',
        'overrides': {
            'rdrop_alpha': 0.5,
            'rdrop_use_sigmoid': True,
            'num_epochs': 20,
            'patience': 5,
        },
    },
    'label_smoothing': {
        'desc': 'Label Smoothing (ε=0.1)',
        'overrides': {
            'label_smoothing': 0.1,
            'num_epochs': 20,
            'patience': 5,
        },
    },
    'augmentation': {
        'desc': '图像增强 (RandomCrop + ColorJitter + Flip + Rotation)',
        'overrides': {
            'use_augmentation': True,
            'num_epochs': 20,
            'patience': 5,
        },
    },
    'cosine_warmup': {
        'desc': 'Cosine Warmup LR 调度器 (warmup 10%)',
        'overrides': {
            'use_scheduler': True,
            'warmup_ratio': 0.1,
            'num_epochs': 20,
            'patience': 5,
        },
    },
    'clf_dropout': {
        'desc': '分类头 Dropout (p=0.1)',
        'overrides': {
            'classifier_dropout': 0.1,
            'num_epochs': 20,
            'patience': 5,
        },
    },
    'grad_clip': {
        'desc': '梯度裁剪 (max_norm=1.0)',
        'overrides': {
            'use_grad_clip': True,
            'num_epochs': 20,
            'patience': 5,
        },
    },
    'all_current': {
        'desc': '全部正则化 (当前 feat/attention 配置, 应复现 ~80.22%)',
        'overrides': {
            'rdrop_alpha': 0.5,
            'rdrop_use_sigmoid': False,
            'label_smoothing': 0.1,
            'use_augmentation': True,
            'use_scheduler': True,
            'warmup_ratio': 0.1,
            'classifier_dropout': 0.1,
            'use_grad_clip': True,
            'num_epochs': 20,
            'patience': 5,
        },
    },
    'all_fixed': {
        'desc': '全部正则化 + R-Drop sigmoid 修复',
        'overrides': {
            'rdrop_alpha': 0.5,
            'rdrop_use_sigmoid': True,
            'label_smoothing': 0.1,
            'use_augmentation': True,
            'use_scheduler': True,
            'warmup_ratio': 0.1,
            'classifier_dropout': 0.1,
            'use_grad_clip': True,
            'num_epochs': 20,
            'patience': 5,
        },
    },
}


# ============================================================
# 日志设置
# ============================================================

def setup_logging(log_dir):
    """设置日志系统：同时输出到文件和控制台"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'ablation_{timestamp}.log')

    # 创建 logger
    logger = logging.getLogger('ablation')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # 文件 handler
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)

    # 控制台 handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"日志文件: {log_file}")
    return logger, log_file


# ============================================================
# 实验运行器
# ============================================================

def create_config(exp_name, overrides):
    """创建实验专属的 Config, 先应用 baseline 默认值, 再应用实验覆盖"""
    config = Config_base("clip", "task_1")

    # 1. 应用 baseline (关闭所有正则化)
    for key, val in BASELINE_OVERRIDES.items():
        setattr(config, key, val)

    # 2. 应用实验特定覆盖
    for key, val in overrides.items():
        setattr(config, key, val)

    # 3. 设置实验标签 (用于区分结果文件)
    config.exp_tag = f'ablation_{exp_name}'

    return config


def run_single_experiment(exp_name, exp_info, logger):
    """运行单个消融实验"""
    logger.info(f"\n{'='*70}")
    logger.info(f"实验: {exp_name}")
    logger.info(f"描述: {exp_info['desc']}")
    logger.info(f"覆盖参数: {json.dumps(exp_info['overrides'], ensure_ascii=False, default=str)}")
    logger.info(f"{'='*70}")

    # 设置随机种子 (确保可复现)
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    # 创建 config
    config = create_config(exp_name, exp_info['overrides'])

    # 打印关键配置
    logger.info(f"关键配置:")
    logger.info(f"  epochs={config.num_epochs}, patience={config.patience}")
    logger.info(f"  rdrop_alpha={getattr(config, 'rdrop_alpha', 0)}")
    logger.info(f"  rdrop_use_sigmoid={getattr(config, 'rdrop_use_sigmoid', False)}")
    logger.info(f"  label_smoothing={getattr(config, 'label_smoothing', 0)}")
    logger.info(f"  use_augmentation={getattr(config, 'use_augmentation', False)}")
    logger.info(f"  use_scheduler={getattr(config, 'use_scheduler', False)}")
    logger.info(f"  classifier_dropout={getattr(config, 'classifier_dropout', 0)}")
    logger.info(f"  use_grad_clip={getattr(config, 'use_grad_clip', True)}")
    logger.info(f"  weight_decay={getattr(config, 'weight_decay', 0.01)}")

    # 加载数据
    trn_data = MemeDataset(config, training=True)
    test_data = MemeDataset(config, training=False)
    logger.info(f"训练集: {len(trn_data)} 样本, 测试集: {len(test_data)} 样本")

    train_iter = DataLoader(trn_data, batch_size=int(config.batch_size), shuffle=False)
    test_iter = DataLoader(test_data, batch_size=int(config.batch_size), shuffle=False)

    # 运行训练
    start_time = time.time()
    best_f1, best_epoch = train(config, train_iter, test_iter)
    elapsed = time.time() - start_time

    logger.info(f"\n实验 [{exp_name}] 完成!")
    logger.info(f"  最优 F1: {best_f1*100:.2f}%")
    logger.info(f"  最优 Epoch: {best_epoch}")
    logger.info(f"  耗时: {elapsed/60:.1f} 分钟")

    return {
        'exp_name': exp_name,
        'desc': exp_info['desc'],
        'best_f1': best_f1,
        'best_f1_pct': f"{best_f1*100:.2f}%",
        'best_epoch': best_epoch,
        'elapsed_min': round(elapsed / 60, 1),
        'overrides': exp_info['overrides'],
    }


def print_summary(results, logger):
    """打印汇总结果表"""
    logger.info(f"\n\n{'='*70}")
    logger.info("消融实验汇总")
    logger.info(f"{'='*70}")
    logger.info(f"{'实验':<20} | {'最优F1':>8} | {'Epoch':>6} | {'耗时(min)':>10} | 描述")
    logger.info(f"{'-'*20}-+-{'-'*8}-+-{'-'*6}-+-{'-'*10}-+-{'-'*30}")

    # 按 F1 排序
    sorted_results = sorted(results, key=lambda x: x['best_f1'], reverse=True)
    for r in sorted_results:
        logger.info(
            f"{r['exp_name']:<20} | {r['best_f1_pct']:>8} | {r['best_epoch']:>6} | "
            f"{r['elapsed_min']:>10} | {r['desc']}"
        )

    logger.info(f"{'='*70}")

    # 与 baseline 对比
    baseline_f1 = None
    for r in results:
        if r['exp_name'] == 'baseline':
            baseline_f1 = r['best_f1']
            break

    if baseline_f1 is not None:
        logger.info(f"\n与 baseline ({baseline_f1*100:.2f}%) 的差异:")
        for r in sorted_results:
            if r['exp_name'] == 'baseline':
                continue
            diff = (r['best_f1'] - baseline_f1) * 100
            symbol = '↑' if diff > 0 else '↓' if diff < 0 else '→'
            logger.info(f"  {r['exp_name']:<20}: {diff:+.2f}% {symbol}")


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='消融实验 (Ablation Study)')
    parser.add_argument('--list', action='store_true', help='列出所有可用实验')
    parser.add_argument('--all', action='store_true', help='运行全部实验')
    parser.add_argument('--exp', nargs='+', help='运行指定实验 (空格分隔)')
    args = parser.parse_args()

    # 列出实验
    if args.list:
        print("\n可用实验:")
        print(f"{'名称':<20} | 描述")
        print(f"{'-'*20}-+-{'-'*50}")
        for name, info in EXPERIMENTS.items():
            print(f"{name:<20} | {info['desc']}")
        print(f"\n共 {len(EXPERIMENTS)} 个实验")
        return

    # 确定要运行的实验
    if args.all:
        exp_names = list(EXPERIMENTS.keys())
    elif args.exp:
        exp_names = args.exp
        # 验证实验名称
        for name in exp_names:
            if name not in EXPERIMENTS:
                print(f"错误: 未知实验 '{name}'")
                print(f"可用实验: {', '.join(EXPERIMENTS.keys())}")
                return
    else:
        parser.print_help()
        return

    # 设置日志
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result', 'ablation')
    logger, log_file = setup_logging(log_dir)

    logger.info(f"消融实验开始")
    logger.info(f"计划运行 {len(exp_names)} 个实验: {', '.join(exp_names)}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # 逐个运行实验
    results = []
    for i, exp_name in enumerate(exp_names):
        logger.info(f"\n进度: [{i+1}/{len(exp_names)}]")
        try:
            result = run_single_experiment(exp_name, EXPERIMENTS[exp_name], logger)
            results.append(result)

            # 增量保存结果 (防止中途崩溃丢失)
            results_file = os.path.join(log_dir, 'ablation_results.json')
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"结果已保存到: {results_file}")

        except Exception as e:
            logger.error(f"实验 [{exp_name}] 失败: {e}", exc_info=True)
            results.append({
                'exp_name': exp_name,
                'desc': EXPERIMENTS[exp_name]['desc'],
                'best_f1': 0,
                'best_f1_pct': 'FAILED',
                'best_epoch': -1,
                'elapsed_min': 0,
                'overrides': EXPERIMENTS[exp_name]['overrides'],
                'error': str(e),
            })

    # 打印汇总
    print_summary(results, logger)

    # 最终保存
    results_file = os.path.join(log_dir, 'ablation_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"\n全部完成! 结果保存至: {results_file}")
    logger.info(f"详细日志: {log_file}")


if __name__ == '__main__':
    main()
