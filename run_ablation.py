"""
消融实验脚本 V2 (Ablation Study - V4 Data)
==========================================
基准: main 分支 + V4 数据 + 无正则化 = 80.67% F1
目标: 在 V4 数据基础上逐项测试各技术的贡献

用法:
    python run_ablation.py --all                   # 运行全部实验
    python run_ablation.py --exp baseline           # 运行单个实验
    python run_ablation.py --exp grad_clip rdrop_sigmoid  # 运行多个实验
    python run_ablation.py --list                  # 列出所有实验
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime

# ============================================================
# 实验定义
# ============================================================

EXPERIMENTS = {
    # ====== 基线 ======
    "baseline": {
        "desc": "V4 数据无正则化基准 (应复现 ~80.67%)",
        "overrides": {
            "num_epochs": 10,
        },
    },

    # ====== 单项消融 ======
    "grad_clip": {
        "desc": "梯度裁剪 (max_norm=1.0)",
        "overrides": {
            "use_grad_clip": True,
            "num_epochs": 10,
        },
    },
    "rdrop_sigmoid": {
        "desc": "R-Drop (α=0.5, sigmoid KL, 修复版)",
        "overrides": {
            "rdrop_alpha": 0.5,
            "rdrop_use_sigmoid": True,
            "num_epochs": 20,
            "patience": 5,
        },
    },
    "rdrop_softmax": {
        "desc": "R-Drop (α=0.5, softmax KL, 原版)",
        "overrides": {
            "rdrop_alpha": 0.5,
            "rdrop_use_sigmoid": False,
            "num_epochs": 20,
            "patience": 5,
        },
    },
    "label_smoothing": {
        "desc": "Label Smoothing (ε=0.1)",
        "overrides": {
            "label_smoothing": 0.1,
            "num_epochs": 20,
            "patience": 5,
        },
    },
    "augmentation": {
        "desc": "图像增强 (RandomCrop + ColorJitter + Flip + Rotation)",
        "overrides": {
            "use_augmentation": True,
            "num_epochs": 20,
            "patience": 5,
        },
    },
    "cosine_warmup": {
        "desc": "Cosine Warmup LR 调度器 (warmup 10%)",
        "overrides": {
            "use_scheduler": True,
            "warmup_ratio": 0.1,
            "num_epochs": 20,
            "patience": 5,
        },
    },
    "clf_dropout": {
        "desc": "分类头 Dropout (p=0.1)",
        "overrides": {
            "classifier_dropout": 0.1,
            "num_epochs": 20,
            "patience": 5,
        },
    },

    # ====== 组合实验 ======
    "grad_clip_rdrop": {
        "desc": "梯度裁剪 + R-Drop sigmoid (V3最优组合在V4上测试)",
        "overrides": {
            "use_grad_clip": True,
            "rdrop_alpha": 0.5,
            "rdrop_use_sigmoid": True,
            "num_epochs": 20,
            "patience": 5,
        },
    },
    "best_v3_combo": {
        "desc": "V3 最优配置: all_fixed (全部正则化 + sigmoid 修复)",
        "overrides": {
            "rdrop_alpha": 0.5,
            "rdrop_use_sigmoid": True,
            "label_smoothing": 0.1,
            "use_augmentation": True,
            "use_scheduler": True,
            "warmup_ratio": 0.1,
            "classifier_dropout": 0.1,
            "use_grad_clip": True,
            "num_epochs": 20,
            "patience": 5,
        },
    },
}

# 实验运行顺序
EXPERIMENT_ORDER = [
    "baseline",
    "grad_clip",
    "rdrop_sigmoid",
    "rdrop_softmax",
    "label_smoothing",
    "augmentation",
    "cosine_warmup",
    "clf_dropout",
    "grad_clip_rdrop",
    "best_v3_combo",
]


# ============================================================
# 消融实验基线配置 (V4 数据, 无正则化)
# ============================================================

def get_baseline_config():
    """返回 V4 基线配置 (无任何正则化)"""
    return {
        "rdrop_alpha": 0,
        "rdrop_use_sigmoid": False,
        "label_smoothing": 0,
        "use_augmentation": False,
        "use_scheduler": False,
        "warmup_ratio": 0.1,
        "classifier_dropout": 0,
        "use_grad_clip": False,
        "use_ema": False,
        "num_epochs": 10,
        "patience": 999,   # baseline 不 early stop
        "pad_size": 128,    # V4 数据需要 128
    }


# ============================================================
# 日志配置
# ============================================================

def setup_logging():
    """配置日志"""
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result", "ablation")
    os.makedirs(result_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(result_dir, f"ablation_v4_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"日志文件: {log_file}")
    return logger, result_dir


# ============================================================
# 运行单个实验
# ============================================================

def run_experiment(exp_name, exp_config, logger):
    """运行单个消融实验"""
    from config.Config_base import Config_base
    from dataset.dataset import MemeDataset
    from torch.utils.data import DataLoader
    from train_eval_ import train

    logger.info(f"\n{'='*70}")
    logger.info(f"实验: {exp_name}")
    logger.info(f"描述: {exp_config['desc']}")
    logger.info(f"覆盖参数: {json.dumps(exp_config['overrides'], ensure_ascii=False)}")
    logger.info(f"{'='*70}")

    # 创建配置
    config = Config_base(model_name="MHKE", task_name="task_1")

    # 应用基线配置 (所有正则化关闭)
    baseline = get_baseline_config()
    for key, value in baseline.items():
        setattr(config, key, value)

    # 应用实验覆盖参数
    for key, value in exp_config["overrides"].items():
        setattr(config, key, value)

    # 添加实验标签
    config.exp_tag = f"ablv4_{exp_name}"

    # 打印关键配置
    logger.info("关键配置:")
    for key in ["num_epochs", "patience", "pad_size", "rdrop_alpha", "rdrop_use_sigmoid",
                 "label_smoothing", "use_augmentation", "use_scheduler",
                 "classifier_dropout", "use_grad_clip", "weight_decay"]:
        val = getattr(config, key, "N/A")
        logger.info(f"  {key}={val}")

    # 加载数据
    train_dataset = MemeDataset(config, training=True)
    dev_dataset = MemeDataset(config, training=False)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
    )
    logger.info(f"训练集: {len(train_dataset)} 样本, 测试集: {len(dev_dataset)} 样本")

    # 训练
    start_time = time.time()
    max_score, best_epoch = train(config, train_loader, dev_loader)
    elapsed = (time.time() - start_time) / 60.0

    logger.info(f"\n实验 [{exp_name}] 完成!")
    logger.info(f"  最优 F1: {max_score*100:.2f}%")
    logger.info(f"  最优 Epoch: {best_epoch}")
    logger.info(f"  耗时: {elapsed:.1f} 分钟")

    return {
        "exp_name": exp_name,
        "desc": exp_config["desc"],
        "best_f1": max_score,
        "best_f1_pct": f"{max_score*100:.2f}%",
        "best_epoch": best_epoch,
        "elapsed_min": round(elapsed, 1),
        "overrides": exp_config["overrides"],
    }


# ============================================================
# 结果保存与汇总
# ============================================================

def save_results(results, result_dir, logger):
    """保存结果到 JSON (增量)"""
    result_file = os.path.join(result_dir, "ablation_v4_results.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"结果已保存到: {result_file}")


def print_summary(results, logger):
    """打印实验汇总表"""
    logger.info(f"\n\n{'='*70}")
    logger.info("消融实验汇总 (V4 数据)")
    logger.info(f"{'='*70}")

    # 按 F1 排序
    sorted_results = sorted(results, key=lambda x: x["best_f1"], reverse=True)

    header = f"{'实验':<22}| {'最优F1':>8} | {'Epoch':>6} | {'耗时(min)':>10} | 描述"
    logger.info(header)
    logger.info("-" * 21 + "+" + "-" * 10 + "+" + "-" * 8 + "+" + "-" * 12 + "+" + "-" * 30)

    for r in sorted_results:
        line = f"{r['exp_name']:<20} | {r['best_f1_pct']:>8} | {r['best_epoch']:>6} | {r['elapsed_min']:>10.1f} | {r['desc']}"
        logger.info(line)

    # 与 baseline 对比
    baseline_f1 = None
    for r in results:
        if r["exp_name"] == "baseline":
            baseline_f1 = r["best_f1"]
            break

    if baseline_f1 is not None:
        logger.info(f"\n与 baseline ({baseline_f1*100:.2f}%) 的差异:")
        for r in sorted_results:
            if r["exp_name"] == "baseline":
                continue
            diff = (r["best_f1"] - baseline_f1) * 100
            arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
            logger.info(f"  {r['exp_name']:<20}: {diff:+.2f}% {arrow}")


# ============================================================
# CLI 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="ToxiCN_MM V4 消融实验")
    parser.add_argument("--all", action="store_true", help="运行全部实验")
    parser.add_argument("--exp", nargs="+", help="运行指定实验")
    parser.add_argument("--list", action="store_true", help="列出所有可用实验")
    args = parser.parse_args()

    if args.list:
        print("\n可用实验:")
        print(f"{'名称':<22}| 描述")
        print("-" * 21 + "+" + "-" * 50)
        for name in EXPERIMENT_ORDER:
            exp = EXPERIMENTS[name]
            print(f"{name:<20} | {exp['desc']}")
        print(f"\n共 {len(EXPERIMENT_ORDER)} 个实验")
        return

    if not args.all and not args.exp:
        parser.print_help()
        return

    # 确定要运行的实验
    if args.all:
        exp_names = EXPERIMENT_ORDER
    else:
        exp_names = args.exp
        for name in exp_names:
            if name not in EXPERIMENTS:
                print(f"错误: 未知实验 '{name}'")
                print(f"可用: {', '.join(EXPERIMENT_ORDER)}")
                return

    logger, result_dir = setup_logging()
    logger.info("消融实验开始 (V4 数据)")
    logger.info(f"计划运行 {len(exp_names)} 个实验: {', '.join(exp_names)}")

    # 打印 GPU 信息
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    except Exception:
        pass

    # 逐个运行
    results = []
    for idx, exp_name in enumerate(exp_names):
        logger.info(f"\n进度: [{idx+1}/{len(exp_names)}]")
        try:
            result = run_experiment(exp_name, EXPERIMENTS[exp_name], logger)
            results.append(result)
            save_results(results, result_dir, logger)
        except Exception as e:
            logger.error(f"实验 [{exp_name}] 失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            results.append({
                "exp_name": exp_name,
                "desc": EXPERIMENTS[exp_name]["desc"],
                "best_f1": 0,
                "best_f1_pct": "FAILED",
                "best_epoch": -1,
                "elapsed_min": 0,
                "overrides": EXPERIMENTS[exp_name]["overrides"],
                "error": str(e),
            })
            save_results(results, result_dir, logger)

    # 打印汇总
    print_summary(results, logger)
    logger.info(f"\n全部完成! 结果保存至: {os.path.join(result_dir, 'ablation_v4_results.json')}")


if __name__ == "__main__":
    main()
