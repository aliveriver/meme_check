"""
消融实验脚本 V3 (Ablation Study - V4 Data, CLIP Model)
======================================================
模型: MHKE_CLIP (ChineseCLIP, model_name="clip")
基准: V4 数据 + 无正则化
目标: 在 V4 数据 + CLIP 模型基础上, 测试有效技术的贡献

已移除的无效实验 (基于 V2 结果):
  - grad_clip 单独: +0.04% (微乎其微)
  - rdrop_sigmoid:   +0.01% (无提升)
  - rdrop_softmax:   +0.16% (微小)
  - augmentation:    -0.02% (负效果)

用法:
    python run_ablation.py --all                   # 运行全部实验
    python run_ablation.py --exp baseline           # 运行单个实验
    python run_ablation.py --exp label_smoothing clf_dropout  # 运行多个实验
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
# 全局配置
# ============================================================
MODEL_NAME = "clip"          # 使用 MHKE_CLIP (ChineseCLIP) 模型
TASK_NAME = "task_1"
DEFAULT_BATCH_SIZE = 32      # 默认 batch_size
RDROP_BATCH_SIZE = 32        # R-Drop 需要双前向, 降低 batch_size 防 OOM

# ============================================================
# 实验定义 (仅保留有效实验)
# ============================================================

EXPERIMENTS = {
    # ====== 基线 ======
    "baseline": {
        "desc": "V4 数据 + CLIP 无正则化基准",
        "overrides": {
            "num_epochs": 10,
        },
    },

    # ====== 单项消融 (仅保留 V2 中 >+0.5% 的技术) ======
    "label_smoothing": {
        "desc": "Label Smoothing (ε=0.1)",
        "overrides": {
            "label_smoothing": 0.1,
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
        "desc": "梯度裁剪 + R-Drop sigmoid (V2最优组合)",
        "overrides": {
            "use_grad_clip": True,
            "rdrop_alpha": 0.5,
            "rdrop_use_sigmoid": True,
            "batch_size": RDROP_BATCH_SIZE,   # R-Drop 双前向, 防 OOM
            "num_epochs": 20,
            "patience": 5,
        },
    },
    "best_v3_combo": {
        "desc": "全部正则化组合 (label_smooth + scheduler + dropout + grad_clip + rdrop)",
        "overrides": {
            "rdrop_alpha": 0.5,
            "rdrop_use_sigmoid": True,
            "label_smoothing": 0.1,
            "use_augmentation": False,         # V2 证实无效, 不再启用
            "use_scheduler": True,
            "warmup_ratio": 0.1,
            "classifier_dropout": 0.1,
            "use_grad_clip": True,
            "batch_size": RDROP_BATCH_SIZE,   # R-Drop 双前向, 防 OOM
            "num_epochs": 20,
            "patience": 5,
        },
    },
}

# 实验运行顺序
EXPERIMENT_ORDER = [
    "baseline",
    "label_smoothing",
    "cosine_warmup",
    "clf_dropout",
    "grad_clip_rdrop",
    "best_v3_combo",
]


# ============================================================
# 消融实验基线配置 (V4 数据, CLIP 模型, 无正则化)
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
        "batch_size": DEFAULT_BATCH_SIZE,
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
    import torch

    logger.info(f"\n{'='*70}")
    logger.info(f"实验: {exp_name}")
    logger.info(f"描述: {exp_config['desc']}")
    logger.info(f"覆盖参数: {json.dumps(exp_config['overrides'], ensure_ascii=False)}")
    logger.info(f"{'='*70}")

    # 创建配置 — 使用 CLIP 模型
    config = Config_base(model_name=MODEL_NAME, task_name=TASK_NAME)

    # 应用基线配置 (所有正则化关闭)
    baseline = get_baseline_config()
    for key, value in baseline.items():
        setattr(config, key, value)

    # 应用实验覆盖参数
    for key, value in exp_config["overrides"].items():
        setattr(config, key, value)

    # 添加实验标签
    config.exp_tag = f"ablv4_{exp_name}"

    # ====== 详细日志: 模型 & 配置 ======
    logger.info(f"模型: {config.model_name} (train_eval_ 分发 → "
                f"{'MHKE_CLIP (ChineseCLIP)' if config.model_name == 'clip' else 'MHKE (ViT+RoBERTa)'})")
    logger.info(f"batch_size: {config.batch_size}")
    use_rdrop = getattr(config, 'rdrop_alpha', 0) > 0
    if use_rdrop:
        logger.info(f"⚠ R-Drop 启用 → 双前向传播, 等效显存 batch={config.batch_size * 2}")

    logger.info("关键配置:")
    for key in ["model_name", "num_epochs", "patience", "pad_size", "batch_size",
                 "learning_rate", "weight_decay", "rdrop_alpha", "rdrop_use_sigmoid",
                 "label_smoothing", "use_augmentation", "use_scheduler", "warmup_ratio",
                 "classifier_dropout", "use_grad_clip", "freeze_layers"]:
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
    logger.info(f"训练 steps/epoch: {len(train_loader)}, 验证 steps/epoch: {len(dev_loader)}")

    # 打印模型参数统计 (捕获 train() 内部的 print 输出)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # 训练
    start_time = time.time()
    max_score, best_epoch = train(config, train_loader, dev_loader)
    elapsed = (time.time() - start_time) / 60.0

    # 记录显存峰值
    peak_mem = ""
    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
        peak_mem = f", 显存峰值: {peak_gb:.2f} GiB"

    logger.info(f"\n实验 [{exp_name}] 完成!")
    logger.info(f"  模型: {config.model_name}")
    logger.info(f"  最优 F1: {max_score*100:.2f}%")
    logger.info(f"  最优 Epoch: {best_epoch}")
    logger.info(f"  耗时: {elapsed:.1f} 分钟{peak_mem}")

    return {
        "exp_name": exp_name,
        "desc": exp_config["desc"],
        "model_name": config.model_name,
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
    logger.info(f"消融实验汇总 (V4 数据, 模型: {MODEL_NAME})")
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
    logger.info(f"消融实验开始 (V4 数据, 模型: {MODEL_NAME})")
    logger.info(f"模型分发: {MODEL_NAME} → {'MHKE_CLIP (ChineseCLIP)' if MODEL_NAME == 'clip' else 'MHKE (ViT+RoBERTa)'}")
    logger.info(f"默认 batch_size: {DEFAULT_BATCH_SIZE}, R-Drop batch_size: {RDROP_BATCH_SIZE}")
    logger.info(f"计划运行 {len(exp_names)} 个实验: {', '.join(exp_names)}")

    # 打印 GPU 信息
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GiB)")
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
                "model_name": MODEL_NAME,
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
