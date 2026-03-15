"""
MHKE 消融实验脚本 (Ablation Study)
====================================
模型: MHKE_CLIP (ChineseCLIP + 知识增强)
数据: V4 (data_discription_4.0)
Seed: 2026

两个消融维度:
  A. 知识增强: full / no_knowledge / text_desc_only / meme_desc_only
  B. 防过拟合: R-Drop / 梯度裁剪 / Label Smoothing / Cosine Warmup / Classifier Dropout

用法:
    python run_ablation_mhke.py --all                           # 运行全部实验
    python run_ablation_mhke.py --exp baseline no_knowledge     # 运行指定实验
    python run_ablation_mhke.py --list                          # 列出所有实验
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
SEED = 2026
DEFAULT_BATCH_SIZE = 16      # MHKE_CLIP 默认 batch_size
RDROP_BATCH_SIZE = 16        # R-Drop 双前向, 防 OOM

# ============================================================
# 基线配置 (无任何正则化, 完整知识增强)
# ============================================================

def get_baseline_config():
    """返回 MHKE_CLIP 基线配置 (无正则化, 完整知识)"""
    return {
        # 防过拟合参数全部关闭
        "rdrop_alpha": 0,
        "rdrop_use_sigmoid": False,
        "label_smoothing": 0,
        "use_augmentation": False,
        "use_scheduler": False,
        "warmup_ratio": 0.1,
        "classifier_dropout": 0,
        "use_grad_clip": False,
        "use_ema": False,
        # 知识增强默认完整
        "knowledge_mode": "full",
        # 训练参数
        "num_epochs": 10,
        "patience": 999,        # baseline 不 early stop
        "pad_size": 128,        # V4 数据
        "batch_size": DEFAULT_BATCH_SIZE,
        "freeze_layers": 0,
    }

# ============================================================
# 实验定义
# ============================================================

EXPERIMENTS = {
    # ===========================
    # A. 知识增强消融 (4 组)
    # ===========================
    "baseline": {
        "desc": "基线: 完整知识增强 + 无正则化",
        "category": "知识增强",
        "overrides": {},
    },
    "no_knowledge": {
        "desc": "去除全部知识增强 (退化为纯 ChineseCLIP)",
        "category": "知识增强",
        "overrides": {
            "knowledge_mode": "no_knowledge",
        },
    },
    "text_desc_only": {
        "desc": "仅文本描述知识 (text_description)",
        "category": "知识增强",
        "overrides": {
            "knowledge_mode": "text_desc_only",
        },
    },
    "meme_desc_only": {
        "desc": "仅模因描述知识 (meme_description, w=0.5)",
        "category": "知识增强",
        "overrides": {
            "knowledge_mode": "meme_desc_only",
        },
    },

    # ===========================
    # B. 防过拟合方法消融 (6 组)
    # ===========================
    "grad_clip": {
        "desc": "梯度裁剪 (max_norm=1.0)",
        "category": "防过拟合",
        "overrides": {
            "use_grad_clip": True,
            "num_epochs": 20,
            "patience": 5,
        },
    },
    "rdrop": {
        "desc": "R-Drop (α=0.5, sigmoid 模式)",
        "category": "防过拟合",
        "overrides": {
            "rdrop_alpha": 0.5,
            "rdrop_use_sigmoid": True,
            "batch_size": RDROP_BATCH_SIZE,
            "num_epochs": 20,
            "patience": 5,
        },
    },
    "grad_clip_rdrop": {
        "desc": "梯度裁剪 + R-Drop (CLIP 消融最优组合)",
        "category": "防过拟合",
        "overrides": {
            "use_grad_clip": True,
            "rdrop_alpha": 0.5,
            "rdrop_use_sigmoid": True,
            "batch_size": RDROP_BATCH_SIZE,
            "num_epochs": 20,
            "patience": 5,
        },
    },
    "label_smoothing": {
        "desc": "Label Smoothing (ε=0.1)",
        "category": "防过拟合",
        "overrides": {
            "label_smoothing": 0.1,
            "num_epochs": 20,
            "patience": 5,
        },
    },
    "cosine_warmup": {
        "desc": "Cosine Warmup LR 调度器 (warmup 10%)",
        "category": "防过拟合",
        "overrides": {
            "use_scheduler": True,
            "warmup_ratio": 0.1,
            "num_epochs": 20,
            "patience": 5,
        },
    },
    "clf_dropout": {
        "desc": "分类头 Dropout (p=0.3)",
        "category": "防过拟合",
        "overrides": {
            "classifier_dropout": 0.3,
            "num_epochs": 20,
            "patience": 5,
        },
    },

    # ===========================
    # C. 最优组合 (1 组)
    # ===========================
    "best_combo": {
        "desc": "最优组合: 完整知识 + 梯度裁剪 + R-Drop + 早停",
        "category": "最优组合",
        "overrides": {
            "knowledge_mode": "full",
            "use_grad_clip": True,
            "rdrop_alpha": 0.5,
            "rdrop_use_sigmoid": True,
            "batch_size": RDROP_BATCH_SIZE,
            "num_epochs": 20,
            "patience": 5,
        },
    },
}

# 实验运行顺序
EXPERIMENT_ORDER = [
    # 知识增强
    "baseline",
    "no_knowledge",
    "text_desc_only",
    "meme_desc_only",
    # 防过拟合
    "grad_clip",
    "rdrop",
    "grad_clip_rdrop",
    "label_smoothing",
    "cosine_warmup",
    "clf_dropout",
    # 最优组合
    "best_combo",
]


# ============================================================
# 日志配置
# ============================================================

def setup_logging():
    """配置日志"""
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result", "ablation_mhke")
    os.makedirs(result_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(result_dir, f"ablation_mhke_{timestamp}.log")

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
# 设置随机种子
# ============================================================

def set_seed(seed):
    """设置全局随机种子"""
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    logger.info(f"实验: {exp_name}  [{exp_config['category']}]")
    logger.info(f"描述: {exp_config['desc']}")
    logger.info(f"覆盖参数: {json.dumps(exp_config['overrides'], ensure_ascii=False)}")
    logger.info(f"{'='*70}")

    # 重设随机种子
    set_seed(SEED)

    # 创建配置 — 使用 MHKE_CLIP 模型
    config = Config_base(model_name=MODEL_NAME, task_name=TASK_NAME)
    config.seed = SEED

    # 应用基线配置 (所有正则化关闭)
    baseline = get_baseline_config()
    for key, value in baseline.items():
        setattr(config, key, value)

    # 应用实验覆盖参数
    for key, value in exp_config["overrides"].items():
        setattr(config, key, value)

    # 添加实验标签
    config.exp_tag = f"ablmhke_clip_{exp_name}"

    # ====== 详细日志 ======
    logger.info(f"模型: MHKE_CLIP (ChineseCLIP)")
    logger.info(f"知识模式: {getattr(config, 'knowledge_mode', 'full')}")
    logger.info(f"batch_size: {config.batch_size}")
    use_rdrop = getattr(config, 'rdrop_alpha', 0) > 0
    if use_rdrop:
        logger.info(f"⚠ R-Drop 启用 → 双前向传播, 等效显存 batch={config.batch_size * 2}")

    logger.info("关键配置:")
    for key in ["model_name", "knowledge_mode", "num_epochs", "patience", "pad_size",
                 "batch_size", "learning_rate", "weight_decay", "rdrop_alpha",
                 "rdrop_use_sigmoid", "label_smoothing", "use_augmentation",
                 "use_scheduler", "warmup_ratio", "classifier_dropout",
                 "use_grad_clip", "freeze_layers", "weight"]:
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

    # 显存统计
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
    logger.info(f"  类别: {exp_config['category']}")
    logger.info(f"  知识模式: {getattr(config, 'knowledge_mode', 'full')}")
    logger.info(f"  最优 F1: {max_score*100:.2f}%")
    logger.info(f"  最优 Epoch: {best_epoch}")
    logger.info(f"  耗时: {elapsed:.1f} 分钟{peak_mem}")

    return {
        "exp_name": exp_name,
        "category": exp_config["category"],
        "desc": exp_config["desc"],
        "knowledge_mode": getattr(config, 'knowledge_mode', 'full'),
        "model_name": MODEL_NAME,
        "seed": SEED,
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
    """保存结果到 JSON"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(result_dir, f"ablation_mhke_results_{timestamp}.json")
    # 同时也写一份固定名称的最新结果
    latest_file = os.path.join(result_dir, "ablation_mhke_results_latest.json")

    output = {
        "experiment": "mhke_ablation",
        "model": MODEL_NAME,
        "seed": SEED,
        "dataset": "V4 (data_discription_4.0)",
        "task": TASK_NAME,
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }

    for fpath in [result_file, latest_file]:
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"结果已保存到: {result_file}")


def print_summary(results, logger):
    """打印实验汇总表 (分类别)"""
    logger.info(f"\n\n{'='*90}")
    logger.info(f"  📊 MHKE_CLIP 消融实验汇总  (seed={SEED}, dataset=V4)")
    logger.info(f"{'='*90}")

    # 获取 baseline F1
    baseline_f1 = None
    for r in results:
        if r["exp_name"] == "baseline":
            baseline_f1 = r["best_f1"]
            break

    # 按类别分组打印
    categories = ["知识增强", "防过拟合", "最优组合"]
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        if not cat_results:
            continue

        logger.info(f"\n  ── {cat} ──")
        logger.info(f"  {'实验':<22} {'F1':>8} {'Δ':>8} {'Epoch':>6} {'Time':>7}  描述")
        logger.info(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*6} {'-'*7}  {'-'*30}")

        sorted_cat = sorted(cat_results, key=lambda x: x["best_f1"], reverse=True)
        for r in sorted_cat:
            delta = ""
            if baseline_f1 is not None and r["exp_name"] != "baseline":
                diff = (r["best_f1"] - baseline_f1) * 100
                arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
                delta = f"{diff:+.2f}%{arrow}"
            else:
                delta = "  base"

            logger.info(
                f"  {r['exp_name']:<22} "
                f"{r['best_f1_pct']:>8} "
                f"{delta:>8} "
                f"{r['best_epoch']:>6} "
                f"{r['elapsed_min']:>5.1f}m  "
                f"{r['desc']}"
            )

    logger.info(f"\n{'='*90}")

    # 最佳实验
    if results:
        best = max(results, key=lambda x: x["best_f1"])
        logger.info(f"  🏆 最佳: {best['exp_name']} (F1={best['best_f1_pct']})")
        if baseline_f1 is not None:
            diff = (best["best_f1"] - baseline_f1) * 100
            logger.info(f"  📈 相对基线提升: {diff:+.2f}%")
    logger.info(f"{'='*90}\n")


# ============================================================
# CLI 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="MHKE_CLIP 消融实验 (知识增强 + 防过拟合, seed=2026, V4)"
    )
    parser.add_argument("--all", action="store_true", help="运行全部实验")
    parser.add_argument("--exp", nargs="+", help="运行指定实验")
    parser.add_argument("--list", action="store_true", help="列出所有可用实验")
    args = parser.parse_args()

    if args.list:
        print(f"\n可用实验 (共 {len(EXPERIMENT_ORDER)} 个):")
        print(f"  {'名称':<22} {'类别':<10} 描述")
        print(f"  {'-'*22} {'-'*10} {'-'*40}")
        for name in EXPERIMENT_ORDER:
            exp = EXPERIMENTS[name]
            print(f"  {name:<22} {exp['category']:<10} {exp['desc']}")
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
                print(f"❌ 错误: 未知实验 '{name}'")
                print(f"   可用: {', '.join(EXPERIMENT_ORDER)}")
                return

    logger, result_dir = setup_logging()

    logger.info(f"{'#'*70}")
    logger.info(f"#  MHKE_CLIP 消融实验")
    logger.info(f"#  模型:    MHKE_CLIP (ChineseCLIP + 知识增强)")
    logger.info(f"#  Seed:    {SEED}")
    logger.info(f"#  数据集:  V4 (data_discription_4.0)")
    logger.info(f"#  任务:    {TASK_NAME}")
    logger.info(f"#  实验数:  {len(exp_names)}")
    logger.info(f"#  实验:    {', '.join(exp_names)}")
    logger.info(f"{'#'*70}\n")

    # GPU 信息
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
        logger.info(f"\n>>> 进度: [{idx+1}/{len(exp_names)}] 即将训练: {exp_name}")
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
                "category": EXPERIMENTS[exp_name]["category"],
                "desc": EXPERIMENTS[exp_name]["desc"],
                "knowledge_mode": EXPERIMENTS[exp_name]["overrides"].get("knowledge_mode", "full"),
                "model_name": MODEL_NAME,
                "seed": SEED,
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
    logger.info(f"全部完成! 结果保存至: {result_dir}")


if __name__ == "__main__":
    main()
