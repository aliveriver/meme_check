#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据提取脚本 - 从 ToxiCN_MM 数据集中提取每张图片的 path 与 label
输出为 CSV 文件，方便后续使用。

使用方法:
    python data_extraction.py
"""

import json
import csv
import os
from collections import Counter

# ============================================================
# 配置区 - 根据实际路径修改
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

TRAIN_JSON = os.path.join(DATA_DIR, "train_data_discription_2.0.json")
TEST_JSON = os.path.join(DATA_DIR, "test_data_discription_2.0.json")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "extracted")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_OUTPUT_CSV = os.path.join(OUTPUT_DIR, "train_path_label.csv")
TEST_OUTPUT_CSV = os.path.join(OUTPUT_DIR, "test_path_label.csv")
ALL_OUTPUT_CSV = os.path.join(OUTPUT_DIR, "all_path_label.csv")

TRAIN_OUTPUT_JSON = os.path.join(OUTPUT_DIR, "train_path_label.json")
TEST_OUTPUT_JSON = os.path.join(OUTPUT_DIR, "test_path_label.json")
ALL_OUTPUT_JSON = os.path.join(OUTPUT_DIR, "all_path_label.json")

# 图片根目录（用于拼接绝对路径）
MEME_DIR = os.path.join(DATA_DIR, "meme")

# 标签映射
LABEL_MAP = {0: "Non-harmful", 1: "Harmful"}
TYPE_MAP = {
    0: "Non-harmful",
    1: "Targeted Harmful",
    2: "Sexual Innuendo",
    3: "General Offense",
    4: "Dispirited Culture",
}
TARGET_MAP = {
    0: "None",
    1: "Gender",
    2: "Race",
    3: "MBTI",
    4: "Region",
    5: "Age",
    6: "Appearance & Body",
    7: "Disease & Health",
    8: "LGBTQ",
    9: "Occupation",
    10: "Community & Organize",
    11: "Specific Individual",
    12: "Poor",
    13: "Marriage & Relationship",
    14: "Parent",
    15: "Others",
}


def load_json(filepath: str) -> list:
    """加载 JSON 数据文件"""
    print(f"  正在加载: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  共 {len(data)} 条样本")
    return data


def extract_path_label(data: list) -> list:
    """
    从数据中提取 path 和 label 信息
    path 输出为绝对路径
    返回: [(path, label, type, text_modal, image_modal, target), ...]
    """
    records = []
    for item in data:
        raw_path = item.get("path", "")
        abs_path = os.path.join(MEME_DIR, raw_path) if raw_path else ""
        record = {
            "path": abs_path,
            "label": item.get("label", -1),
            "label_name": LABEL_MAP.get(item.get("label", -1), "Unknown"),
            "type": item.get("type", -1),
            "type_name": TYPE_MAP.get(item.get("type", -1), "Unknown"),
            "text_modal": item.get("text_modal", 0),
            "image_modal": item.get("image_modal", 0),
            "target": item.get("target", 0),
            "target_name": TARGET_MAP.get(item.get("target", 0), "Unknown"),
        }
        records.append(record)
    return records


def save_csv(records: list, output_path: str):
    """保存提取结果到 CSV"""
    if not records:
        print(f"  ⚠️ 无数据可保存")
        return

    fieldnames = records[0].keys()
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"  ✅ CSV 已保存到: {output_path} ({len(records)} 条)")


def save_json(records: list, output_path: str):
    """保存提取结果到 JSON（与原始数据格式一致）"""
    if not records:
        print(f"  ⚠️ 无数据可保存")
        return

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"  ✅ JSON 已保存到: {output_path} ({len(records)} 条)")


def print_statistics(records: list, split_name: str):
    """打印数据统计信息"""
    print(f"\n{'='*60}")
    print(f"📊 {split_name} 统计")
    print(f"{'='*60}")
    print(f"  样本总数: {len(records)}")

    # label 分布
    label_counter = Counter(r["label_name"] for r in records)
    print(f"\n  [Label 分布]")
    for name, count in label_counter.most_common():
        pct = count / len(records) * 100
        print(f"    {name:20s}: {count:6d} ({pct:.1f}%)")

    # type 分布
    type_counter = Counter(r["type_name"] for r in records)
    print(f"\n  [Type 分布]")
    for name, count in type_counter.most_common():
        pct = count / len(records) * 100
        print(f"    {name:20s}: {count:6d} ({pct:.1f}%)")

    # target 分布 (仅有害样本)
    harmful_records = [r for r in records if r["label"] == 1]
    if harmful_records:
        target_counter = Counter(r["target_name"] for r in harmful_records)
        print(f"\n  [Target 分布] (仅有害样本, 共 {len(harmful_records)} 条)")
        for name, count in target_counter.most_common():
            pct = count / len(harmful_records) * 100
            print(f"    {name:20s}: {count:6d} ({pct:.1f}%)")

    # modality 分布
    modal_counter = Counter(
        f"text={r['text_modal']}, image={r['image_modal']}" for r in records
    )
    print(f"\n  [Modality 分布]")
    for name, count in modal_counter.most_common():
        pct = count / len(records) * 100
        print(f"    {name:30s}: {count:6d} ({pct:.1f}%)")


def main():
    print("=" * 60)
    print("🔍 ToxiCN_MM 数据提取工具")
    print("=" * 60)

    # ---- 训练集 ----
    print("\n📂 处理训练集...")
    train_data = load_json(TRAIN_JSON)
    train_records = extract_path_label(train_data)
    save_csv(train_records, TRAIN_OUTPUT_CSV)
    save_json(train_records, TRAIN_OUTPUT_JSON)
    print_statistics(train_records, "训练集")

    # ---- 测试集 ----
    print("\n📂 处理测试集...")
    test_data = load_json(TEST_JSON)
    test_records = extract_path_label(test_data)
    save_csv(test_records, TEST_OUTPUT_CSV)
    save_json(test_records, TEST_OUTPUT_JSON)
    print_statistics(test_records, "测试集")

    # ---- 合并 ----
    all_records = train_records + test_records
    save_csv(all_records, ALL_OUTPUT_CSV)
    save_json(all_records, ALL_OUTPUT_JSON)
    print_statistics(all_records, "全部数据")

    print(f"\n{'='*60}")
    print(f"✅ 提取完成！输出目录: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
