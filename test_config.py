#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试配置文件 - 验证所有路径都是相对路径且可以正常访问
"""

from config.Config_base import Config_base
import os

def test_config():
    """测试配置文件"""
    print("="*70)
    print("配置文件测试 - 验证相对路径")
    print("="*70)
    
    # 创建配置
    config = Config_base('clip', 'task_1')
    
    print(f"\n✅ 配置加载成功")
    print(f"\n📁 项目根目录: {config.project_root}")
    
    # 检查模型路径
    print(f"\n{'='*70}")
    print("模型路径检查（相对路径）")
    print("="*70)
    
    model_paths = {
        "Chinese-CLIP": config.clip_path,
        "Chinese-CLIP (MHKE)": config.chinese_clip_path,
        "Chinese-RoBERTa": config.chinese_roberta_path,
        "RoBERTa": config.roberta_path,
        "BERT-Chinese": config.bert_path,
        "ViT": config.vit_path,
    }
    
    for name, path in model_paths.items():
        rel_path = os.path.relpath(path, config.project_root)
        exists = "✅" if os.path.exists(path) else "❌"
        print(f"{exists} {name:25s} -> {rel_path}")
    
    # 检查数据路径
    print(f"\n{'='*70}")
    print("数据路径检查（相对路径）")
    print("="*70)
    
    data_paths = {
        "训练集": config.train_path,
        "验证集": config.dev_path,
        "测试集": config.test_path,
        "图片目录": config.meme_path,
    }
    
    for name, path in data_paths.items():
        rel_path = os.path.relpath(path, config.project_root)
        exists = "✅" if os.path.exists(path) else "❌"
        print(f"{exists} {name:15s} -> {rel_path}")
    
    # 检查输出路径
    print(f"\n{'='*70}")
    print("输出路径检查（相对路径）")
    print("="*70)
    
    output_paths = {
        "结果目录": config.result_path,
        "检查点目录": config.checkpoint_path,
        "数据缓存": config.data_path,
    }
    
    for name, path in output_paths.items():
        rel_path = os.path.relpath(path, config.project_root)
        exists = "✅" if os.path.exists(path) else "⚠️ (将自动创建)"
        print(f"{exists} {name:15s} -> {rel_path}")
    
    # 检查训练参数
    print(f"\n{'='*70}")
    print("训练参数")
    print("="*70)
    print(f"模型: {config.model_name}")
    print(f"任务: {config.task_name}")
    print(f"类别数: {config.num_classes}")
    print(f"批次大小: {config.batch_size}")
    print(f"训练轮数: {config.num_epochs}")
    print(f"学习率: {config.learning_rate}")
    print(f"设备: {config.device}")
    print(f"隐藏层维度: {config.hidden_dim}")
    
    # 验证所有必需文件是否存在
    print(f"\n{'='*70}")
    print("必需文件检查")
    print("="*70)
    
    required_files = [
        config.train_path,
        config.test_path,
        config.clip_path,
        config.roberta_path,
        config.vit_path,
    ]
    
    all_exist = True
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ 缺失: {os.path.relpath(file_path, config.project_root)}")
            all_exist = False
    
    if all_exist:
        print("✅ 所有必需文件都存在")
    else:
        print("\n⚠️  部分文件缺失，请检查")
    
    # 总结
    print(f"\n{'='*70}")
    if all_exist:
        print("✅ 配置检查通过！所有路径都是相对路径，可以上传到服务器")
        print("\n📦 上传到服务器时需要包含以下目录:")
        print("   - config/          (配置文件)")
        print("   - data/            (数据集)")
        print("   - dataset/         (数据加载器)")
        print("   - model/           (模型定义)")
        print("   - models/          (预训练模型)")
        print("   - run.py           (训练脚本)")
        print("   - train_eval_.py   (训练逻辑)")
        print("   - requirements.txt (依赖列表)")
        print("\n🚀 在服务器上运行: python run.py")
    else:
        print("❌ 配置检查失败，请修复缺失的文件")
    print("="*70)

if __name__ == "__main__":
    test_config()
