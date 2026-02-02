"""
E2TC训练脚本
在MHKE基础上增加图像描述监督模块
"""

import torch
from torch.utils.data import DataLoader
from config.Config_base import Config_base
from dataset.dataset import MemeDataset
from train_eval_ import train, eval

def main():
    # 设置配置
    config = Config_base(model_name="MHKE-E2TC", task_name="task_1")
    
    # 可以调整E2TC的权重系数 (建议范围: 0.5-2.0)
    config.e2tc_weight = 1.0
    
    # 可以调整其他训练参数
    config.num_epochs = 10
    config.batch_size = 32
    config.learning_rate = 1e-5
    
    print("=" * 50)
    print("E2TC Training Configuration:")
    print(f"Model: {config.model_name}")
    print(f"Task: {config.task_name}")
    print(f"E2TC Weight: {config.e2tc_weight}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Device: {config.device}")
    print("=" * 50)
    
    # 加载数据集
    print("\nLoading datasets...")
    train_dataset = MemeDataset(config, training=True)
    dev_dataset = MemeDataset(config, training=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0  # Windows上建议设为0
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Dev samples: {len(dev_dataset)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Dev batches: {len(dev_loader)}")
    
    # 开始训练
    print("\nStarting training with E2TC supervision...")
    train(config, train_loader, dev_loader)
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main()
