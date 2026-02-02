import os
import numpy as np

import torch
import torch.nn as nn
import json

from torchvision.transforms import functional as F
# from transformers import BertTokenizer, CLIPProcessor, CLIPModel, CLIPTokenizer

from config.Config_base import Config_base
from dataset.dataset import *
from train_eval_ import train, test

from model.clip import *
from model.vit_roberta import *
from model.MHKE import *

if __name__ == '__main__':

    # ============================================
    # 模型选择
    # ============================================
    # 基础模型
    # model_name = "clip"
    # model_name = "vit-roberta"
    # model_name = "MHKE"
    
    # E2TC 增强模型（推荐）
    model_name = "MHKE-E2TC"
    
    # 任务选择
    task_name = "task_1"  # task_1: 二分类, task_2: 多分类
    
    config = Config_base(model_name, task_name)
    
    # ============================================
    # E2TC 超参数配置（仅当使用 MHKE-E2TC 时有效）
    # ============================================
    if model_name == "MHKE-E2TC":
        config.e2tc_weight = 1.0      # Caption Loss 权重 (建议: 0.5-2.0)
        config.num_epochs = 10         # 训练轮数
        config.batch_size = 32         # 批次大小（E2TC显存占用较大，可适当减小）
        config.learning_rate = 1e-5    # 学习率
        
        print("=" * 60)
        print("E2TC Configuration:")
        print(f"  E2TC Weight: {config.e2tc_weight}")
        print(f"  Epochs: {config.num_epochs}")
        print(f"  Batch Size: {config.batch_size}")
        print(f"  Learning Rate: {config.learning_rate}")
        print("=" * 60)

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    if not os.path.exists(config.data_path): 
        trn_data = MemeDataset(config, training=True)
        test_data = MemeDataset(config, training=False)
        torch.save({
            'trn_data' : trn_data,
            'test_data' : test_data,
            }, config.data_path)
    else:
        checkpoint = torch.load(config.data_path)
        trn_data = checkpoint['trn_data']
        test_data = checkpoint['test_data']

    print('The size of the Training dataset: {}'.format(len(trn_data)))
    print('The size of the Test dataset: {}'.format(len(test_data)))

    train_iter = DataLoader(trn_data, batch_size=int(config.batch_size), shuffle=False)
    test_iter = DataLoader(test_data, batch_size=int(config.batch_size), shuffle=False)

    # ============================================
    # 开始训练
    # ============================================
    print(f"\nTraining with model: {model_name}")
    print(f"Task: {task_name}")
    print(f"Training samples: {len(trn_data)}")
    print(f"Test samples: {len(test_data)}")
    print("=" * 60 + "\n")
    
    train(config, train_iter, test_iter)
    
    # ============================================
    # 超参数搜索（可选）
    # ============================================
    # 1. E2TC权重搜索
    # if model_name == "MHKE-E2TC":
    #     e2tc_weights = [0.5, 1.0, 2.0]
    #     for weight in e2tc_weights:
    #         config.e2tc_weight = weight
    #         print(f"\n>>> Testing E2TC weight: {weight}")
    #         train(config, train_iter, test_iter)
    
    # 2. Batch size搜索
    # all_batch_size = [16, 32, 64]
    # for batch_size in all_batch_size:
    #     config.batch_size = batch_size
    #     train(config, train_iter, test_iter)
    
    # 3. 学习率搜索
    # learning_rates = [1e-5, 2e-5, 5e-5]
    # for lr in learning_rates:
    #     config.learning_rate = lr
    #     train(config, train_iter, test_iter)

    # ============================================
    # 模型加载和测试（可选）
    # ============================================
    # 加载训练好的模型进行推理
    # if model_name == "MHKE-E2TC":
    #     model = MHKE_E2TC(config).to(config.device)
    # elif model_name == "MHKE":
    #     model = MHKE(config).to(config.device)
    # 
    # checkpoint_path = f'{config.checkpoint_path}/ckp-{model_name}_B-{config.batch_size}_E-{config.num_epochs}_Lr-{config.learning_rate}_w-{config.weight}_{task_name}_add-BEST.tar'
    # checkpoint = torch.load(checkpoint_path)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # 
    # preds = test(model, test_iter)



