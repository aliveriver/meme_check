import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os
from os import path

class Config_base(object):

    """配置参数"""
    def __init__(self, model_name, task_name):

        # 项目根目录（相对路径）
        self.project_root = path.dirname(path.dirname(path.abspath(__file__)))
        
        # 模型名称和任务类型
        self.model_name = model_name
        self.task_name = task_name
        
        # 预训练模型路径（相对路径）
        models_dir = path.join(self.project_root, 'models')
        self.clip_path = path.join(models_dir, "chinese-clip-vit-base-patch16")
        self.chinese_clip_path = path.join(models_dir, "chinese-clip-vit-base-patch16")
        self.chinese_roberta_path = path.join(models_dir, "chinese-roberta-wwm-ext")
        self.roberta_path = path.join(models_dir, "chinese-roberta-wwm-ext")
        self.bert_path = path.join(models_dir, "bert-base-chinese")
        self.vit_path = path.join(models_dir, "vit-base-patch16-224")
        self.resnet_path = path.join(models_dir, "resnet-50") 

        # 特征维度
        if self.model_name == "clip":               
            self.hidden_dim = 512
        else:
            self.hidden_dim = 768 

        # 数据路径（相对路径）
        data_dir = path.join(self.project_root, 'data')
        self.meme_path = path.join(data_dir, 'meme') + path.sep
        self.train_path = path.join(data_dir, 'train_data_discription_4.0.json')
        self.dev_path = path.join(data_dir, 'test_data_discription_4.0.json')
        self.test_path = path.join(data_dir, 'test_data_discription_4.0.json')
        
        # 输出路径（相对路径）
        self.result_path = path.join(self.project_root, 'result')
        self.checkpoint_path = path.join(self.project_root, 'saved_dict')
        self.data_path = path.join(self.checkpoint_path, f'{self.model_name}_data.tar')

        if self.task_name == "task_1":
            self.num_classes = 2                                             # 类别数
        else:
            self.num_classes = 5                                             # 类别数

        # dataset
        self.seed = 1        
        self.pad_size = 128                                             # V4数据描述更长，从64增加到128

        # R-Drop 正则化 (V4消融: sigmoid +0.11%, softmax +0.23%)
        self.rdrop_alpha = 0.5                                      # R-Drop KL 散度权重
        self.rdrop_use_sigmoid = True                               # sigmoid 模式 (与 BCE 语义一致)

        # model
        self.dropout = 0.5                                              # 随机失活
        self.fc_hidden_dim = 256
        self.weight = 0.5

        # ====== 防过拟合配置 (V4消融最优: best_v3_combo = 81.09%) ======
        self.freeze_layers = 0                                         # 冻结预训练模型底部N层
        self.weight_decay = 0.01                                       # AdamW 权重衰减
        self.label_smoothing = 0.1                                     # V4消融: +0.47%
        self.use_augmentation = True                                   # V4消融: +0.20%
        self.backbone_lr_scale = 1.0                                   # backbone学习率=learning_rate*1.0
        self.use_scheduler = True                                      # V4消融: cosine_warmup
        self.warmup_ratio = 0.1                                        # 预热比例 10%
        self.classifier_dropout = 0.1                                  # V4消融: -0.06% (轻微负, 但组合有效)
        self.use_ema = False                                           # 是否使用EMA
        self.use_grad_clip = True                                      # V4消融: +0.43%

        # train
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # 设备
        self.learning_rate = 1e-5                                       # 学习率
        self.num_epochs = 20                                            # best_v3_combo 最优在 epoch 6
        self.num_warm = 0                                              # 预热
        self.batch_size = 32                                           # mini-batch大小
        self.patience = 5                                              # Early stopping patience

        # evaluate
        self.score_key = "F1"                                            # 评价指标

# if __name__ == '__main__':
#     config = Config_base("BERT")
#     print(config.train_path)
