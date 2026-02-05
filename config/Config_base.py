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
        self.train_path = path.join(data_dir, 'train_data_discription.json')
        self.dev_path = path.join(data_dir, 'test_data_discription.json')
        self.test_path = path.join(data_dir, 'test_data_discription.json')
        
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
        self.pad_size = 64                                              # 每句话处理成的长度(短填长切)

        # model
        self.dropout = 0.5                                              # 随机失活
        self.fc_hidden_dim = 256
        self.weight = 0.5

        # train
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # 设备
        self.learning_rate = 1e-5                                       # 学习率 
        self.num_epochs = 10                                            # epoch数 
        self.num_warm = 0                                              # 预热
        self.batch_size = 32                                           # mini-batch大小
        self.patience = 3                                              # Early stopping patience

        # evaluate
        self.score_key = "F1"                                            # 评价指标

# if __name__ == '__main__':
#     config = Config_base("BERT")
#     print(config.train_path)
