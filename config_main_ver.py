import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os
from os import path

class Config_base(object):

    """閰嶇疆鍙傛暟"""
    def __init__(self, model_name, task_name):

        # 椤圭洰鏍圭洰褰曪紙鐩稿璺緞锛?        self.project_root = path.dirname(path.dirname(path.abspath(__file__)))
        
        # 妯″瀷鍚嶇О鍜屼换鍔＄被鍨?        self.model_name = model_name
        self.task_name = task_name
        
        # 棰勮缁冩ā鍨嬭矾寰勶紙鐩稿璺緞锛?        models_dir = path.join(self.project_root, 'models')
        self.clip_path = path.join(models_dir, "chinese-clip-vit-base-patch16")
        self.chinese_clip_path = path.join(models_dir, "chinese-clip-vit-base-patch16")
        self.chinese_roberta_path = path.join(models_dir, "chinese-roberta-wwm-ext")
        self.roberta_path = path.join(models_dir, "chinese-roberta-wwm-ext")
        self.bert_path = path.join(models_dir, "bert-base-chinese")
        self.vit_path = path.join(models_dir, "vit-base-patch16-224")
        self.resnet_path = path.join(models_dir, "resnet-50") 

        # 鐗瑰緛缁村害
        if self.model_name == "clip":               
            self.hidden_dim = 512
        else:
            self.hidden_dim = 768 

        # 鏁版嵁璺緞锛堢浉瀵硅矾寰勶級
        data_dir = path.join(self.project_root, 'data')
        self.meme_path = path.join(data_dir, 'meme') + path.sep
        self.train_path = path.join(data_dir, 'train_data_discription.json')
        self.dev_path = path.join(data_dir, 'test_data_discription.json')
        self.test_path = path.join(data_dir, 'test_data_discription.json')
        
        # 杈撳嚭璺緞锛堢浉瀵硅矾寰勶級
        self.result_path = path.join(self.project_root, 'result')
        self.checkpoint_path = path.join(self.project_root, 'saved_dict')
        self.data_path = path.join(self.checkpoint_path, f'{self.model_name}_data.tar')

        if self.task_name == "task_1":
            self.num_classes = 2                                             # 绫诲埆鏁?        else:
            self.num_classes = 5                                             # 绫诲埆鏁?
        # dataset
        self.seed = 1        
        self.pad_size = 64                                              # 姣忓彞璇濆鐞嗘垚鐨勯暱搴?鐭～闀垮垏)

        # model
        self.dropout = 0.5                                              # 闅忔満澶辨椿
        self.fc_hidden_dim = 256
        self.weight = 0.5

        # train
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # 璁惧
        self.learning_rate = 1e-5                                       # 瀛︿範鐜? transformer:5e-4 
        self.num_epochs = 10                                            # epoch鏁?
        self.num_warm = 0                                              # 棰勭儹
        self.batch_size = 32                                           # mini-batch澶у皬

        # evaluate
        self.score_key = "F1"                                            # 璇勪环鎸囨爣

# if __name__ == '__main__':
#     config = Config_base("BERT")
#     print(config.train_path)
