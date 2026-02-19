import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from tqdm import tqdm
import copy

import time
import json
from dataset.dataset import get_time_dif, convert_onehot
from model.clip import *
from model.vit_roberta import *
from model.MHKE import *


# ====== EMA: 指数移动平均 ======
class EMA:
    """
    Exponential Moving Average for model parameters.
    维护模型参数的指数移动平均值，验证时使用平滑后的参数，
    能有效提升泛化性能（相当于集成训练过程中多个模型）。
    """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}  # 存储 EMA 参数
        self.backup = {}  # 备份原始参数
        self._register()
    
    def _register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """每个 optimizer.step() 之后调用"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """验证前调用：用 EMA 参数替换模型参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """验证后调用：恢复原始参数继续训练"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# ====== FGM: Fast Gradient Method 对抗训练 ======
class FGM:
    """
    Fast Gradient Method 对抗训练。
    对 embedding 层的参数添加微小扰动，迫使模型在扰动后仍能做出正确预测，
    显著增强模型鲁棒性和泛化能力。
    """
    def __init__(self, model, epsilon=1.0, emb_names=('embeddings',)):
        self.model = model
        self.epsilon = epsilon
        self.emb_names = emb_names
        self.backup = {}
    
    def attack(self):
        """在 loss.backward() 之后调用，对 embedding 参数添加扰动"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and any(emb in name for emb in self.emb_names):
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)
    
    def restore(self):
        """对抗训练结束后调用，恢复原始 embedding 参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and any(emb in name for emb in self.emb_names):
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def get_parameter_groups(config, model):
    """
    将模型参数分为 backbone（低学习率）和 head（高学习率）两组。
    backbone 包括预训练权重，head 包括分类头和新增模块。
    同时对 backbone 施加更强的 weight_decay。
    """
    backbone_params = []
    head_params = []
    
    backbone_keywords = ['model', 'cv_model', 'nlp_model', 'vision_model', 'text_model']
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # 判断是否属于预训练 backbone
        is_backbone = False
        for kw in backbone_keywords:
            if name.startswith(kw + '.'):
                is_backbone = True
                break
        
        if is_backbone:
            backbone_params.append(param)
        else:
            head_params.append(param)
    
    backbone_lr = config.learning_rate * getattr(config, 'backbone_lr_scale', 0.1)
    weight_decay = getattr(config, 'weight_decay', 0.01)
    
    print(f"  参数组: backbone {len(backbone_params)} 个张量 (lr={backbone_lr:.1e}, wd={weight_decay})")
    print(f"  参数组: head     {len(head_params)} 个张量 (lr={config.learning_rate:.1e}, wd={weight_decay})")
    
    param_groups = [
        {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': weight_decay},
        {'params': head_params, 'lr': config.learning_rate, 'weight_decay': weight_decay},
    ]
    return param_groups


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """带 Warmup 的余弦退火学习率调度器"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def train(config, train_iter, dev_iter):

    if config.model_name == "clip":
        # model = CLIPMemesClassifier(config).to(config.device)
        model = MHKE_CLIP(config).to(config.device)
    elif config.model_name == "vit-roberta":
        model = VitRobertaMemesClassifier(config).to(config.device)
    elif config.model_name == "vit":
        model = VitClassifier(config).to(config.device)
    elif config.model_name == "resnet":
        model = ResNetClassifier(config).to(config.device)
    elif config.model_name == "roberta":
        model = RobertaClassifier(config).to(config.device)
    elif config.model_name == "bert":
        model = BertClassifier(config).to(config.device)
    elif config.model_name == "MHKE":
        model = MHKE(config).to(config.device)
    elif config.model_name == "MHKE_CrossAttention":
        model = MHKE_CrossAttention(config).to(config.device)
    elif config.model_name == "MHKE_CrossAttention_V2":
        model = MHKE_CrossAttention_V2(config).to(config.device)
    elif config.model_name == "MHKE_ISSUES":
        model = MHKE_ISSUES(config).to(config.device)

    model_name = '{}_B-{}_E-{}_Lr-{}_w-{}_{}_add'.format(config.model_name, config.batch_size,
                                                         config.num_epochs, config.learning_rate, config.weight, config.task_name)
    
    # ====== 打印模型参数统计 ======
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"\n{'='*60}")
    print(f"模型参数统计:")
    print(f"  总参数:   {total_params:>12,}")
    print(f"  可训练:   {trainable_params:>12,} ({trainable_params/total_params*100:.1f}%)")
    print(f"  已冻结:   {frozen_params:>12,} ({frozen_params/total_params*100:.1f}%)")
    print(f"{'='*60}\n")

    # ====== 优化器: 分层学习率 + Weight Decay ======
    if config.model_name == "resnet":
        model_optimizer = optim.Adam(
            model.parameters(), lr=config.learning_rate)
    else:
        param_groups = get_parameter_groups(config, model)
        model_optimizer = optim.AdamW(param_groups)
    
    # ====== 学习率调度器: Cosine Warmup ======
    scheduler = None
    use_scheduler = getattr(config, 'use_scheduler', False)
    if use_scheduler:
        total_steps = config.num_epochs * len(train_iter)
        warmup_ratio = getattr(config, 'warmup_ratio', 0.1)
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(model_optimizer, warmup_steps, total_steps)
        print(f"✓ Cosine Warmup 调度器: warmup {warmup_steps}/{total_steps} steps")

    # ====== 损失函数: Label Smoothing ======
    label_smoothing = getattr(config, 'label_smoothing', 0.0)
    if label_smoothing > 0:
        print(f"✓ Label Smoothing = {label_smoothing}")
    loss_fn = nn.BCEWithLogitsLoss()
    
    # ====== EMA: 指数移动平均 ======
    ema = None
    use_ema = getattr(config, 'use_ema', False)
    if use_ema:
        ema_decay = getattr(config, 'ema_decay', 0.999)
        ema = EMA(model, decay=ema_decay)
        print(f"✓ EMA enabled (decay={ema_decay})")
    
    # ====== FGM: 对抗训练 ======
    fgm = None
    use_fgm = getattr(config, 'use_fgm', False)
    if use_fgm:
        fgm_epsilon = getattr(config, 'fgm_epsilon', 1.0)
        fgm = FGM(model, epsilon=fgm_epsilon)
        print(f"✓ FGM adversarial training enabled (epsilon={fgm_epsilon})")
    
    # ====== Mixup ======
    use_mixup = getattr(config, 'use_mixup', False)
    mixup_alpha = getattr(config, 'mixup_alpha', 0.2)
    if use_mixup:
        print(f"✓ Mixup enabled (alpha={mixup_alpha})")
    
    # R-Drop 正则化配置
    rdrop_alpha = getattr(config, 'rdrop_alpha', 0.5)
    use_rdrop = rdrop_alpha > 0
    if use_rdrop:
        print(f"✓ R-Drop regularization enabled (alpha={rdrop_alpha})")
    
    max_score = 0
    
    # Early stopping 参数
    patience = getattr(config, 'patience', 3)  # 默认 patience=3
    no_improve_count = 0
    best_epoch = 0

    for epoch in range(config.num_epochs):
        model.train()
        start_time = time.time()
        print("Model is training in epoch {}".format(epoch))
        loss_all = 0.
        preds = []
        labels = []

        for batch in tqdm(train_iter, desc='Training', colour='MAGENTA'):
            model.zero_grad()
            
            # 获取标签
            if config.task_name == "task_1":
                label = batch['label']
            else:
                label = batch['type_label']
            
            # Label Smoothing
            if label_smoothing > 0:
                smoothed_label = label.float() * (1 - label_smoothing) + label_smoothing / label.size(-1)
            else:
                smoothed_label = label.float()
            
            # ====== Mixup: 对 batch 内样本做插值混合 ======
            if use_mixup and np.random.random() < 0.5:  # 50% 概率使用 mixup
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                batch_size_cur = label.size(0)
                index = torch.randperm(batch_size_cur)
                
                # 混合标签
                mixed_label = lam * smoothed_label + (1 - lam) * smoothed_label[index]
                
                # 混合输入：对文本 input_ids 不做混合（离散的），仅混合图像
                mixed_batch = {k: v for k, v in batch.items()}
                if 'image_tensor' in batch:
                    mixed_batch['image_tensor'] = lam * batch['image_tensor'] + (1 - lam) * batch['image_tensor'][index]
                
                if use_rdrop:
                    logit1 = model(**mixed_batch).cpu()
                    logit2 = model(**mixed_batch).cpu()
                    pred = get_preds(config, logit1) if config.task_name == "task_1" else get_preds_task2(config, logit1)
                    cls_loss = (loss_fn(logit1, mixed_label) + loss_fn(logit2, mixed_label)) / 2
                    p1 = F.softmax(logit1, dim=-1)
                    p2 = F.softmax(logit2, dim=-1)
                    kl_loss = (F.kl_div(p1.log(), p2, reduction='batchmean') + 
                               F.kl_div(p2.log(), p1, reduction='batchmean')) / 2
                    loss = cls_loss + rdrop_alpha * kl_loss
                else:
                    logit = model(**mixed_batch).cpu()
                    pred = get_preds(config, logit) if config.task_name == "task_1" else get_preds_task2(config, logit)
                    loss = loss_fn(logit, mixed_label)
            else:
                # ====== 标准前向传播 ======
                if use_rdrop:
                    logit1 = model(**batch).cpu()
                    logit2 = model(**batch).cpu()
                    pred = get_preds(config, logit1) if config.task_name == "task_1" else get_preds_task2(config, logit1)
                    cls_loss = (loss_fn(logit1, smoothed_label) + loss_fn(logit2, smoothed_label)) / 2
                    p1 = F.softmax(logit1, dim=-1)
                    p2 = F.softmax(logit2, dim=-1)
                    kl_loss = (F.kl_div(p1.log(), p2, reduction='batchmean') + 
                               F.kl_div(p2.log(), p1, reduction='batchmean')) / 2
                    loss = cls_loss + rdrop_alpha * kl_loss
                else:
                    logit = model(**batch).cpu()
                    pred = get_preds(config, logit) if config.task_name == "task_1" else get_preds_task2(config, logit)
                    loss = loss_fn(logit, smoothed_label)

            preds.extend(pred)
            labels.extend(label.detach().numpy())
            loss_all += loss.item()
            
            # 反向传播
            model_optimizer.zero_grad()
            loss.backward()
            
            # ====== FGM 对抗训练 ======
            if fgm is not None:
                fgm.attack()  # 在 embedding 上添加对抗扰动
                # 对抗样本前向传播
                if use_rdrop:
                    adv_logit = model(**batch).cpu()
                    adv_loss = loss_fn(adv_logit, smoothed_label)
                else:
                    adv_logit = model(**batch).cpu()
                    adv_loss = loss_fn(adv_logit, smoothed_label)
                adv_loss.backward()  # 累积对抗梯度
                fgm.restore()  # 恢复 embedding
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            model_optimizer.step()
            
            # EMA 更新
            if ema is not None:
                ema.update()
            
            # 更新学习率
            if scheduler is not None:
                scheduler.step()

        end_time = time.time()
        print(" took: {:.1f} min".format((end_time - start_time)/60.))
        print("TRAINED for {} epochs".format(epoch))

        # 验证
        if epoch >= config.num_warm:
            trn_scores = get_scores(preds, labels, loss_all, len(
                train_iter), data_name="TRAIN")
            
            # EMA: 验证时使用平滑参数
            if ema is not None:
                ema.apply_shadow()
            
            dev_scores, _ = eval(config, model, loss_fn,
                                 dev_iter, data_name='DEV')
            
            # EMA: 验证后恢复原始参数
            if ema is not None:
                ema.restore()
            
            f = open(
                '{}/{}.all_scores.txt'.format(config.result_path, model_name), 'a')
            f.write(' ==================================================  Epoch: {}  ==================================================\n'.format(epoch))
            f.write('TrainScore: \n{}\nEvalScore: \n{}\n'.format(
                json.dumps(trn_scores), json.dumps(dev_scores)))
            
            # Early stopping 检查
            curr_score = dev_scores[config.score_key]
            if curr_score > max_score:
                max_score = curr_score
                best_epoch = epoch
                no_improve_count = 0
                # 保存最佳模型 (EMA 时保存平滑参数)
                if ema is not None:
                    ema.apply_shadow()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                }, '{}/ckp-{}-{}.tar'.format(config.checkpoint_path, model_name, 'BEST'))
                if ema is not None:
                    ema.restore()
                print(f"✓ New best F1: {curr_score:.4f} at epoch {epoch}")
            else:
                no_improve_count += 1
                print(f"✗ No improvement for {no_improve_count} epoch(s). Best F1: {max_score:.4f} at epoch {best_epoch}")
            
            # Early stopping 触发
            if no_improve_count >= patience:
                print(f"\n⚠️ Early stopping triggered! No improvement for {patience} consecutive epochs.")
                print(f"Best F1: {max_score:.4f} achieved at epoch {best_epoch}")
                f.write(f'\n=== Early Stopping at epoch {epoch} ===\n')
                f.write(f'Best F1: {max_score:.4f} at epoch {best_epoch}\n')
                f.close()
                break
                
            f.close()
        print("ALLTRAINED for {} epochs".format(epoch))


def eval(config, model, loss_fn, dev_iter, data_name='DEV'):

    loss_all = 0.
    preds = []
    labels = []

    for batch in tqdm(dev_iter, desc='Evaling', colour='CYAN'):
        with torch.no_grad():
            logit = model(**batch).cpu()

            if config.task_name == "task_1":
                label = batch['label']
                pred = get_preds(config, logit)
            else:
                label = batch['type_label']
                pred = get_preds_task2(config, logit)

            loss = loss_fn(logit, label.float())

            preds.extend(pred)
            labels.extend(label.detach().numpy())
            loss_all += loss.item()

    dev_scores = get_scores(preds, labels, loss_all,
                            len(dev_iter), data_name=data_name)

    return dev_scores, preds


def test(model, dev_iter):

    preds = []
    labels = []

    for batch in tqdm(dev_iter, desc='Testing', colour='CYAN'):
        with torch.no_grad():
            logit = model(**batch).cpu()

            label = batch['label']
            pred = output_preds(logit)

            preds.extend(pred)
            labels.extend(label.detach().numpy())

        df = pd.DataFrame({'new_pred': preds})
        output_file = 'preds.csv'
        df.to_csv(output_file, index=False)

    return preds


# Task 1: Harmful Meme Detection
def get_preds(config, logit):
    results = torch.max(logit.data, 1)[1].cpu().numpy()
    new_results = []
    for result in results:
        result = convert_onehot(config, result)
        new_results.append(result)
    return new_results


# Task 2: Harmful Type Discrimination
def get_preds_task2(config, logit):
    all_results = []
    logit_ = torch.sigmoid(logit)
    results = torch.max(logit_.data, 1)[1].cpu().numpy()
    for i in range(len(results)):
        result = convert_onehot(config, results[i])
        all_results.append(result)
    return all_results


def output_preds(logit):
    results = torch.max(logit.data, 1)[1].cpu().numpy()
    new_results = []
    for result in results:
        new_results.append(result)
    return new_results


def get_scores(all_preds, all_lebels, loss_all, len, data_name):
    score_dict = dict()
    f1 = f1_score(all_preds, all_lebels, average='macro')
    acc = accuracy_score(all_preds, all_lebels)
    all_f1 = f1_score(all_preds, all_lebels, average=None)
    pre = precision_score(all_preds, all_lebels, average='macro')
    recall = recall_score(all_preds, all_lebels, average='macro')

    score_dict['F1'] = f1
    score_dict['accuracy'] = acc
    score_dict['all_f1'] = all_f1.tolist()
    score_dict['precision'] = pre
    score_dict['recall'] = recall

    score_dict['all_loss'] = loss_all/len
    print("Evaling on \"{}\" data".format(data_name))
    for s_name, s_val in score_dict.items():
        print("{}: {}".format(s_name, s_val))
    return score_dict


def save_best(config, epoch, model_name, model, score, max_score):
    score_key = config.score_key
    curr_score = score[score_key]
    print('The epoch_{} {}: {}\nCurrent max {}: {}'.format(
        epoch, score_key, curr_score, score_key, max_score))

    if curr_score > max_score or epoch == 0:
        torch.save({
            'epoch': config.num_epochs,
            'model_state_dict': model.state_dict(),
        }, '{}/ckp-{}-{}.tar'.format(config.checkpoint_path, model_name, 'BEST'))
        return curr_score
    else:
        return max_score
