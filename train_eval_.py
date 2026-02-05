import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import time
import json
from dataset.dataset import get_time_dif, convert_onehot
from model.clip import *
from model.vit_roberta import *
from model.MHKE import *
from model.uncertainty_weighting import MultiTaskUncertaintyWeighting


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
    elif config.model_name == "MHKE-E2TC":
        # 使用带E2TC监督模块的MHKE模型
        model = MHKE_E2TC(config).to(config.device)
    elif config.model_name == "MHKE":
        model = MHKE(config).to(config.device)

    # 生成模型名称 (用于保存路径)
    # 区分 uncertainty weighting 模式和固定权重模式
    use_uw = getattr(config, 'use_uncertainty_weighting', False)
    if use_uw:
        weight_str = "UW"  # Uncertainty Weighting
    else:
        weight_str = f"w-{config.weight}"
    model_name = '{}_B-{}_E-{}_Lr-{}_{}_{}_add'.format(
        config.model_name, config.batch_size, config.num_epochs, 
        config.learning_rate, weight_str, config.task_name
    )
    # for name, parameters in model.named_parameters():
    #     print(name)
    params = list(model.named_parameters())

    if config.model_name == "resnet":
        model_optimizer = optim.Adam(
            model.parameters(), lr=config.learning_rate)
    # elif config.model_name == "vit-roberta":
    #     model_optimizer = optim.AdamW([
    #         {'params':model.cv_model.parameters(), 'lr': 5e-5},
    #         {'params':model.nlp_model.parameters(), 'lr': config.learning_rate},
    #         {'params':model.classifier.parameters(), 'lr': config.learning_rate}
    #     ])
    else:
        model_optimizer = optim.AdamW(
            model.parameters(), lr=config.learning_rate)

    loss_fn = nn.BCEWithLogitsLoss()
    # E2TC: 为caption生成任务添加CrossEntropyLoss (忽略padding)
    cap_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    # 是否使用E2TC
    use_e2tc = (config.model_name == "MHKE-E2TC")
    
    # Kendall Uncertainty Weighting: 是否使用自动学习权重
    use_uncertainty_weighting = use_e2tc and getattr(config, 'use_uncertainty_weighting', False)
    
    if use_uncertainty_weighting:
        # 初始化不确定性权重模块
        # init_log_vars: [cls_log_var, cap_log_var]
        # 默认 [0.0, 0.0] -> σ² = 1 -> 初始权重约为 0.5
        # 可通过 config.init_log_vars 自定义
        init_log_vars = getattr(config, 'init_log_vars', [0.0, 0.0])
        uncertainty_module = MultiTaskUncertaintyWeighting(
            num_tasks=2,
            init_log_vars=init_log_vars
        ).to(config.device)
        
        # 优化器需要包含 uncertainty_module 的参数
        model_optimizer = optim.AdamW([
            {'params': model.parameters(), 'lr': config.learning_rate},
            {'params': uncertainty_module.parameters(), 'lr': config.learning_rate}
        ])
        
        print("=" * 60)
        print("Kendall Uncertainty Weighting ENABLED")
        print(f"  Initial log_vars: {init_log_vars}")
        print(f"  Initial sigmas: {uncertainty_module.get_sigmas()}")
        print(f"  Initial weights: {uncertainty_module.get_weights()}")
        print("=" * 60)
    else:
        uncertainty_module = None
        e2tc_weight = config.e2tc_weight if hasattr(config, 'e2tc_weight') else 1.0
    
    max_score = 0

    for epoch in range(config.num_epochs):
        model.train()
        start_time = time.time()
        print("Model is training in epoch {}".format(epoch))
        loss_all = 0.
        loss_cls_all = 0.
        loss_cap_all = 0.
        preds = []
        labels = []

        for batch in tqdm(train_iter, desc='Training', colour='MAGENTA'):
            model.zero_grad()
            
            # 前向传播
            if use_e2tc:
                # E2TC模型返回 (cls_logits, cap_logits)
                cls_logits, cap_logits = model(**batch)
                # NOTE: 不要在这里 .cpu()，保持在GPU上计算loss
            else:
                # 原始模型只返回 cls_logits
                cls_logits = model(**batch)
                cap_logits = None

            if config.task_name == "task_1":
                label = batch['label'].to(config.device)  # 确保 label 也在 GPU
                pred = get_preds(config, cls_logits.detach().cpu())  # 只在计算pred时移到CPU
            else:
                label = batch['type_label'].to(config.device)
                pred = get_preds_task2(config, cls_logits.detach().cpu())
            
            # 1. 计算分类 Loss (主任务) - 在 GPU 上计算
            loss_cls = loss_fn(cls_logits, label.float())
            
            # 2. 计算描述生成 Loss (辅助任务)
            loss_cap = 0.0
            if use_e2tc and cap_logits is not None:
                cap_labels = batch['cap_labels'].to(config.device)
                # Reshape: [Batch*Seq, Vocab] vs [Batch*Seq]
                vocab_size = cap_logits.size(-1)
                loss_cap = cap_loss_fn(
                    cap_logits.view(-1, vocab_size),
                    cap_labels.view(-1)
                )
                loss_cap_all += loss_cap.item()
            
            # 3. 联合 Loss
            if use_uncertainty_weighting and isinstance(loss_cap, torch.Tensor):
                # Kendall Uncertainty Weighting: 使用可学习的 σ 自动平衡损失
                total_loss, current_weights = uncertainty_module(loss_cls, loss_cap)
            elif use_e2tc and isinstance(loss_cap, torch.Tensor):
                # 固定权重模式 (原始 E2TC)
                total_loss = loss_cls + (e2tc_weight * loss_cap)
            else:
                total_loss = loss_cls

            preds.extend(pred)
            labels.extend(label.detach().cpu().numpy())

            loss_all += total_loss.item()
            loss_cls_all += loss_cls.item()
            
            model_optimizer.zero_grad()
            total_loss.backward()
            model_optimizer.step()

        end_time = time.time()
        print(" took: {:.1f} min".format((end_time - start_time)/60.))
        print("TRAINED for {} epochs".format(epoch))
        
        # 打印E2TC的loss信息
        if use_e2tc:
            print("Epoch {} - Cls Loss: {:.4f}, Cap Loss: {:.4f}, Total Loss: {:.4f}".format(
                epoch, loss_cls_all/len(train_iter), loss_cap_all/len(train_iter), loss_all/len(train_iter)
            ))
            
            # 打印 Kendall Uncertainty Weighting 学习到的参数
            if use_uncertainty_weighting:
                sigmas = uncertainty_module.get_sigmas()
                weights = uncertainty_module.get_weights()
                log_vars = uncertainty_module.get_log_vars()
                print("  Learned σ (uncertainties): cls_σ={:.4f}, cap_σ={:.4f}".format(
                    sigmas[0], sigmas[1]
                ))
                print("  Effective weights: cls_w={:.4f}, cap_w={:.4f}".format(
                    weights[0], weights[1]
                ))
                print("  log(σ²) values: cls={:.4f}, cap={:.4f}".format(
                    log_vars[0], log_vars[1]
                ))

        # 验证
        if epoch >= config.num_warm:
            # print("training loss: loss={}".format(loss_all/len(data)))
            trn_scores = get_scores(preds, labels, loss_all, len(
                train_iter), data_name="TRAIN")
            dev_scores, _ = eval(config, model, loss_fn,
                                 dev_iter, data_name='DEV')
            f = open(
                '{}/{}.all_scores.txt'.format(config.result_path, model_name), 'a')
            f.write(' ==================================================  Epoch: {}  ==================================================\n'.format(epoch))
            f.write('TrainScore: \n{}\nEvalScore: \n{}\n'.format(
                json.dumps(trn_scores), json.dumps(dev_scores)))
            max_score = save_best(config, epoch, model_name,
                                  model, dev_scores, max_score)
        print("ALLTRAINED for {} epochs".format(epoch))


def eval(config, model, loss_fn, dev_iter, data_name='DEV'):

    loss_all = 0.
    preds = []
    labels = []
    
    # 检查模型类型
    use_e2tc = isinstance(model, MHKE_E2TC)

    for batch in tqdm(dev_iter, desc='Evaling', colour='CYAN'):
        with torch.no_grad():
            # 前向传播
            if use_e2tc:
                # E2TC模型在eval时不需要计算caption loss
                # 设置training=False避免生成caption
                batch_eval = {k: v for k, v in batch.items()}
                batch_eval['training'] = False
                cls_logits, _ = model(**batch_eval)
                logit = cls_logits.cpu()
            else:
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

            # if config.task_name == "task_1":
            #     label = batch['label']
            #     pred = get_preds(config, logit)
            # else:
            #     label = batch['type_label']
            #     pred = get_preds_task2(config, logit)
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
    # results_pred = torch.max(logit_.data, 1)[0].cpu().numpy()
    # index for maximum probability
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
