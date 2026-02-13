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
from transformers import BertTokenizer
from dataset.dataset import get_time_dif, convert_onehot
from model.clip import *
from model.vit_roberta import *
from model.MHKE import *


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

    # Label Smoothing: 缓解过拟合
    label_smoothing = getattr(config, 'label_smoothing', 0.1)
    cot_loss_weight = getattr(config, 'cot_loss_weight', 0.5)
    print(f"✓ Using Label Smoothing: {label_smoothing}")
    print(f"✓ CoT loss weight: {cot_loss_weight}")
    
    # CoT loss: CrossEntropy (ignore padding token 0)
    cot_loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    
    def label_smoothing_loss(logits, labels, smoothing=0.1):
        """
        Label Smoothing for BCEWithLogitsLoss
        将标签从 0/1 平滑为 smoothing/2 和 1-smoothing/2
        例如 smoothing=0.1 时: 0 -> 0.05, 1 -> 0.95
        """
        with torch.no_grad():
            smoothed_labels = labels * (1.0 - smoothing) + smoothing / 2
        return nn.functional.binary_cross_entropy_with_logits(logits, smoothed_labels)
    
    loss_fn = lambda logits, labels: label_smoothing_loss(logits, labels, label_smoothing)
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
            output = model(**batch)
            
            # 兼容 dict 输出和纯 tensor 输出
            if isinstance(output, dict):
                logit = output['logit'].cpu()
            else:
                logit = output.cpu()

            if config.task_name == "task_1":
                label = batch['label']
                pred = get_preds(config, logit)
            else:
                label = batch['type_label']
                pred = get_preds_task2(config, logit)
            cls_loss = loss_fn(logit, label.float())
            
            # CoT loss
            total_loss = cls_loss
            if isinstance(output, dict) and 'cot_logits' in output:
                cot_logits = output['cot_logits']  # [batch, seq_len, vocab_size]
                cot_labels = batch['cot_input_ids'].to(cot_logits.device)
                # Shift: 预测下一个 token
                shift_logits = cot_logits[:, :-1, :].contiguous()
                shift_labels = cot_labels[:, 1:].contiguous()
                cot_loss = cot_loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                total_loss = cls_loss + cot_loss_weight * cot_loss

            preds.extend(pred)
            labels.extend(label.detach().numpy())

            loss_all += total_loss.item()
            model_optimizer.zero_grad()
            total_loss.backward()
            model_optimizer.step()

        end_time = time.time()
        print(" took: {:.1f} min".format((end_time - start_time)/60.))
        print("TRAINED for {} epochs".format(epoch))

        # 验证
        if epoch >= config.num_warm:
            # print("training loss: loss={}".format(loss_all/len(data)))
            trn_scores = get_scores(preds, labels, loss_all, len(
                train_iter), data_name="TRAIN")
            dev_scores, _, cot_texts = eval(config, model, loss_fn,
                                 dev_iter, data_name='DEV')
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
                # 保存最佳模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                }, '{}/ckp-{}-{}.tar'.format(config.checkpoint_path, model_name, 'BEST'))
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
    cot_texts = []  # 收集生成的 CoT 文本

    # 用于 CoT 生成的 tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.roberta_path)

    for batch in tqdm(dev_iter, desc='Evaling', colour='CYAN'):
        with torch.no_grad():
            output = model(**batch)
            
            if isinstance(output, dict):
                logit = output['logit'].cpu()
                # CoT 生成
                if hasattr(model, 'cot_decoder'):
                    fused = output.get('fused_features', None)
                    if fused is not None:
                        batch_cot = model.cot_decoder.generate(fused.to(config.device), tokenizer)
                        cot_texts.extend(batch_cot)
            else:
                logit = output.cpu()

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
    
    # 打印几条 CoT 示例
    if cot_texts:
        print(f"\n--- CoT 生成示例 (\u5171 {len(cot_texts)} 条) ---")
        for i, txt in enumerate(cot_texts[:3]):
            print(f"  [{i}] {txt[:200]}")
        print("---")

    return dev_scores, preds, cot_texts


def test(model, dev_iter):

    preds = []
    labels = []
    cot_texts = []

    # 用于 CoT 生成的 tokenizer (如果模型有 cot_decoder)
    tokenizer = None
    if hasattr(model, 'cot_decoder'):
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(model.nlp_path)

    for batch in tqdm(dev_iter, desc='Testing', colour='CYAN'):
        with torch.no_grad():
            output = model(**batch)

            if isinstance(output, dict):
                logit = output['logit'].cpu()
                # CoT 生成
                if tokenizer is not None:
                    fused = output.get('fused_features', None)
                    if fused is not None:
                        batch_cot = model.cot_decoder.generate(fused.to(model.device), tokenizer)
                        cot_texts.extend(batch_cot)
            else:
                logit = output.cpu()

            label = batch['label']
            pred = output_preds(logit)

            preds.extend(pred)
            labels.extend(label.detach().numpy())

        df = pd.DataFrame({'new_pred': preds})
        if cot_texts:
            # 对齐长度
            df['cot_reason'] = cot_texts[:len(preds)] + [''] * max(0, len(preds) - len(cot_texts))
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
