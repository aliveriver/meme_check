import os
import sys
import glob
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.Config_base import Config_base
from dataset.dataset import MemeDataset
from model.clip import MHKE_CLIP
from train_eval_ import get_preds, get_preds_task2

def find_best_checkpoint(checkpoint_dir, task_name):
    """查找最新生成的对应 task 的 clip 模型最佳 checkpoint"""
    # 匹配 run_comparison.py 产生的命名格式，或者基础的 ckp-*clip*-BEST.tar
    search_pattern = os.path.join(checkpoint_dir, f'*clip*{task_name}*BEST.tar')
    checkpoints = glob.glob(search_pattern)
    
    if not checkpoints:
        # 尝试更宽泛的搜索
        search_pattern = os.path.join(checkpoint_dir, f'*clip*BEST.tar')
        checkpoints = glob.glob(search_pattern)
        
    if not checkpoints:
        print(f"❌ 未找到对应 {task_name} 的模型 checkpoint: {search_pattern}")
        return None
        
    # 按时间排序，取最新的
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    best_ckp = checkpoints[0]
    print(f"✅ 找到最新 checkpoint: {best_ckp}")
    return best_ckp

def analyze_errors(task_name="task_1"):
    print(f"\n{'='*70}")
    print(f"  开始对 MHKE_CLIP 在 {task_name} 上的预测进行错误分析  ")
    print(f"{'='*70}\n")
    
    # 1. 初始化配置 (与 run_comparison 中保持一致)
    config = Config_base("clip", task_name)
    config.hidden_dim = 512  # CLIP的设定
    
    # 2. 获取对应的测试数据
    print("正在加载数据集...")
    test_data = MemeDataset(config, training=False)
    test_iter = DataLoader(test_data, batch_size=int(config.batch_size), shuffle=False)
    
    # 读取原始 JSON 方便提取原始文本和知识
    df_test = pd.read_json(config.test_path)
    if len(df_test) != len(test_data):
        print("警告：JSON数据行数与Dataset长度不匹配！")
        
    # 3. 初始化模型并加载权重
    model = MHKE_CLIP(config).to(config.device)
    ckp_path = find_best_checkpoint(config.checkpoint_path, task_name)
    if not ckp_path:
        return
        
    print("正在加载模型权重...")
    checkpoint = torch.load(ckp_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 4. 运行推理，收集预测结果
    preds = []
    labels = []
    
    print("正在测试集上运行推理...")
    for batch in tqdm(test_iter, desc='Inferencing', colour='CYAN'):
        with torch.no_grad():
            logit = model(**batch).cpu()
            if config.task_name == "task_1":
                label = batch['label']
                pred = get_preds(config, logit)
            else:
                label = batch['type_label']
                pred = get_preds_task2(config, logit)
                
            preds.extend(pred)
            labels.extend(label.detach().numpy())
    
    # 5. 对比提取错误案例
    error_cases = []
    for i in range(len(preds)):
        pred_label = preds[i]
        true_label = int(labels[i])
        
        # 将 one-hot list 转换回 int index（如果预测是 one-hot）
        if isinstance(pred_label, list):
            pred_idx = pred_label.index(1) if 1 in pred_label else 0
        else:
            pred_idx = pred_label
            
        if pred_idx != true_label:
            # 拿到原始数据，构建记录
            row = df_test.iloc[i]
            
            # 判断错误类型
            if task_name == "task_1":
                err_type = "False Positive (无害误判为有害)" if pred_idx == 1 else "False Negative (有害漏判为无害)"
            else:
                err_type = f"Confused (实为 {true_label} -> 错判为 {pred_idx})"
                
            error_cases.append({
                "Index": i,
                "Image_Path": row.get('new_path', ''),
                "Error_Type": err_type,
                "True_Label": true_label,
                "Pred_Label": pred_idx,
                "Text": row.get('text', ''),
                "Text_Description": row.get('text_discription', ''),
                "Meme_Description": row.get('meme_discription', '')
            })
            
    # 6. 导出结果
    if not error_cases:
        print("🎉 恭喜，所有测试集样本全部预测正确！")
        return
        
    df_errors = pd.DataFrame(error_cases)
    out_dir = "result"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"MHKE_CLIP_Error_Analysis_{task_name}.csv")
    
    # 为了防止中文乱码，加 utf-8-sig
    df_errors.to_csv(out_file, index=False, encoding="utf-8-sig")
    
    print(f"\n✅ 分析完成！共找到 {len(error_cases)} 个被 MHKE_CLIP 错判的样本。")
    print(f"✅ 错误明细已导出至: {out_file}")
    
    if task_name == "task_1":
        fps = sum(1 for e in error_cases if e["Pred_Label"] == 1)
        fns = sum(1 for e in error_cases if e["Pred_Label"] == 0)
        print(f"📊 其中包含假阳性 (False Positives) 样本: {fps} 个")
        print(f"📊 包含假阴性 (False Negatives) 样本: {fns} 个")
    else:
        print("📊 (请在 CSV 中按 True_Label 或 Pred_Label 筛选，观察具体的分类混淆边界)")

if __name__ == "__main__":
    task = "task_1"
    if len(sys.argv) > 1:
        task = sys.argv[1]
    
    if task not in ["task_1", "task_2"]:
        print("任务类型仅支持 task_1 或 task_2")
        sys.exit(1)
        
    analyze_errors(task)
