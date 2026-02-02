"""
下载预训练模型脚本
运行此脚本前请先安装依赖: pip install transformers torch
"""

from transformers import (
    AutoModel, 
    AutoTokenizer, 
    AutoImageProcessor,
    BertModel,
    BertTokenizer,
    ViTModel,
    ViTImageProcessor
)
import os

# 模型保存根目录
MODEL_DIR = r"d:\models"
os.makedirs(MODEL_DIR, exist_ok=True)

def download_model(model_name, save_path, model_type="auto"):
    """下载并保存模型"""
    print(f"\n{'='*60}")
    print(f"正在下载: {model_name}")
    print(f"保存路径: {save_path}")
    print(f"{'='*60}")
    
    try:
        if model_type == "auto":
            model = AutoModel.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        elif model_type == "bert":
            model = BertModel.from_pretrained(model_name)
            tokenizer = BertTokenizer.from_pretrained(model_name)
        elif model_type == "vit":
            model = ViTModel.from_pretrained(model_name)
            tokenizer = ViTImageProcessor.from_pretrained(model_name)
        
        # 保存模型和tokenizer
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"✅ 下载成功！")
        return True
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

if __name__ == "__main__":
    print("开始下载预训练模型...")
    print("这可能需要一些时间，请耐心等待...")
    
    models_to_download = [
        {
            "name": "OFA-Sys/chinese-clip-vit-base-patch16",
            "save_path": os.path.join(MODEL_DIR, "chinese-clip-vit-base-patch16"),
            "type": "auto",
            "required": True,
            "description": "Chinese-CLIP (必需 - MHKE模型使用)"
        },
        {
            "name": "hfl/chinese-roberta-wwm-ext",
            "save_path": os.path.join(MODEL_DIR, "chinese-roberta-wwm-ext"),
            "type": "auto",
            "required": True,
            "description": "Chinese-RoBERTa (必需 - 文本编码器)"
        },
        {
            "name": "google/vit-base-patch16-224",
            "save_path": os.path.join(MODEL_DIR, "vit-base-patch16-224"),
            "type": "vit",
            "required": True,
            "description": "Vision Transformer (必需 - 图像编码器)"
        },
        {
            "name": "bert-base-chinese",
            "save_path": os.path.join(MODEL_DIR, "bert-base-chinese"),
            "type": "bert",
            "required": False,
            "description": "BERT中文 (可选 - 基线模型)"
        }
    ]
    
    success_count = 0
    for model_info in models_to_download:
        if model_info["required"]:
            status = "必需"
        else:
            status = "可选"
        
        print(f"\n[{status}] {model_info['description']}")
        
        if os.path.exists(model_info["save_path"]):
            print(f"⏭️  模型已存在，跳过下载")
            success_count += 1
            continue
        
        success = download_model(
            model_info["name"],
            model_info["save_path"],
            model_info["type"]
        )
        if success:
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"下载完成！成功: {success_count}/{len(models_to_download)}")
    print(f"{'='*60}")
    
    # 检查必需模型是否都已下载
    required_models = [m for m in models_to_download if m["required"]]
    all_required_exist = all(
        os.path.exists(m["save_path"]) for m in required_models
    )
    
    if all_required_exist:
        print("\n✅ 所有必需模型已准备就绪！")
        print("现在可以运行训练脚本: python run.py")
    else:
        print("\n⚠️  部分必需模型缺失，请重新运行此脚本")
