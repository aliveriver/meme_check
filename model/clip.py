import torch
import torch.nn as nn
from transformers import ChineseCLIPModel


def freeze_clip_layers(model, freeze_layers=10):
    """
    冻结 ChineseCLIPModel 的底层参数，只训练顶部几层和投影层。
    
    ChineseCLIP 结构:
      - text_model: 12层 Transformer (基于 BERT)
      - vision_model: 12层 ViT
      - text_projection / visual_projection: 投影层(始终可训练)
    
    Args:
        model: ChineseCLIPModel 实例
        freeze_layers: 冻结前 N 层 (共12层), 设为 0 则不冻结
    """
    if freeze_layers <= 0:
        return
    
    # 冻结视觉模型嵌入层 + 底部 N 层
    for param in model.vision_model.embeddings.parameters():
        param.requires_grad = False
    for i, layer in enumerate(model.vision_model.encoder.layers):
        if i < freeze_layers:
            for param in layer.parameters():
                param.requires_grad = False
    
    # 冻结文本模型嵌入层 + 底部 N 层
    for param in model.text_model.embeddings.parameters():
        param.requires_grad = False
    for i, layer in enumerate(model.text_model.encoder.layer):
        if i < freeze_layers:
            for param in layer.parameters():
                param.requires_grad = False
    
    # 投影层始终可训练（不冻结）
    for param in model.text_projection.parameters():
        param.requires_grad = True
    for param in model.visual_projection.parameters():
        param.requires_grad = True
    
    # 统计
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    print(f"  [CLIP freeze] 总参数: {total:,} | 冻结: {frozen:,} | 可训练: {trainable:,} "
          f"({trainable/total*100:.1f}%)")


def freeze_bert_vit_layers(bert_model, vit_model, freeze_layers=10):
    """
    冻结 BertModel + ViTModel 的底层参数。
    
    Args:
        bert_model: BertModel 实例
        vit_model: ViTModel 实例
        freeze_layers: 冻结前 N 层 (共12层)
    """
    if freeze_layers <= 0:
        return
    
    # 冻结 ViT
    for param in vit_model.embeddings.parameters():
        param.requires_grad = False
    for i, layer in enumerate(vit_model.encoder.layer):
        if i < freeze_layers:
            for param in layer.parameters():
                param.requires_grad = False
    
    # 冻结 BERT/RoBERTa
    for param in bert_model.embeddings.parameters():
        param.requires_grad = False
    for i, layer in enumerate(bert_model.encoder.layer):
        if i < freeze_layers:
            for param in layer.parameters():
                param.requires_grad = False
    
    total_vit = sum(p.numel() for p in vit_model.parameters())
    trainable_vit = sum(p.numel() for p in vit_model.parameters() if p.requires_grad)
    total_bert = sum(p.numel() for p in bert_model.parameters())
    trainable_bert = sum(p.numel() for p in bert_model.parameters() if p.requires_grad)
    print(f"  [ViT freeze]  总参数: {total_vit:,} | 可训练: {trainable_vit:,} ({trainable_vit/total_vit*100:.1f}%)")
    print(f"  [BERT freeze] 总参数: {total_bert:,} | 可训练: {trainable_bert:,} ({trainable_bert/total_bert*100:.1f}%)")


class CLIPMemesClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.path = config.clip_path
        self.model = ChineseCLIPModel.from_pretrained(self.path)
        
        # 冻结底层参数
        freeze_layers = getattr(config, 'freeze_layers', 10)
        print(f"✓ CLIPMemesClassifier: 冻结前 {freeze_layers}/12 层")
        freeze_clip_layers(self.model, freeze_layers)
        
        # 带 Dropout 的分类头
        clf_dropout = getattr(config, 'classifier_dropout', 0.3)
        self.classifier = nn.Sequential(
            nn.Dropout(clf_dropout),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(clf_dropout),
            nn.Linear(config.hidden_dim, config.num_classes)
        )
        self.device = config.device
  
    def forward(self, **args):
        text_features = self.model.get_text_features(input_ids = args['input_ids'].to(self.device),
                                                    attention_mask = args['attention_mask'].to(self.device))
        image_features = self.model.get_image_features(pixel_values = args['image_tensor'].to(self.device))

        features = torch.cat((text_features, image_features), dim=1)
        output = self.classifier(features)
        return output