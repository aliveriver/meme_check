import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import BertModel, ViTModel
from transformers import ChineseCLIPModel


class MHKE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cv_path = config.vit_path
        self.nlp_path = config.chinese_roberta_path
        self.cv_model = ViTModel.from_pretrained(self.cv_path)
        self.nlp_model = BertModel.from_pretrained(self.nlp_path)
        # self.dropout = nn.Dropout(p=0.1)
        self.attention = QKVAttention()
        self.classifier = nn.Linear(config.hidden_dim*2, config.num_classes)
        self.device = config.device
        self.weight = config.weight

    def forward(self, **args):
        text_outputs = self.nlp_model(input_ids=args['input_ids'].to(self.device),
                                      attention_mask=args['attention_mask'].to(self.device))
        text_discription_outputs = self.nlp_model(input_ids=args['text_discription_input_ids'].to(self.device),
                                                  attention_mask=args['text_discription_attention_mask'].to(self.device))
        meme_discription_outputs = self.nlp_model(input_ids=args['meme_discription_input_ids'].to(self.device),
                                                  attention_mask=args['meme_discription_attention_mask'].to(self.device))

        # text_features = self.dropout(text_outputs['pooler_output'])
        # text_discription_features = self.dropout(
        #     text_discription_outputs['pooler_output'])
        # meme_discription_features = self.dropout(
        #     meme_discription_outputs['pooler_output'])

        # text_with_k_s = self.attention(text_outputs['pooler_output'],
        #                                text_discription_outputs['pooler_output'], text_outputs['pooler_output'])[0]
        # text_with_k_v = self.attention(text_outputs['pooler_output'],
        #                                meme_discription_outputs['pooler_output'], text_outputs['pooler_output'])[0]

        # text_with_k = torch.mean(torch.stack(
        #     [text_outputs['pooler_output'], text_with_k_s]), dim=0)
        text_with_k = text_outputs['pooler_output'] + \
            self.weight * meme_discription_outputs['pooler_output'] + text_discription_outputs['pooler_output']

        # text_with_k = text_outputs['pooler_output'] + \
        #     text_discription_outputs['pooler_output']
        # text_with_k = text_outputs['pooler_output'] + \
        #     meme_discription_outputs['pooler_output']
        # text_with_k = text_outputs['pooler_output'] + \
        #     text_discription_outputs['pooler_output'] + \
        #     meme_discription_outputs['pooler_output']

        image_outputs = self.cv_model(
            pixel_values=args['image_tensor'].to(self.device))

        features = torch.cat(
            (text_with_k, image_outputs['pooler_output']), dim=1)

        output = self.classifier(features)
        return output


class MHKE_CLIP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.path = config.chinese_clip_path
        self.model = ChineseCLIPModel.from_pretrained(self.path)
        self.attention = QKVAttention()
        self.classifier = nn.Linear(config.hidden_dim*2, config.num_classes)
        self.device = config.device

    def forward(self, **args):
        text_features = self.model.get_text_features(input_ids=args['input_ids'].to(self.device),
                                                     attention_mask=args['attention_mask'].to(self.device))
        text_discription_features = self.model.get_text_features(input_ids=args['text_discription_input_ids'].to(self.device),
                                                                 attention_mask=args['text_discription_attention_mask'].to(self.device))
        meme_discription_features = self.model.get_text_features(input_ids=args['meme_discription_input_ids'].to(self.device),
                                                                 attention_mask=args['meme_discription_attention_mask'].to(self.device))

        # text_with_k_v = self.attention(text_features,
        #                                meme_discription_features, text_features)[0]

        # text_with_k = text_features + text_discription_features
        # text_with_k = text_features + meme_discription_features
        text_with_k = text_features + text_discription_features + meme_discription_features

        image_features = self.model.get_image_features(
            pixel_values=args['image_tensor'].to(self.device))

        features = torch.cat((text_with_k, image_features), dim=1)
        output = self.classifier(features)
        return output


class CrossModalAttention(nn.Module):
    """
    交叉模态注意力模块
    让一个模态的特征"关注"另一个模态的特征
    """
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key_value):
        """
        query: 查询模态特征 [batch, hidden_dim]
        key_value: 被关注模态特征 [batch, hidden_dim]
        返回: 增强后的query特征, 注意力权重
        """
        # 扩展维度以适配 MultiheadAttention
        if query.dim() == 2:
            query = query.unsqueeze(1)  # [batch, 1, hidden_dim]
        if key_value.dim() == 2:
            key_value = key_value.unsqueeze(1)  # [batch, 1, hidden_dim]
            
        # Cross-attention: Q来自query模态，K和V来自key_value模态
        attn_output, attn_weights = self.multihead_attn(
            query=query,
            key=key_value,
            value=key_value
        )
        
        # 残差连接 + LayerNorm
        attn_output = self.dropout(attn_output)
        output = self.layer_norm(query + attn_output)
        
        return output.squeeze(1), attn_weights


class MHKE_CrossAttention(nn.Module):
    """
    使用交叉注意力的多模态知识增强检测器
    - 保留 GPT-4V 生成的知识描述增强
    - 使用双向交叉注意力进行多模态融合
    - 冻结预训练模型参数以防止过拟合
    """
    def __init__(self, config):
        super().__init__()
        self.cv_path = config.vit_path
        self.nlp_path = config.chinese_roberta_path
        self.cv_model = ViTModel.from_pretrained(self.cv_path)
        self.nlp_model = BertModel.from_pretrained(self.nlp_path)
        
        # 🧊 冻结预训练模型参数
        for param in self.cv_model.parameters():
            param.requires_grad = False
        for param in self.nlp_model.parameters():
            param.requires_grad = False
        print("✓ Pretrained ViT and RoBERTa parameters frozen")
        
        # 双向交叉注意力 (Dropout 增加到 0.3)
        self.text_to_image_attn = CrossModalAttention(config.hidden_dim, num_heads=8, dropout=0.3)
        self.image_to_text_attn = CrossModalAttention(config.hidden_dim, num_heads=8, dropout=0.3)
        
        # 融合层 (Dropout 增加到 0.3)
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Linear(config.hidden_dim, config.num_classes)
        self.device = config.device
        self.weight = config.weight

    def forward(self, **args):
        # 1. 提取文本特征（包含知识增强）
        text_outputs = self.nlp_model(
            input_ids=args['input_ids'].to(self.device),
            attention_mask=args['attention_mask'].to(self.device)
        )
        text_discription_outputs = self.nlp_model(
            input_ids=args['text_discription_input_ids'].to(self.device),
            attention_mask=args['text_discription_attention_mask'].to(self.device)
        )
        meme_discription_outputs = self.nlp_model(
            input_ids=args['meme_discription_input_ids'].to(self.device),
            attention_mask=args['meme_discription_attention_mask'].to(self.device)
        )
        
        # 知识增强：融合原始文本和描述
        text_features = text_outputs['pooler_output'] + \
            self.weight * meme_discription_outputs['pooler_output'] + \
            text_discription_outputs['pooler_output']
        
        # 2. 提取图像特征
        image_outputs = self.cv_model(
            pixel_values=args['image_tensor'].to(self.device)
        )
        image_features = image_outputs['pooler_output']
        
        # 3. 双向交叉注意力融合
        text_enhanced, _ = self.text_to_image_attn(text_features, image_features)   # 文本关注图像
        image_enhanced, _ = self.image_to_text_attn(image_features, text_features)  # 图像关注文本
        
        # 4. 特征融合
        fused_features = torch.cat((text_enhanced, image_enhanced), dim=1)
        fused_features = self.fusion_layer(fused_features)
        
        # 5. 分类
        output = self.classifier(fused_features)
        return output


class QKVAttention(nn.Module):
    def __init__(self):
        super(QKVAttention, self).__init__()

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        matmul_qk = torch.matmul(Q, K.transpose(-2, -1))
        dk = K.shape[-1]
        scaled_attention_logits = matmul_qk / \
            torch.sqrt(torch.tensor(dk).float())

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

    def forward(self, Q, K, V, mask=None):
        output, attention_weights = self.scaled_dot_product_attention(
            Q, K, V, mask)
        return output, attention_weights
