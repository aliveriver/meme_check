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


class SequenceCrossAttention(nn.Module):
    """
    序列级交叉注意力模块
    让一个模态的完整序列"关注"另一个模态的完整序列
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
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, query_seq, key_value_seq, query_mask=None, kv_mask=None):
        """
        query_seq: 查询模态序列 [batch, seq_len_q, hidden_dim]
        key_value_seq: 被关注模态序列 [batch, seq_len_kv, hidden_dim]
        query_mask: 查询序列的attention mask [batch, seq_len_q]
        kv_mask: KV序列的attention mask [batch, seq_len_kv]
        返回: 增强后的query序列 [batch, seq_len_q, hidden_dim]
        """
        # 创建 key_padding_mask (True 表示 padding 位置)
        key_padding_mask = None
        if kv_mask is not None:
            key_padding_mask = (kv_mask == 0)  # [batch, seq_len_kv]
            
        # Cross-attention
        attn_output, _ = self.multihead_attn(
            query=query_seq,
            key=key_value_seq,
            value=key_value_seq,
            key_padding_mask=key_padding_mask
        )
        
        # 残差连接 + LayerNorm
        attn_output = self.dropout(attn_output)
        hidden = self.layer_norm(query_seq + attn_output)
        
        # FFN + 残差
        ffn_output = self.ffn(hidden)
        output = self.layer_norm2(hidden + ffn_output)
        
        return output


class MHKE_CrossAttention(nn.Module):
    """
    使用序列级交叉注意力的多模态知识增强检测器
    - 使用 last_hidden_state (完整序列) 而非 pooler_output
    - 文本序列的每个 token 关注图像的每个 patch
    - 图像序列的每个 patch 关注文本的每个 token
    """
    def __init__(self, config):
        super().__init__()
        self.cv_path = config.vit_path
        self.nlp_path = config.chinese_roberta_path
        self.cv_model = ViTModel.from_pretrained(self.cv_path)
        self.nlp_model = BertModel.from_pretrained(self.nlp_path)
        
        # 🧊 部分冻结：只冻结底层，保留顶层可训练
        freeze_layers = 8  # 冻结前 8 层（共 12 层），顶部 4 层可训练
        
        # 冻结 ViT embeddings 和底层
        for param in self.cv_model.embeddings.parameters():
            param.requires_grad = False
        for i, layer in enumerate(self.cv_model.encoder.layer):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        
        # 冻结 RoBERTa embeddings 和底层
        for param in self.nlp_model.embeddings.parameters():
            param.requires_grad = False
        for i, layer in enumerate(self.nlp_model.encoder.layer):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        
        # 统计可训练参数
        trainable_params = sum(p.numel() for p in self.cv_model.parameters() if p.requires_grad)
        trainable_params += sum(p.numel() for p in self.nlp_model.parameters() if p.requires_grad)
        print(f"✓ Partial freezing: bottom {freeze_layers} layers frozen, top {12-freeze_layers} layers trainable")
        print(f"  Trainable params in pretrained models: {trainable_params:,}")
        
        # 序列级双向交叉注意力
        self.text_to_image_attn = SequenceCrossAttention(config.hidden_dim, num_heads=8, dropout=0.2)
        self.image_to_text_attn = SequenceCrossAttention(config.hidden_dim, num_heads=8, dropout=0.2)
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        self.classifier = nn.Linear(config.hidden_dim, config.num_classes)
        self.device = config.device
        self.weight = config.weight
        
        print(f"✓ Sequence-level cross-attention enabled")

    def mean_pooling(self, hidden_states, attention_mask=None):
        """对序列进行平均池化"""
        if attention_mask is not None:
            # 使用 attention_mask 进行加权平均
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            return sum_hidden / sum_mask
        else:
            return hidden_states.mean(dim=1)

    def forward(self, **args):
        # 1. 提取文本序列特征
        text_outputs = self.nlp_model(
            input_ids=args['input_ids'].to(self.device),
            attention_mask=args['attention_mask'].to(self.device)
        )
        text_seq = text_outputs['last_hidden_state']  # [batch, 64, 768]
        text_mask = args['attention_mask'].to(self.device)
        
        # 知识描述也用序列
        text_desc_outputs = self.nlp_model(
            input_ids=args['text_discription_input_ids'].to(self.device),
            attention_mask=args['text_discription_attention_mask'].to(self.device)
        )
        meme_desc_outputs = self.nlp_model(
            input_ids=args['meme_discription_input_ids'].to(self.device),
            attention_mask=args['meme_discription_attention_mask'].to(self.device)
        )
        
        # 知识增强：在序列级别融合（使用 pooler 进行加权）
        text_seq = text_seq + \
            self.weight * meme_desc_outputs['pooler_output'].unsqueeze(1) + \
            text_desc_outputs['pooler_output'].unsqueeze(1)
        
        # 2. 提取图像序列特征
        image_outputs = self.cv_model(
            pixel_values=args['image_tensor'].to(self.device)
        )
        image_seq = image_outputs['last_hidden_state']  # [batch, 197, 768] (1 CLS + 196 patches)
        
        # 3. 序列级双向交叉注意力
        # 文本序列关注图像序列
        text_enhanced_seq = self.text_to_image_attn(text_seq, image_seq, query_mask=text_mask)
        # 图像序列关注文本序列
        image_enhanced_seq = self.image_to_text_attn(image_seq, text_seq, kv_mask=text_mask)
        
        # 4. 池化：将序列压缩为单个向量
        text_pooled = self.mean_pooling(text_enhanced_seq, text_mask)  # [batch, 768]
        image_pooled = image_enhanced_seq[:, 0, :]  # 使用 CLS token [batch, 768]
        # 5. 特征融合
        fused_features = torch.cat((text_pooled, image_pooled), dim=1)  # [batch, 1536]
        fused_features = self.fusion_layer(fused_features)  # [batch, 768]
        
        # 6. 分类
        output = self.classifier(fused_features)
        return output


class Combiner(nn.Module):
    """
    ISSUES 论文中的 Combiner 网络
    用于融合文本特征和图像伪词特征
    """
    def __init__(self, hidden_dim, projection_dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(projection_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, projection_dim)
        self.layer_norm = nn.LayerNorm(projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        
    def forward(self, text_proj, image_as_text):
        """
        text_proj: 投影后的文本特征 [batch, projection_dim]
        image_as_text: 图像映射的伪词特征 [batch, projection_dim]
        返回: 融合后的特征 [batch, projection_dim]
        """
        combined = torch.cat([text_proj, image_as_text], dim=-1)
        hidden = self.gelu(self.fc1(combined))
        hidden = self.dropout(hidden)
        output = self.fc2(hidden)
        # 残差连接 + LayerNorm
        output = self.layer_norm(output + text_proj)
        return output


class MHKE_ISSUES(nn.Module):
    """
    融合 MHKE 和 ISSUES 论文方法的多模态检测器
    
    核心思想：
    1. MHKE 的知识增强：融合 GPT-4V 生成的描述
    2. ISSUES 的解耦投影：为图像和文本创建独立的投影空间
    3. ISSUES 的伪词映射：将图像特征映射到文本嵌入空间
    4. ISSUES 的 Combiner：融合文本和图像伪词特征
    
    支持两阶段训练：
    - Stage 1: 只训练图像投影层
    - Stage 2: 联合训练所有投影层、Combiner 和分类器
    """
    def __init__(self, config, training_stage=2):
        super().__init__()
        self.cv_path = config.vit_path
        self.nlp_path = config.chinese_roberta_path
        self.cv_model = ViTModel.from_pretrained(self.cv_path)
        self.nlp_model = BertModel.from_pretrained(self.nlp_path)
        
        self.hidden_dim = config.hidden_dim  # 768
        self.projection_dim = 256  # 投影后的维度
        self.device = config.device
        self.weight = config.weight
        self.training_stage = training_stage
        
        # 🧊 冻结预训练模型的底层
        freeze_layers = 8
        for param in self.cv_model.embeddings.parameters():
            param.requires_grad = False
        for i, layer in enumerate(self.cv_model.encoder.layer):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        for param in self.nlp_model.embeddings.parameters():
            param.requires_grad = False
        for i, layer in enumerate(self.nlp_model.encoder.layer):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        
        # ===== ISSUES 核心组件 =====
        
        # 1. 解耦投影层 (Disentangled Projections)
        # 将图像和文本投影到各自独立的空间，避免 CLIP 对齐空间的局限
        self.image_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.projection_dim)
        )
        
        self.text_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.projection_dim)
        )
        
        # 2. 图像到伪词映射 (Image-to-Pseudo-Word Mapping)
        # 将图像特征映射到文本嵌入空间，作为"伪词"
        self.image_to_text_mapper = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.projection_dim),
            nn.LayerNorm(self.projection_dim)
        )
        
        # 3. Combiner 网络
        self.combiner = Combiner(self.hidden_dim, self.projection_dim, dropout=0.2)
        
        # 4. 分类器 (使用投影维度)
        self.classifier = nn.Sequential(
            nn.Linear(self.projection_dim * 2, self.projection_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.projection_dim, config.num_classes)
        )
        
        # 设置训练阶段
        self._set_training_stage(training_stage)
        
        print(f"✓ MHKE_ISSUES initialized with training_stage={training_stage}")
        print(f"  Projection dim: {self.projection_dim}")
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Trainable parameters: {trainable:,}")
    
    def _set_training_stage(self, stage):
        """设置训练阶段"""
        self.training_stage = stage
        if stage == 1:
            # 阶段1：只训练图像投影层
            for param in self.text_projection.parameters():
                param.requires_grad = False
            for param in self.image_to_text_mapper.parameters():
                param.requires_grad = False
            for param in self.combiner.parameters():
                param.requires_grad = False
            print("  Stage 1: Only image_projection is trainable")
        else:
            # 阶段2：所有投影层都可训练
            for param in self.text_projection.parameters():
                param.requires_grad = True
            for param in self.image_to_text_mapper.parameters():
                param.requires_grad = True
            for param in self.combiner.parameters():
                param.requires_grad = True
            print("  Stage 2: All projections and combiner are trainable")
    
    def set_stage(self, stage):
        """外部调用切换训练阶段"""
        self._set_training_stage(stage)
    
    def forward(self, **args):
        # ===== 1. 提取原始特征 =====
        
        # 文本特征（带知识增强）
        text_outputs = self.nlp_model(
            input_ids=args['input_ids'].to(self.device),
            attention_mask=args['attention_mask'].to(self.device)
        )
        text_desc_outputs = self.nlp_model(
            input_ids=args['text_discription_input_ids'].to(self.device),
            attention_mask=args['text_discription_attention_mask'].to(self.device)
        )
        meme_desc_outputs = self.nlp_model(
            input_ids=args['meme_discription_input_ids'].to(self.device),
            attention_mask=args['meme_discription_attention_mask'].to(self.device)
        )
        
        # MHKE 知识增强
        text_features = text_outputs['pooler_output'] + \
            self.weight * meme_desc_outputs['pooler_output'] + \
            text_desc_outputs['pooler_output']  # [batch, 768]
        
        # 图像特征
        image_outputs = self.cv_model(
            pixel_values=args['image_tensor'].to(self.device)
        )
        image_features = image_outputs['pooler_output']  # [batch, 768]
        
        # ===== 2. 解耦投影 =====
        
        # 投影到独立空间
        text_proj = self.text_projection(text_features)    # [batch, 256]
        image_proj = self.image_projection(image_features)  # [batch, 256]
        
        # ===== 3. 图像→伪词映射 =====
        
        # 将图像映射到文本空间的"伪词"
        image_as_text = self.image_to_text_mapper(image_features)  # [batch, 256]
        
        # ===== 4. Combiner 融合 =====
        
        # 用 Combiner 融合文本特征和图像伪词
        combined_text = self.combiner(text_proj, image_as_text)  # [batch, 256]
        
        # ===== 5. 最终融合与分类 =====
        
        # 拼接：融合后的文本 + 图像投影
        final_features = torch.cat([combined_text, image_proj], dim=-1)  # [batch, 512]
        
        # 分类
        output = self.classifier(final_features)
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
