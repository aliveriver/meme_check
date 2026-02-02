import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import BertModel, ViTModel
from transformers import ChineseCLIPModel
import math


class PositionalEncoding(nn.Module):
    """位置编码模块 for Transformer Decoder"""
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [Batch, Seq_Len, Hidden_Dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MHKE_E2TC(nn.Module):
    """MHKE with E2TC Image Description Supervision Module"""
    def __init__(self, config):
        super().__init__()
        self.cv_path = config.vit_path
        self.nlp_path = config.chinese_roberta_path
        self.cv_model = ViTModel.from_pretrained(self.cv_path)
        self.nlp_model = BertModel.from_pretrained(self.nlp_path)
        self.attention = QKVAttention()
        self.classifier = nn.Linear(config.hidden_dim*2, config.num_classes)
        self.device = config.device
        self.weight = config.weight
        
        # --- E2TC: 图像描述监督模块 ---
        # Decoder 使用 Transformer Decoder Layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim, 
            nhead=8, 
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.caption_decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)
        
        # Token Embedding (词嵌入)
        vocab_size = self.nlp_model.config.vocab_size
        self.cap_embedding = nn.Embedding(vocab_size, config.hidden_dim)
        
        # Positional Encoding (位置编码)
        self.cap_pos_encoding = PositionalEncoding(config.hidden_dim, max_len=config.pad_size)
        
        # 解码头：将向量映射回词表
        self.cap_head = nn.Linear(config.hidden_dim, vocab_size)
        
        # E2TC Loss权重系数
        self.e2tc_weight = config.e2tc_weight if hasattr(config, 'e2tc_weight') else 1.0

    def forward(self, **args):
        # 获取是否处于训练模式
        training = args.get('training', self.training)
        
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

        # --- A. 分类任务 (Classification Branch) ---
        features = torch.cat(
            (text_with_k, image_outputs['pooler_output']), dim=1)
        cls_logits = self.classifier(features)
        
        # --- B. E2TC 监督任务 (Supervision Branch) ---
        cap_logits = None
        if training and 'cap_input_ids' in args:
            # 获取图像序列特征作为 Decoder 的 memory
            # last_hidden_state: [Batch, Seq_Len, Hidden_Dim]
            image_seq_feats = image_outputs.last_hidden_state
            
            # 准备 Decoder 输入
            cap_input_ids = args['cap_input_ids'].to(self.device)
            
            # Token Embedding
            tgt_emb = self.cap_embedding(cap_input_ids)  # [Batch, Seq_Len, Hidden_Dim]
            
            # 添加位置编码
            tgt_emb = self.cap_pos_encoding(tgt_emb)
            
            # 🔧 关键修复：生成 Causal Mask (防止看到未来的token)
            seq_len = tgt_emb.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(self.device)
            
            # 🔧 关键修复：生成 Padding Mask (防止 Attention 关注 padding token)
            # tgt_key_padding_mask: [Batch, Seq_Len], True 表示该位置是 padding
            # 获取 tokenizer 的 pad_token_id
            if hasattr(self.nlp_model.config, 'pad_token_id'):
                pad_token_id = self.nlp_model.config.pad_token_id
            else:
                pad_token_id = 0  # 默认 padding token id
            
            tgt_key_padding_mask = (cap_input_ids == pad_token_id)  # [Batch, Seq_Len]
            
            # Transformer Decoder: tgt=文字, memory=图像序列特征
            decoder_out = self.caption_decoder(
                tgt=tgt_emb,
                memory=image_seq_feats,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask  # 🔧 添加 padding mask
            )
            
            # 预测词表概率
            cap_logits = self.cap_head(decoder_out)  # [Batch, Seq_Len, Vocab_Size]
        
        return cls_logits, cap_logits


class MHKE(nn.Module):
    """Original MHKE Model (without E2TC)"""
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
