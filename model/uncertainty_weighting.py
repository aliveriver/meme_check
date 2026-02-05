"""
Kendall Uncertainty Weighting for Multi-Task Learning

基于论文 "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
(Kendall et al., CVPR 2018)

核心思想: 使用任务的同方差不确定性 (Homoscedastic Uncertainty) 来自动学习多任务损失的权重

公式: L_total = (1 / (2 * σ₁²)) * L₁ + (1 / (2 * σ₂²)) * L₂ + log(σ₁) + log(σ₂)

为了数值稳定性，我们学习 log(σ²) 而不是直接学习 σ:
    L_total = (1/2) * exp(-log_var₁) * L₁ + (1/2) * exp(-log_var₂) * L₂ + (1/2) * log_var₁ + (1/2) * log_var₂
"""

import torch
import torch.nn as nn


class MultiTaskUncertaintyWeighting(nn.Module):
    """
    Kendall's Homoscedastic Uncertainty Weighting Module
    
    自动学习多任务损失的权重，通过学习每个任务的 log(σ²)
    
    参数:
        num_tasks: 任务数量
        init_log_vars: 初始化 log(σ²) 的值列表，默认为 [0.0] * num_tasks
                       较大的初始值 -> 较小的初始权重
                       较小的初始值 -> 较大的初始权重
    
    使用示例:
        uncertainty = MultiTaskUncertaintyWeighting(num_tasks=2)
        total_loss = uncertainty(loss_cls, loss_cap)
    """
    
    def __init__(self, num_tasks=2, init_log_vars=None):
        super().__init__()
        
        if init_log_vars is None:
            # 默认初始化: log(σ²) = 0 -> σ² = 1 -> 权重约为 0.5
            init_log_vars = [0.0] * num_tasks
        
        # 学习 log(σ²) 参数 (可学习的参数)
        # 使用 nn.Parameter 使其成为模型的可学习参数
        self.log_vars = nn.ParameterList([
            nn.Parameter(torch.tensor(init_val, dtype=torch.float32))
            for init_val in init_log_vars
        ])
        
        self.num_tasks = num_tasks
    
    def forward(self, *losses):
        """
        计算加权后的总损失
        
        Args:
            *losses: 各个任务的损失值 (可以是 tensor 或 float)
        
        Returns:
            total_loss: 加权总损失
            loss_weights: 各任务的有效权重 (用于监控)
            
        公式: L_total = Σ [ (1/2) * exp(-log_var_i) * L_i + (1/2) * log_var_i ]
        """
        assert len(losses) == self.num_tasks, \
            f"Expected {self.num_tasks} losses, got {len(losses)}"
        
        total_loss = 0.0
        weights = []
        
        for i, loss in enumerate(losses):
            if isinstance(loss, (int, float)) and loss == 0:
                # 跳过零损失 (例如某些任务在某些批次不计算)
                weights.append(0.0)
                continue
            
            log_var = self.log_vars[i]
            
            # 精度 (precision) = 1 / σ² = exp(-log_var)
            precision = torch.exp(-log_var)
            
            # 加权损失 + 正则化项
            # L_i_weighted = (1/2) * precision * L_i + (1/2) * log_var
            weighted_loss = 0.5 * precision * loss + 0.5 * log_var
            
            total_loss = total_loss + weighted_loss
            
            # 记录有效权重 (用于监控)
            weights.append(precision.item())
        
        return total_loss, weights
    
    def get_sigmas(self):
        """
        获取各任务的 σ 值 (标准差)
        
        Returns:
            sigmas: [σ₁, σ₂, ...] 列表
        """
        sigmas = []
        for log_var in self.log_vars:
            # σ = sqrt(exp(log_var)) = exp(log_var / 2)
            sigma = torch.exp(0.5 * log_var).item()
            sigmas.append(sigma)
        return sigmas
    
    def get_weights(self):
        """
        获取各任务的有效权重 (1 / 2σ²)
        
        Returns:
            weights: [w₁, w₂, ...] 列表
        """
        weights = []
        for log_var in self.log_vars:
            # w = 1 / (2σ²) = (1/2) * exp(-log_var)
            weight = 0.5 * torch.exp(-log_var).item()
            weights.append(weight)
        return weights
    
    def get_log_vars(self):
        """
        获取各任务的 log(σ²) 值
        
        Returns:
            log_vars: [log(σ₁²), log(σ₂²), ...] 列表
        """
        return [lv.item() for lv in self.log_vars]
    
    def __repr__(self):
        sigmas = self.get_sigmas()
        weights = self.get_weights()
        info = f"MultiTaskUncertaintyWeighting(num_tasks={self.num_tasks})\n"
        for i in range(self.num_tasks):
            info += f"  Task {i}: σ={sigmas[i]:.4f}, weight={weights[i]:.4f}\n"
        return info


class UncertaintyWeightedLoss(nn.Module):
    """
    便捷的损失计算模块，封装了损失函数和不确定性加权
    
    使用示例:
        loss_module = UncertaintyWeightedLoss(
            loss_fns=[nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss(ignore_index=-100)],
            task_names=['classification', 'caption']
        )
        total_loss, info = loss_module(
            preds=[cls_logits, cap_logits],
            targets=[labels, cap_labels]
        )
    """
    
    def __init__(self, loss_fns, task_names=None, init_log_vars=None):
        super().__init__()
        
        self.num_tasks = len(loss_fns)
        self.loss_fns = nn.ModuleList(loss_fns)
        self.task_names = task_names or [f"task_{i}" for i in range(self.num_tasks)]
        
        self.uncertainty = MultiTaskUncertaintyWeighting(
            num_tasks=self.num_tasks,
            init_log_vars=init_log_vars
        )
    
    def forward(self, preds, targets, masks=None):
        """
        计算加权总损失
        
        Args:
            preds: 预测结果列表 [pred_1, pred_2, ...]
            targets: 目标值列表 [target_1, target_2, ...]
            masks: 可选的掩码列表，用于指示是否计算某个任务的损失
        
        Returns:
            total_loss: 加权总损失
            info: 包含各任务损失和权重的字典
        """
        individual_losses = []
        
        for i, (pred, target, loss_fn) in enumerate(zip(preds, targets, self.loss_fns)):
            if pred is None or target is None:
                individual_losses.append(0.0)
            else:
                individual_losses.append(loss_fn(pred, target))
        
        total_loss, weights = self.uncertainty(*individual_losses)
        
        info = {
            'total_loss': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
            'individual_losses': {
                self.task_names[i]: (l.item() if isinstance(l, torch.Tensor) else l)
                for i, l in enumerate(individual_losses)
            },
            'weights': {
                self.task_names[i]: w for i, w in enumerate(weights)
            },
            'sigmas': {
                self.task_names[i]: s for i, s in enumerate(self.uncertainty.get_sigmas())
            }
        }
        
        return total_loss, info
