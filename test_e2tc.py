"""
测试E2TC模块是否正确实现
"""

import torch
from config.Config_base import Config_base
from dataset.dataset import MemeDataset
from model.MHKE import MHKE_E2TC
from torch.utils.data import DataLoader

def test_dataset():
    """测试数据集是否正确返回E2TC所需的字段"""
    print("=" * 50)
    print("Testing Dataset...")
    
    config = Config_base(model_name="MHKE-E2TC", task_name="task_1")
    dataset = MemeDataset(config, training=True)
    
    # 获取一个样本
    sample = dataset[0]
    
    print(f"Sample keys: {sample.keys()}")
    print(f"input_ids shape: {sample['input_ids'].shape}")
    print(f"image_tensor shape: {sample['image_tensor'].shape}")
    print(f"cap_input_ids shape: {sample['cap_input_ids'].shape}")
    print(f"cap_labels shape: {sample['cap_labels'].shape}")
    
    # 检查cap_labels中-100的数量
    num_padding = (sample['cap_labels'] == -100).sum().item()
    print(f"Number of padding tokens (label=-100): {num_padding}")
    
    print("✓ Dataset test passed!\n")
    return True

def test_model_forward():
    """测试模型前向传播"""
    print("=" * 50)
    print("Testing Model Forward Pass...")
    
    config = Config_base(model_name="MHKE-E2TC", task_name="task_1")
    config.batch_size = 2  # 小批次测试
    
    # 创建数据加载器
    dataset = MemeDataset(config, training=True)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    
    # 获取一个batch
    batch = next(iter(dataloader))
    
    # 创建模型
    model = MHKE_E2TC(config).to(config.device)
    model.train()
    
    # 前向传播
    print("Running forward pass...")
    cls_logits, cap_logits = model(**batch)
    
    print(f"cls_logits shape: {cls_logits.shape}")  # [Batch, 2]
    print(f"cap_logits shape: {cap_logits.shape}")  # [Batch, Seq_Len, Vocab_Size]
    
    # 检查输出维度
    assert cls_logits.shape[0] == config.batch_size
    assert cls_logits.shape[1] == config.num_classes
    assert cap_logits.shape[0] == config.batch_size
    assert cap_logits.shape[1] == config.pad_size  # sequence length
    
    print("✓ Model forward pass test passed!\n")
    return True

def test_loss_computation():
    """测试Loss计算"""
    print("=" * 50)
    print("Testing Loss Computation...")
    
    config = Config_base(model_name="MHKE-E2TC", task_name="task_1")
    config.batch_size = 2
    
    # 创建数据
    dataset = MemeDataset(config, training=True)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    batch = next(iter(dataloader))
    
    # 创建模型
    model = MHKE_E2TC(config).to(config.device)
    model.train()
    
    # 前向传播
    cls_logits, cap_logits = model(**batch)
    
    # 计算分类Loss
    cls_loss_fn = torch.nn.BCEWithLogitsLoss()
    labels = batch['label'].to(config.device)
    loss_cls = cls_loss_fn(cls_logits, labels.float())
    print(f"Classification Loss: {loss_cls.item():.4f}")
    
    # 计算Caption Loss
    cap_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    cap_labels = batch['cap_labels'].to(config.device)
    vocab_size = cap_logits.size(-1)
    loss_cap = cap_loss_fn(
        cap_logits.view(-1, vocab_size),
        cap_labels.view(-1)
    )
    print(f"Caption Loss: {loss_cap.item():.4f}")
    
    # 总Loss
    total_loss = loss_cls + config.e2tc_weight * loss_cap
    print(f"Total Loss: {total_loss.item():.4f}")
    
    # 测试反向传播
    print("Testing backward pass...")
    total_loss.backward()
    
    print("✓ Loss computation test passed!\n")
    return True

def test_eval_mode():
    """测试eval模式下不计算caption loss"""
    print("=" * 50)
    print("Testing Eval Mode...")
    
    config = Config_base(model_name="MHKE-E2TC", task_name="task_1")
    config.batch_size = 2
    
    dataset = MemeDataset(config, training=False)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    batch = next(iter(dataloader))
    
    model = MHKE_E2TC(config).to(config.device)
    model.eval()
    
    # eval模式下的前向传播
    batch_eval = {k: v for k, v in batch.items()}
    batch_eval['training'] = False
    
    with torch.no_grad():
        cls_logits, cap_logits = model(**batch_eval)
    
    print(f"cls_logits shape: {cls_logits.shape}")
    print(f"cap_logits: {cap_logits}")  # 应该是None
    
    assert cap_logits is None, "cap_logits should be None in eval mode"
    
    print("✓ Eval mode test passed!\n")
    return True

def main():
    print("\n" + "=" * 50)
    print("E2TC Module Testing")
    print("=" * 50 + "\n")
    
    try:
        # 测试数据集
        test_dataset()
        
        # 测试模型前向传播
        test_model_forward()
        
        # 测试Loss计算
        test_loss_computation()
        
        # 测试eval模式
        test_eval_mode()
        
        print("=" * 50)
        print("✓ All tests passed successfully!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
