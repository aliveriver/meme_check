"""
将 3.0 数据格式转换为代码所需的旧格式字段名
3.0 -> 旧格式的映射:
  path -> new_path
  label (标量) -> new_label (one-hot, 2类)
  type (标量) -> new_type (one-hot, 5类)
  text_modal + image_modal -> modal (数组)
  cot_fusion_analysis -> 保留不变
  其他字段保留不变
"""
import json
import sys
import os


def convert_record(record):
    """转换单条记录"""
    new_record = {}
    
    # 保留原始字段
    new_record['text'] = record.get('text', '')
    new_record['text_discription'] = record.get('text_discription', '')
    new_record['meme_discription'] = record.get('meme_discription', '')
    
    # path -> new_path
    new_record['new_path'] = record.get('path', '')
    
    # label (标量) -> new_label (one-hot, 2类)
    label = int(record.get('label', 0))
    new_label = [0, 0]
    new_label[label] = 1
    new_record['new_label'] = new_label
    
    # type (标量) -> new_type (one-hot, 5类)
    type_val = int(record.get('type', 0))
    new_type = [0, 0, 0, 0, 0]
    new_type[type_val] = 1
    new_record['new_type'] = new_type
    
    # text_modal + image_modal -> modal
    text_modal = int(record.get('text_modal', 0))
    image_modal = int(record.get('image_modal', 0))
    new_record['modal'] = [text_modal, image_modal]
    
    # 保留 target
    if 'target' in record:
        new_record['target'] = record['target']
    
    # 保留 cot_fusion_analysis
    if 'cot_fusion_analysis' in record:
        new_record['cot_fusion_analysis'] = record['cot_fusion_analysis']
    
    return new_record


def convert_file(input_path, output_path=None):
    """转换整个 JSON 文件"""
    if output_path is None:
        # 覆盖原文件
        output_path = input_path
    
    print(f"读取: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"共 {len(data)} 条记录，开始转换...")
    converted = [convert_record(record) for record in data]
    
    print(f"写入: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 转换完成！")
    
    # 打印示例
    print(f"\n转换前示例:")
    print(json.dumps(data[0], ensure_ascii=False, indent=2)[:500])
    print(f"\n转换后示例:")
    print(json.dumps(converted[0], ensure_ascii=False, indent=2)[:500])


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    
    files_to_convert = [
        'train_data_discription_3.0.json',
        'test_data_discription_3.0.json',
    ]
    
    for filename in files_to_convert:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            convert_file(filepath)
        else:
            print(f"⚠ 文件不存在: {filepath}")
