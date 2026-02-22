#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据增强脚本 V4 - 生成高质量的 meme_discription / text_discription / cot_fusion_analysis
======================================================================================
与 V3 的区别：
  - V3 只新增 cot_fusion_analysis 和 text_modal/image_modal，未改动 meme_discription 和 text_discription
  - V4 **重新生成** meme_discription 和 text_discription，大幅增加信息量（从 ~16字 → ~50-80字）
  - V4 同时改进 cot_fusion_analysis 的生成质量

输出 train_data_discription_4.0.json 和 test_data_discription_4.0.json

使用方法 (在 AutoDL 服务器上运行):
    # 1. 先启动 LLaMA-Factory API 服务（另开一个终端）
    cd /root/autodl-tmp/LLaMA-Factory
    API_PORT=8000 llamafactory-cli api chat \
        --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
        --template qwen3_vl \
        --infer_backend vllm

    # 2. 运行增强脚本 (推荐使用多模态模式)
    cd /root/MEME

    # 先用 dry-run 测试 3 条样本
    python data_augmentation_v4.py --dry-run --use-image

    # 处理训练集
    python data_augmentation_v4.py --split train --use-image

    # 处理测试集
    python data_augmentation_v4.py --split test --use-image

    # 同时处理训练集和测试集
    python data_augmentation_v4.py --split all --use-image

    # 从断点续传
    python data_augmentation_v4.py --split train --use-image --resume
"""

import json
import os
import re
import sys
import time
import argparse
import logging
import base64
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

# ============================================================
# 配置区
# ============================================================

# API 配置 (LLaMA-Factory OpenAI 兼容格式)
API_BASE = "http://localhost:8000/v1"
API_KEY = "0"
MODEL_NAME = "Qwen3-VL-8B-Instruct"

# 生成参数 - V4 需要更长输出
MAX_TOKENS = 800          # 比 V3 (512) 更大，容纳更详细的描述
TEMPERATURE = 0.5         # 比 V3 (0.7) 更低，提高一致性和准确性
TOP_P = 0.9

# 并发与重试
MAX_WORKERS = 1
MAX_RETRIES = 3
RETRY_DELAY = 3
REQUEST_DELAY = 0.2

# 路径配置
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MEME_DIR = os.path.join(DATA_DIR, "meme")

# V4: 输入使用 3.0，输出为 4.0
TRAIN_INPUT = os.path.join(DATA_DIR, "train_data_discription_3.0.json")
TEST_INPUT = os.path.join(DATA_DIR, "test_data_discription_3.0.json")
TRAIN_OUTPUT = os.path.join(DATA_DIR, "train_data_discription_4.0.json")
TEST_OUTPUT = os.path.join(DATA_DIR, "test_data_discription_4.0.json")

# 断点续传
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(PROJECT_ROOT, "data_augmentation_v4.log"),
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================
# 标签映射
# ============================================================
TYPE_MAP = {
    0: "非有害",
    1: "针对性有害 (Targeted Harmful)",
    2: "性暗示 (Sexual Innuendo)",
    3: "一般冒犯 (General Offense)",
    4: "丧文化 (Dispirited Culture)",
}

TARGET_MAP = {
    0: "无特定目标",
    1: "性别 (Gender)",
    2: "种族 (Race)",
    3: "MBTI",
    4: "地域 (Region)",
    5: "年龄 (Age)",
    6: "外貌与身体 (Appearance & Body)",
    7: "疾病与健康 (Disease & Health)",
    8: "LGBTQ",
    9: "职业 (Occupation)",
    10: "社群与组织 (Community & Organize)",
    11: "特定个人 (Specific Individual)",
    12: "贫穷 (Poor)",
    13: "婚姻与关系 (Marriage & Relationship)",
    14: "父母 (Parent)",
    15: "其他 (Others)",
}


# ============================================================
# API 调用
# ============================================================

def _try_import_openai():
    try:
        import openai
        return openai
    except ImportError:
        return None


def _try_import_requests():
    try:
        import requests
        return requests
    except ImportError:
        return None


def call_qwen_api(
    prompt: str,
    system_prompt: str = "",
    image_path: Optional[str] = None,
) -> str:
    """调用 Qwen API（OpenAI 兼容格式），支持多模态"""
    openai = _try_import_openai()
    requests = _try_import_requests()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if image_path and os.path.exists(image_path):
        content = []
        with open(image_path, "rb") as img_f:
            img_b64 = base64.b64encode(img_f.read()).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
        })
        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": prompt})

    if openai is not None:
        client = openai.OpenAI(base_url=API_BASE, api_key=API_KEY)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return response.choices[0].message.content.strip()
    elif requests is not None:
        url = f"{API_BASE}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        }
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=180)
        resp.raise_for_status()
        result = resp.json()
        return result["choices"][0]["message"]["content"].strip()
    else:
        raise ImportError("请安装 openai 或 requests 库")


# ============================================================
# V4 Prompt - 生成高质量描述
# ============================================================

SYSTEM_PROMPT = """你是一个专业的中文网络梗图内容分析专家，精通中文互联网文化和表情包。
你的任务是对梗图进行详细的视觉描述、文本语义分析和情感判定。

核心要求：
- 图像描述要具体详实，包含画面元素、人物表情、文字布局等视觉细节
- 文本分析要挖掘字面含义、隐含含义、文化指涉和修辞手法
- 情感判定理由要结合图文关系来分析
- 全部使用中文回答，严格按照指定格式输出
- 不要输出多余的解释或前言"""


def build_cot_prompt(item: dict) -> str:
    """
    V4 Prompt：要求模型生成详细的图像描述、文本分析和标签判定理由
    """
    text = item.get("text", "")
    old_meme_desc = item.get("meme_discription", "")
    old_text_desc = item.get("text_discription", "")
    label = item.get("label", item.get("new_label", -1))

    label_str = "有害 (Harmful)" if label == 1 else "非有害 (Non-harmful)"

    prompt = f"""请对以下中文梗图进行详细分析，生成高质量的描述和判定。

## 梗图信息
- **梗图上的文字**: {text}
- **原始图像描述(仅供参考)**: {old_meme_desc}
- **原始文本分析(仅供参考)**: {old_text_desc}
- **标注标签**: {label_str}

请严格按照以下格式输出，每个部分要有足够的信息量：

[图像描述]
（请详细描述图片的视觉内容：画面中有什么人物/动物/物体、什么表情/姿态、什么场景/背景、文字如何排布、整体视觉风格和色调。50-80字。如果能看到图片请基于图片描述，否则基于原始描述扩展。）

[文本语义]
（请分析梗图文字的含义：字面意思是什么、隐含什么潜台词、是否引用了网络梗/流行语/文化典故、使用了什么修辞手法（反讽、夸张、比喻等）。50-80字。）

[文本情感]
（判断文本情感倾向：负向或非负向。用1句话说明理由。）

[图像情感]
（判断图像情感倾向：负向或非负向。用1句话说明理由。）

[标签判定]
（结合图文关系，解释该梗图被判定为 {label_str} 的理由。分析图文如何协作/对比来传达含义，是否存在讽刺、冒犯、歧视等有害元素。50-100字。）"""

    return prompt


# ============================================================
# 解析模型输出 (V4)
# ============================================================

def _parse_sentiment_to_binary(text: str) -> int:
    """将情感分析文本转换为 0/1（负向=1, 非负向=0）"""
    if "负向" in text or "负面" in text:
        return 1
    return 0


def _extract_section(response: str, section_name: str, next_sections: list) -> str:
    """
    通用的 section 提取函数
    从 [section_name] 到下一个 [next_section] 之间提取内容
    """
    # 构建下一个 section 的匹配模式
    next_pattern = "|".join(re.escape(f"[{s}]") for s in next_sections) if next_sections else "$"

    pattern = rf"\[{re.escape(section_name)}\]\s*(.*?)(?:\[(?:{next_pattern.replace(chr(91), '').replace(chr(93), '')})\]|$)" if next_sections else rf"\[{re.escape(section_name)}\]\s*(.*?)$"

    # Simpler approach: find section start, then find next section start or end
    start_marker = f"[{section_name}]"
    start_idx = response.find(start_marker)
    if start_idx == -1:
        return ""

    content_start = start_idx + len(start_marker)

    # Find the next section
    end_idx = len(response)
    for ns in next_sections:
        ns_marker = f"[{ns}]"
        ns_idx = response.find(ns_marker, content_start)
        if ns_idx != -1 and ns_idx < end_idx:
            end_idx = ns_idx

    content = response[content_start:end_idx].strip()

    # 清理括号提示文字（如果模型把提示也输出了）
    content = re.sub(r'^[（(]请[^）)]*[）)]\s*', '', content)
    content = re.sub(r'^[（(]判断[^）)]*[）)]\s*', '', content)
    content = re.sub(r'^[（(]结合[^）)]*[）)]\s*', '', content)

    return content.strip()


def parse_cot_response(response: str) -> dict:
    """
    V4: 解析模型输出，提取所有字段

    提取:
    - meme_discription: 图像描述
    - text_discription: 文本语义分析
    - text_modal: 文本情感 (0/1)
    - image_modal: 图像情感 (0/1)
    - cot_fusion_analysis: 标签判定理由
    """
    sections = ["图像描述", "文本语义", "文本情感", "图像情感", "标签判定"]

    result = {
        "meme_discription": "",
        "text_discription": "",
        "text_modal": 0,
        "image_modal": 0,
        "cot_fusion_analysis": "",
    }

    # 提取图像描述 → meme_discription
    meme_desc = _extract_section(response, "图像描述", ["文本语义", "文本情感", "图像情感", "标签判定"])
    if meme_desc:
        result["meme_discription"] = meme_desc

    # 提取文本语义 → text_discription
    text_desc = _extract_section(response, "文本语义", ["文本情感", "图像情感", "标签判定"])
    if text_desc:
        result["text_discription"] = text_desc

    # 提取文本情感 → text_modal
    text_sent = _extract_section(response, "文本情感", ["图像情感", "标签判定"])
    if text_sent:
        result["text_modal"] = _parse_sentiment_to_binary(text_sent)

    # 提取图像情感 → image_modal
    img_sent = _extract_section(response, "图像情感", ["标签判定"])
    if img_sent:
        result["image_modal"] = _parse_sentiment_to_binary(img_sent)

    # 提取标签判定 → cot_fusion_analysis
    fusion = _extract_section(response, "标签判定", [])
    if fusion:
        result["cot_fusion_analysis"] = fusion

    return result


# ============================================================
# 单条样本处理
# ============================================================

def process_single_item(
    index: int,
    item: dict,
    total: int,
    use_image: bool = False,
) -> dict:
    """处理单条样本：构造 prompt -> 调用模型 -> 解析结果 -> 合并到原数据"""
    path = item.get("path", item.get("new_path", "unknown"))
    prompt = build_cot_prompt(item)

    # 图片路径
    image_path = None
    if use_image:
        image_path = os.path.join(MEME_DIR, path)
        if not os.path.exists(image_path):
            image_path = None

    # 重试调用
    response = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = call_qwen_api(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                image_path=image_path,
            )
            break
        except Exception as e:
            logger.warning(
                f"[{index+1}/{total}] {path} 第 {attempt} 次调用失败: {e}"
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                logger.error(f"[{index+1}/{total}] {path} 所有重试均失败，保留原始描述")
                response = ""

    # 解析 CoT 输出
    if response:
        cot_result = parse_cot_response(response)

        # 质量检查：如果新描述太短（少于 10 字），保留原始描述
        if len(cot_result["meme_discription"]) < 10:
            cot_result["meme_discription"] = item.get("meme_discription", "")
            logger.warning(f"[{index+1}/{total}] {path} 图像描述过短，保留原始")
        if len(cot_result["text_discription"]) < 10:
            cot_result["text_discription"] = item.get("text_discription", "")
            logger.warning(f"[{index+1}/{total}] {path} 文本分析过短，保留原始")
    else:
        # API 调用全部失败，保留原始数据
        cot_result = {
            "meme_discription": item.get("meme_discription", ""),
            "text_discription": item.get("text_discription", ""),
            "text_modal": item.get("text_modal", 0),
            "image_modal": item.get("image_modal", 0),
            "cot_fusion_analysis": item.get("cot_fusion_analysis", ""),
        }

    # 合并原数据和增强数据
    augmented_item = dict(item)
    augmented_item.update(cot_result)

    if (index + 1) % 50 == 0 or index == 0:
        md_len = len(augmented_item.get("meme_discription", ""))
        td_len = len(augmented_item.get("text_discription", ""))
        cf_len = len(augmented_item.get("cot_fusion_analysis", ""))
        logger.info(
            f"[{index+1}/{total}] ✅ {path} (meme_desc={md_len}字, "
            f"text_desc={td_len}字, cot={cf_len}字)"
        )

    return augmented_item


# ============================================================
# 断点续传
# ============================================================

def load_checkpoint(checkpoint_path: str) -> dict:
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"📥 加载断点: {len(data)} 条已完成")
        # 兼容 path 和 new_path
        return {item.get("path", item.get("new_path", "")): item for item in data}
    return {}


def save_checkpoint(results: list, checkpoint_path: str):
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


# ============================================================
# 主处理流程
# ============================================================

def process_dataset(
    input_path: str,
    output_path: str,
    split_name: str,
    resume: bool = False,
    use_image: bool = False,
    max_workers: int = MAX_WORKERS,
):
    logger.info(f"\n{'='*60}")
    logger.info(f"🚀 V4 数据增强 - 处理 {split_name} 数据集")
    logger.info(f"{'='*60}")
    logger.info(f"  输入: {input_path}")
    logger.info(f"  输出: {output_path}")
    logger.info(f"  并发数: {max_workers}")
    logger.info(f"  多模态: {'是' if use_image else '否'}")
    logger.info(f"  温度: {TEMPERATURE}")
    logger.info(f"  最大 tokens: {MAX_TOKENS}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    total = len(data)
    logger.info(f"  样本总数: {total}")

    # 断点续传
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{split_name}_v4_checkpoint.json")
    completed = {}
    if resume:
        completed = load_checkpoint(checkpoint_path)
        logger.info(f"  已完成: {len(completed)}, 待处理: {total - len(completed)}")

    # 过滤待处理样本
    pending_items = []
    for i, item in enumerate(data):
        key = item.get("path", item.get("new_path", ""))
        if resume and key in completed:
            continue
        pending_items.append((i, item))

    if not pending_items:
        logger.info("  所有样本已处理完毕！")
        results = list(completed.values())
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        return

    results = list(completed.values()) if resume else []
    start_time = time.time()

    if max_workers <= 1:
        # 单线程处理
        for i, item in pending_items:
            result = process_single_item(i, item, total, use_image)
            results.append(result)

            current_count = len(results)
            if current_count % 50 == 0 and current_count > 0:
                logger.info(f"💾 正在保存断点 ({current_count}/{total})...")
                save_checkpoint(results, checkpoint_path)
                elapsed = time.time() - start_time
                speed = current_count / elapsed if elapsed > 0 else 0
                eta = (total - current_count) / speed if speed > 0 else 0
                logger.info(
                    f"  速度: {speed:.1f} 条/秒, 预计剩余: {eta/60:.1f} 分钟"
                )

            time.sleep(REQUEST_DELAY)
    else:
        # 多线程并发处理
        logger.info(f"  🚀 使用 {max_workers} 线程并发处理")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(process_single_item, i, item, total, use_image): i
                for i, item in pending_items
            }
            for future in as_completed(future_to_idx):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    idx = future_to_idx[future]
                    logger.error(f"[{idx+1}/{total}] 处理异常: {e}")
                    # 保留原始数据
                    results.append(dict(pending_items[idx][1]))

                current_count = len(results)
                if current_count % 50 == 0 and current_count > 0:
                    logger.info(f"💾 正在保存断点 ({current_count}/{total})...")
                    save_checkpoint(results, checkpoint_path)
                    elapsed = time.time() - start_time
                    speed = current_count / elapsed if elapsed > 0 else 0
                    eta = (total - current_count) / speed if speed > 0 else 0
                    logger.info(
                        f"  速度: {speed:.1f} 条/秒, 预计剩余: {eta/60:.1f} 分钟"
                    )

    # 按原始顺序排序
    path_key = "path" if "path" in data[0] else "new_path"
    path_to_result = {r.get(path_key, ""): r for r in results}
    ordered_results = []
    for item in data:
        key = item.get(path_key, "")
        if key in path_to_result:
            ordered_results.append(path_to_result[key])
        else:
            ordered_results.append(item)

    # 保存最终结果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ordered_results, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ {split_name} V4 增强完成！")
    logger.info(f"  总耗时: {elapsed/60:.1f} 分钟")
    logger.info(f"  输出: {output_path}")
    logger.info(f"  样本数: {len(ordered_results)}")

    # 统计描述长度
    md_lens = [len(r.get("meme_discription", "")) for r in ordered_results]
    td_lens = [len(r.get("text_discription", "")) for r in ordered_results]
    cf_lens = [len(r.get("cot_fusion_analysis", "")) for r in ordered_results]
    logger.info(f"  meme_discription 字数: 平均={sum(md_lens)/len(md_lens):.0f}, "
                f"最短={min(md_lens)}, 最长={max(md_lens)}")
    logger.info(f"  text_discription 字数: 平均={sum(td_lens)/len(td_lens):.0f}, "
                f"最短={min(td_lens)}, 最长={max(td_lens)}")
    logger.info(f"  cot_fusion_analysis 字数: 平均={sum(cf_lens)/len(cf_lens):.0f}, "
                f"最短={min(cf_lens)}, 最长={max(cf_lens)}")
    logger.info(f"{'='*60}")

    # 清理断点
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        logger.info(f"  🗑️ 已清理断点文件")


# ============================================================
# CLI 入口
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="ToxiCN_MM V4 数据增强 - 生成高质量描述"
    )
    parser.add_argument(
        "--split", type=str, default="all",
        choices=["train", "test", "all"],
        help="处理哪个数据集 (default: all)",
    )
    parser.add_argument(
        "--api-base", type=str, default=API_BASE,
        help=f"API 地址 (default: {API_BASE})",
    )
    parser.add_argument(
        "--api-key", type=str, default=API_KEY,
        help="API Key",
    )
    parser.add_argument(
        "--model", type=str, default=MODEL_NAME,
        help=f"模型名称 (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="从断点续传",
    )
    parser.add_argument(
        "--use-image", action="store_true",
        help="启用多模态模式（推荐：将图片发送给模型以获得更准确的描述）",
    )
    parser.add_argument(
        "--temperature", type=float, default=TEMPERATURE,
        help=f"生成温度 (default: {TEMPERATURE})",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=MAX_TOKENS,
        help=f"最大 token 数 (default: {MAX_TOKENS})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="试运行：仅处理前 3 条样本",
    )
    parser.add_argument(
        "--workers", type=int, default=MAX_WORKERS,
        help=f"并发线程数 (default: {MAX_WORKERS}，建议 2-4)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    global API_BASE, API_KEY, MODEL_NAME, TEMPERATURE, MAX_TOKENS, MAX_WORKERS
    API_BASE = args.api_base
    API_KEY = args.api_key
    MODEL_NAME = args.model
    TEMPERATURE = args.temperature
    MAX_TOKENS = args.max_tokens
    MAX_WORKERS = args.workers

    logger.info("=" * 60)
    logger.info("🤖 ToxiCN_MM V4 数据增强 - 高质量描述生成")
    logger.info("=" * 60)
    logger.info(f"  API: {API_BASE}")
    logger.info(f"  模型: {MODEL_NAME}")
    logger.info(f"  温度: {TEMPERATURE}")
    logger.info(f"  最大 tokens: {MAX_TOKENS}")
    logger.info(f"  并发数: {MAX_WORKERS}")
    logger.info(f"  多模态: {'是' if args.use_image else '否'}")
    logger.info(f"  断点续传: {'是' if args.resume else '否'}")
    logger.info("")
    logger.info("  ⚡ V4 改进: 重新生成 meme_discription 和 text_discription")
    logger.info("     目标: 从 ~16字 提升到 ~50-80字，大幅增加信息量")

    # Dry-run
    if args.dry_run:
        logger.info("\n⚡ 试运行模式：仅处理前 3 条样本")
        with open(TRAIN_INPUT, "r", encoding="utf-8") as f:
            data = json.load(f)[:3]

        for i, item in enumerate(data):
            logger.info(f"\n{'─'*50}")
            logger.info(f"样本 {i+1}")
            path = item.get("path", item.get("new_path", "unknown"))
            logger.info(f"  Path: {path}")
            logger.info(f"  Text: {item.get('text', '')[:60]}...")
            logger.info(f"  旧 meme_desc ({len(item.get('meme_discription',''))}字): "
                        f"{item.get('meme_discription', '')}")
            logger.info(f"  旧 text_desc ({len(item.get('text_discription',''))}字): "
                        f"{item.get('text_discription', '')}")

            result = process_single_item(i, item, 3, args.use_image)

            logger.info(f"  新 meme_desc ({len(result['meme_discription'])}字): "
                        f"{result['meme_discription']}")
            logger.info(f"  新 text_desc ({len(result['text_discription'])}字): "
                        f"{result['text_discription']}")
            logger.info(f"  cot_fusion  ({len(result.get('cot_fusion_analysis',''))}字): "
                        f"{result.get('cot_fusion_analysis', '')}")
            logger.info(f"  text_modal={result['text_modal']}, "
                        f"image_modal={result['image_modal']}")

        dry_run_output = os.path.join(DATA_DIR, "dry_run_v4_preview.json")
        with open(dry_run_output, "w", encoding="utf-8") as f:
            json.dump([process_single_item(i, item, 3, args.use_image)
                      for i, item in enumerate(data)], f, ensure_ascii=False, indent=2)
        logger.info(f"\n📄 试运行结果: {dry_run_output}")
        logger.info("\n✅ 试运行完成！检查结果后请正式运行。")
        return

    # 正式处理
    if args.split in ("train", "all"):
        process_dataset(
            TRAIN_INPUT, TRAIN_OUTPUT, "train",
            resume=args.resume, use_image=args.use_image,
            max_workers=MAX_WORKERS,
        )

    if args.split in ("test", "all"):
        process_dataset(
            TEST_INPUT, TEST_OUTPUT, "test",
            resume=args.resume, use_image=args.use_image,
            max_workers=MAX_WORKERS,
        )

    logger.info("\n🎉 V4 数据增强全部完成！")
    logger.info("下一步: 更新 Config_base.py 中的数据路径为 4.0，然后重新训练模型")


if __name__ == "__main__":
    main()
