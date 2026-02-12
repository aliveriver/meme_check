#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据增强脚本 - 调用 Qwen3-VL-8B-Instruct 大模型，以思维链 (CoT) 形式生成增强版数据描述
输出 train_data_discription_3.0.json 和 test_data_discription_3.0.json

使用方法 (在 AutoDL 服务器上运行):
    # 1. 先启动 LLaMA-Factory API 服务（另开一个终端）
    cd /root/autodl-tmp/LLaMA-Factory
    API_PORT=8000 llamafactory-cli api chat \
        --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
        --template qwen3_vl \
        --infer_backend vllm

    # 2. 运行增强脚本
    cd /root/MEME

    # 先用 dry-run 测试 3 条样本，确认 API 正常
    python data_augmentation.py --dry-run --use-image

    # 处理训练集（默认开启多模态）
    python data_augmentation.py --split train --use-image

    # 处理测试集
    python data_augmentation.py --split test --use-image

    # 同时处理训练集和测试集
    python data_augmentation.py --split all --use-image

    # 从断点续传（中断后继续）
    python data_augmentation.py --split train --use-image --resume

    # 自定义 API 端口
    python data_augmentation.py --split all --use-image --api-base http://localhost:8000/v1

环境说明:
    - 服务器: AutoDL
    - 模型: Qwen3-VL-8B-Instruct (视觉语言模型)
    - 框架: LLaMA-Factory (位于 /root/autodl-tmp/LLaMA-Factory)
    - 项目: /root/MEME
    - API: OpenAI 兼容格式 (LLaMA-Factory 默认端口 8000)
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
# 配置区 - AutoDL 服务器环境
# ============================================================

# API 配置 (LLaMA-Factory OpenAI 兼容格式)
API_BASE = "http://localhost:8000/v1"      # LLaMA-Factory API 默认端口
API_KEY = "0"                               # LLaMA-Factory 默认 key
MODEL_NAME = "Qwen3-VL-8B-Instruct"        # 模型名称

# 生成参数
MAX_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9

# 并发与重试
MAX_WORKERS = 1           # 单卡建议设为 1，避免 OOM
MAX_RETRIES = 3           # 单条失败最大重试次数
RETRY_DELAY = 3           # 重试间隔（秒）
REQUEST_DELAY = 0.2       # 请求间隔（秒），VL 模型推理较慢，适当放宽

# 路径配置 (AutoDL 服务器: /root/MEME)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MEME_DIR = os.path.join(DATA_DIR, "meme")

TRAIN_INPUT = os.path.join(DATA_DIR, "train_data_discription_2.0.json")
TEST_INPUT = os.path.join(DATA_DIR, "test_data_discription_2.0.json")
TRAIN_OUTPUT = os.path.join(DATA_DIR, "train_data_discription_3.0.json")
TEST_OUTPUT = os.path.join(DATA_DIR, "test_data_discription_3.0.json")

# 断点续传临时文件
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
            os.path.join(PROJECT_ROOT, "data_augmentation.log"),
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================
# 标签映射（用于构造 prompt）
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
# API 调用（OpenAI 兼容格式）
# ============================================================

def _try_import_openai():
    """尝试导入 openai 库，失败则使用 requests 回退"""
    try:
        import openai
        return openai
    except ImportError:
        return None


def _try_import_requests():
    """尝试导入 requests 库"""
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
    """
    调用 Qwen 大模型 API（OpenAI 兼容格式）

    优先使用 openai SDK，若未安装则回退使用 requests。
    支持纯文本和图文多模态两种模式。

    Args:
        prompt: 用户提示词
        system_prompt: 系统提示词
        image_path: 可选，图片路径（用于多模态模型）

    Returns:
        模型生成的文本
    """
    openai = _try_import_openai()
    requests = _try_import_requests()

    # 构建消息
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # 判断是否使用多模态（图文）
    if image_path and os.path.exists(image_path):
        # 多模态消息格式
        content = []
        # 读取图片并编码为 base64
        with open(image_path, "rb") as img_f:
            img_b64 = base64.b64encode(img_f.read()).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
        })
        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})
    else:
        # 纯文本消息
        messages.append({"role": "user", "content": prompt})

    # --- 方式1: 使用 openai SDK ---
    if openai is not None:
        client = openai.OpenAI(
            base_url=API_BASE,
            api_key=API_KEY,
        )
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return response.choices[0].message.content.strip()

    # --- 方式2: 使用 requests 回退 ---
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
        resp = requests.post(url, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        result = resp.json()
        return result["choices"][0]["message"]["content"].strip()

    else:
        raise ImportError(
            "请安装 openai 或 requests 库:\n"
            "  pip install openai\n"
            "  或\n"
            "  pip install requests"
        )


# ============================================================
# CoT Prompt 构造
# ============================================================

SYSTEM_PROMPT = """你是一个专业的中文网络梗图内容分析专家，熟悉中文互联网各种文化与梗图。
你的任务是对给定的梗图进行简洁的情感分析和标签判定理由分析。

要求：
- 每个步骤请控制在2-3句话以内
- 情感分析用1句话给出结论（负向或非负向）和理由
- 用中文回答，严格按照指定格式输出"""


def build_cot_prompt(item: dict) -> str:
    """
    为单条样本构建 CoT 推理 prompt

    Args:
        item: 数据集中的一条样本（dict）

    Returns:
        构造好的 prompt 文本
    """
    text = item.get("text", "")
    meme_desc = item.get("meme_discription", "")
    text_desc = item.get("text_discription", "")
    label = item.get("label", -1)

    label_str = "有害 (Harmful)" if label == 1 else "非有害 (Non-harmful)"

    prompt = f"""请对以下中文梗图进行简洁分析。

## 梗图信息
- **梗图文本**: {text}
- **图像描述**: {meme_desc}
- **文本语义**: {text_desc}

## 参考标签
- 该梗图被标注为: {label_str}

请严格按照以下格式输出：

# 第一步：文本内容分析
（2-3句概括文字的字面含义和隐含含义）

# 第二步：图像内容分析
（2-3句概括图像内容和表达效果）

# 第三步：文本情感分析
（1句话判断文本情感是否负向：负向或非负向，并简要说明理由）

# 第四步：图像情感分析
（1句话判断图像情感是否负向：负向或非负向，并简要说明理由）

# 第五步：标签判定理由
（2-3句解释为什么该梗图被判定为{label_str}）
"""

    return prompt


# ============================================================
# 解析模型输出
# ============================================================

def _parse_sentiment_to_binary(text: str) -> int:
    """将情感分析文本转换为 0/1（负向=1, 非负向=0）"""
    if "负向" in text or "负面" in text:
        return 1
    return 0


def parse_cot_response(response: str) -> dict:
    """
    解析模型输出，提取：
    - text_modal: 文本情感是否负向 (0/1)
    - image_modal: 图像情感是否负向 (0/1)
    - cot_fusion_analysis: 标签判定理由

    Args:
        response: 模型的原始输出文本

    Returns:
        包含 text_modal, image_modal, cot_fusion_analysis 的字典
    """
    result = {
        "text_modal": 0,
        "image_modal": 0,
        "cot_fusion_analysis": "",
    }

    # 提取文本情感 → text_modal
    m = re.search(
        r"(?:#+\s*第三步[:：]?\s*文本情感分析)(.*?)(?=#+\s*第四步|$)",
        response, re.DOTALL | re.IGNORECASE,
    )
    if m:
        result["text_modal"] = _parse_sentiment_to_binary(m.group(1))

    # 提取图像情感 → image_modal
    m = re.search(
        r"(?:#+\s*第四步[:：]?\s*图像情感分析)(.*?)(?=#+\s*第五步|$)",
        response, re.DOTALL | re.IGNORECASE,
    )
    if m:
        result["image_modal"] = _parse_sentiment_to_binary(m.group(1))

    # 提取标签判定理由 → cot_fusion_analysis
    m = re.search(
        r"(?:#+\s*第五步[:：]?\s*标签判定理由)(.*$)",
        response, re.DOTALL | re.IGNORECASE,
    )
    if m:
        result["cot_fusion_analysis"] = m.group(1).strip()

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
    """
    处理单条样本：构造 prompt -> 调用模型 -> 解析结果 -> 合并到原数据

    Args:
        index: 样本索引
        item: 原始样本数据
        total: 总样本数
        use_image: 是否传入图片（多模态模式）

    Returns:
        增强后的样本数据（原字段 + CoT 字段）
    """
    path = item.get("path", "unknown")
    prompt = build_cot_prompt(item)

    # 图片路径（可选）
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
                logger.error(f"[{index+1}/{total}] {path} 所有重试均失败，跳过")
                response = ""

    # 解析 CoT 输出
    if response:
        cot_result = parse_cot_response(response)
    else:
        cot_result = {
            "text_modal": 0,
            "image_modal": 0,
            "cot_fusion_analysis": "",
        }

    # 合并原数据和 CoT 增强数据（text_modal / image_modal 会被覆盖）
    augmented_item = dict(item)
    augmented_item.update(cot_result)

    if (index + 1) % 50 == 0 or index == 0:
        logger.info(f"[{index+1}/{total}] ✅ {path} 处理完成")

    return augmented_item


# ============================================================
# 断点续传
# ============================================================

def load_checkpoint(checkpoint_path: str) -> dict:
    """加载断点数据"""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"📥 加载断点: {len(data)} 条已完成")
        return {item["path"]: item for item in data}
    return {}


def save_checkpoint(results: list, checkpoint_path: str):
    """保存断点数据"""
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
    """
    处理整个数据集

    Args:
        input_path: 输入 JSON 路径
        output_path: 输出 JSON 路径
        split_name: 数据集名称(train/test)
        resume: 是否启用断点续传
        use_image: 是否使用多模态（传入图片）
        max_workers: 并发数
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"🚀 开始处理 {split_name} 数据集")
    logger.info(f"{'='*60}")
    logger.info(f"  输入: {input_path}")
    logger.info(f"  输出: {output_path}")
    logger.info(f"  并发数: {max_workers}")
    logger.info(f"  多模态: {'是' if use_image else '否（仅文本）'}")

    # 加载原始数据
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    total = len(data)
    logger.info(f"  样本总数: {total}")

    # 断点续传
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{split_name}_checkpoint.json")
    completed = {}
    if resume:
        completed = load_checkpoint(checkpoint_path)
        logger.info(f"  已完成: {len(completed)}, 待处理: {total - len(completed)}")

    # 过滤出待处理的样本
    pending_items = []
    for i, item in enumerate(data):
        if resume and item.get("path", "") in completed:
            continue
        pending_items.append((i, item))

    if not pending_items:
        logger.info("  所有样本已处理完毕！")
        # 直接从 checkpoint 输出
        results = list(completed.values())
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        return

    # 处理样本
    results = list(completed.values()) if resume else []
    start_time = time.time()

    if max_workers <= 1:
        # 单线程模式
        for i, item in pending_items:
            result = process_single_item(i, item, total, use_image)
            results.append(result)

            # 定期保存断点（每 50 条保存一次，减少 I/O 同时保证安全）
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
        # 多线程模式
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for i, item in pending_items:
                future = executor.submit(
                    process_single_item, i, item, total, use_image
                )
                futures[future] = (i, item)
                time.sleep(REQUEST_DELAY)

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    i, item = futures[future]
                    logger.error(f"[{i+1}/{total}] 处理失败: {e}")
                    # 保留原始数据，标记失败
                    failed_item = dict(item)
                    failed_item["cot_full"] = f"ERROR: {str(e)}"
                    results.append(failed_item)

                # 定期保存断点（每 50 条）
                current_count = len(results)
                if current_count % 50 == 0 and current_count > 0:
                    logger.info(f"💾 正在保存断点 ({current_count}/{total})...")
                    save_checkpoint(results, checkpoint_path)

    # 按原始顺序排序（通过 path 匹配）
    path_to_result = {r["path"]: r for r in results}
    ordered_results = []
    for item in data:
        path = item.get("path", "")
        if path in path_to_result:
            ordered_results.append(path_to_result[path])
        else:
            ordered_results.append(item)  # 未处理的保留原数据

    # 保存最终结果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ordered_results, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ {split_name} 处理完成！")
    logger.info(f"  总耗时: {elapsed/60:.1f} 分钟")
    logger.info(f"  输出文件: {output_path}")
    logger.info(f"  样本数: {len(ordered_results)}")
    logger.info(f"{'='*60}")

    # 清理断点文件
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        logger.info(f"  🗑️ 已清理断点文件")


# ============================================================
# CLI 入口
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="ToxiCN_MM 数据增强脚本 - 使用 Qwen 大模型生成 CoT 分析"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "test", "all"],
        help="处理哪个数据集 (default: all)",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=API_BASE,
        help=f"模型 API 地址 (default: {API_BASE})",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=API_KEY,
        help="API Key (default: EMPTY)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help=f"模型名称 (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help=f"并发请求数 (default: {MAX_WORKERS})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从断点续传（跳过已处理的样本）",
    )
    parser.add_argument(
        "--use-image",
        action="store_true",
        help="启用多模态模式（将图片发送给模型）",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        help=f"生成温度 (default: {TEMPERATURE})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_TOKENS,
        help=f"最大生成 token 数 (default: {MAX_TOKENS})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="试运行模式：仅处理前 3 条样本，用于测试",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 更新全局配置
    global API_BASE, API_KEY, MODEL_NAME, MAX_WORKERS, TEMPERATURE, MAX_TOKENS
    API_BASE = args.api_base
    API_KEY = args.api_key
    MODEL_NAME = args.model
    MAX_WORKERS = args.workers
    TEMPERATURE = args.temperature
    MAX_TOKENS = args.max_tokens

    logger.info("=" * 60)
    logger.info("🤖 ToxiCN_MM 数据增强工具 (CoT with Qwen)")
    logger.info("=" * 60)
    logger.info(f"  API 地址: {API_BASE}")
    logger.info(f"  模型名称: {MODEL_NAME}")
    logger.info(f"  并发数: {MAX_WORKERS}")
    logger.info(f"  温度: {TEMPERATURE}")
    logger.info(f"  最大 tokens: {MAX_TOKENS}")
    logger.info(f"  断点续传: {'是' if args.resume else '否'}")
    logger.info(f"  多模态: {'是' if args.use_image else '否'}")

    # Dry-run 模式：仅处理少量样本测试
    if args.dry_run:
        logger.info("\n⚡ 试运行模式：仅处理前 3 条样本")
        with open(TRAIN_INPUT, "r", encoding="utf-8") as f:
            data = json.load(f)[:3]

        dry_run_results = []
        for i, item in enumerate(data):
            logger.info(f"\n--- 样本 {i+1} ---")
            logger.info(f"  Path: {item['path']}")
            logger.info(f"  Text: {item['text'][:50]}...")
            result = process_single_item(i, item, 3, args.use_image)
            dry_run_results.append(result)
            logger.info(f"  text_modal={result['text_modal']}, image_modal={result['image_modal']}")
            logger.info(f"  判定理由: {result.get('cot_fusion_analysis', 'N/A')[:100]}...")

        # 保存临时 JSON 供查看
        dry_run_output = os.path.join(DATA_DIR, "dry_run_preview.json")
        with open(dry_run_output, "w", encoding="utf-8") as f:
            json.dump(dry_run_results, f, ensure_ascii=False, indent=2)
        logger.info(f"\n📄 试运行结果已保存至: {dry_run_output}")

        logger.info("\n✅ 试运行完成！")
        return

    # 正式处理
    if args.split in ("train", "all"):
        process_dataset(
            TRAIN_INPUT, TRAIN_OUTPUT, "train",
            resume=args.resume,
            use_image=args.use_image,
            max_workers=args.workers,
        )

    if args.split in ("test", "all"):
        process_dataset(
            TEST_INPUT, TEST_OUTPUT, "test",
            resume=args.resume,
            use_image=args.use_image,
            max_workers=args.workers,
        )

    logger.info("\n🎉 全部处理完成！")


if __name__ == "__main__":
    main()
