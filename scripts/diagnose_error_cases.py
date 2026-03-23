from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="ignore")

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.Config_base import Config_base
from dataset.dataset import MemeDataset
from model.MHKE import MHKE, MHKE_CLIP, MHKE_CrossAttention, MHKE_CrossAttention_V2, MHKE_ISSUES
from model.clip import CLIPMemesClassifier
from model.vit_roberta import (
    BertClassifier,
    ResNetClassifier,
    RobertaClassifier,
    VitClassifier,
    VitRobertaMemesClassifier,
)


TASK1_LABELS = ["non_harmful", "harmful"]
TASK2_LABELS = [
    "non_harmful",
    "targeted_harmful",
    "sexual_innuendo",
    "general_offense",
    "dispirited_culture",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose bad samples with dataset sanity checks and checkpoint-based error analysis."
    )
    parser.add_argument("--task", choices=["task_1", "task_2"], default="task_1")
    parser.add_argument(
        "--model",
        default="clip",
        choices=[
            "clip",
            "CLIP",
            "vit-roberta",
            "vit",
            "resnet",
            "roberta",
            "bert",
            "MHKE",
            "MHKE_CrossAttention",
            "MHKE_CrossAttention_V2",
            "MHKE_ISSUES",
        ],
        help="Model family used to load the checkpoint.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="test",
        help="Dataset split to inspect.",
    )
    parser.add_argument(
        "--data-file",
        default="",
        help="Optional custom JSON file. If set, it overrides --split.",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Optional checkpoint path. If omitted, the newest matching BEST checkpoint is used.",
    )
    parser.add_argument(
        "--output-dir",
        default="result/error_diagnosis",
        help="Directory used to save CSV and JSON reports.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Optional inference batch size override.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=30,
        help="Number of hard error cases to keep in the summary JSON.",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Only run dataset sanity checks, without loading a model checkpoint.",
    )
    return parser.parse_args()


def one_hot_index(value: Any, expected_len: int) -> Optional[int]:
    if not isinstance(value, list) or len(value) != expected_len:
        return None
    if sum(1 for item in value if item == 1) != 1:
        return None
    return value.index(1)


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def label_name(task: str, index: Optional[int]) -> str:
    if index is None:
        return "invalid"
    names = TASK1_LABELS if task == "task_1" else TASK2_LABELS
    if 0 <= index < len(names):
        return names[index]
    return f"unknown_{index}"


def find_checkpoint(checkpoint_dir: Path, model_name: str, task_name: str) -> Optional[Path]:
    patterns = [
        f"*{model_name}*{task_name}*BEST.tar",
        f"*{task_name}*BEST.tar",
        "*BEST.tar",
    ]
    candidates: List[Path] = []
    for pattern in patterns:
        candidates.extend(checkpoint_dir.glob(pattern))
        if candidates:
            break
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def build_config(args: argparse.Namespace) -> Tuple[Config_base, Path]:
    config = Config_base(args.model, args.task)
    if args.batch_size > 0:
        config.batch_size = args.batch_size

    if args.data_file:
        dataset_path = Path(args.data_file)
    elif args.split == "train":
        dataset_path = Path(config.train_path)
    else:
        dataset_path = Path(config.test_path)

    config.test_path = str(dataset_path)
    return config, dataset_path


def load_records(dataset_path: Path) -> List[Dict[str, Any]]:
    return json.loads(dataset_path.read_text(encoding="utf-8"))


def get_modal_tuple(record: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    modal = record.get("modal")
    if isinstance(modal, list) and len(modal) == 2:
        return modal[0], modal[1]
    return record.get("text_modal"), record.get("image_modal")


def sanity_check_records(records: List[Dict[str, Any]], meme_dir: Path, task_name: str) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    issue_counter: Counter[str] = Counter()
    sanity_rows: List[Dict[str, Any]] = []

    for index, record in enumerate(records):
        issues: List[str] = []
        binary_idx = one_hot_index(record.get("new_label"), 2)
        type_idx = one_hot_index(record.get("new_type"), 5)
        modal_tuple = get_modal_tuple(record)

        new_path = safe_text(record.get("new_path") or record.get("path"))
        image_exists = bool(new_path) and (meme_dir / new_path).exists()
        if not image_exists:
            issues.append("missing_image")

        if binary_idx is None:
            issues.append("invalid_new_label")
        if type_idx is None:
            issues.append("invalid_new_type")

        if binary_idx is not None and type_idx is not None:
            label_type_mismatch = (binary_idx == 0 and type_idx != 0) or (binary_idx == 1 and type_idx == 0)
            if label_type_mismatch:
                issues.append("label_type_mismatch")

        text_modal = record.get("text_modal")
        image_modal = record.get("image_modal")
        if (
            isinstance(record.get("modal"), list)
            and len(record["modal"]) == 2
            and text_modal is not None
            and image_modal is not None
            and tuple(record["modal"]) != (text_modal, image_modal)
        ):
            issues.append("modal_mismatch")

        text = safe_text(record.get("text")).strip()
        text_description = safe_text(record.get("text_discription")).strip()
        meme_description = safe_text(record.get("meme_discription")).strip()
        cot = safe_text(record.get("cot_fusion_analysis")).strip()

        if not text:
            issues.append("empty_text")
        if not text_description:
            issues.append("empty_text_discription")
        if not meme_description:
            issues.append("empty_meme_discription")
        if not cot:
            issues.append("empty_cot_fusion_analysis")

        for issue in issues:
            issue_counter[issue] += 1

        sanity_rows.append(
            {
                "index": index,
                "task": task_name,
                "image_path": new_path,
                "image_exists": image_exists,
                "true_label_idx": binary_idx,
                "true_label_name": label_name("task_1", binary_idx),
                "true_type_idx": type_idx,
                "true_type_name": label_name("task_2", type_idx),
                "modal": list(modal_tuple),
                "text_modal": text_modal,
                "image_modal": image_modal,
                "target": record.get("target"),
                "text_len": len(text),
                "text_discription_len": len(text_description),
                "meme_discription_len": len(meme_description),
                "cot_fusion_analysis_len": len(cot),
                "issue_count": len(issues),
                "issues": "|".join(issues),
                "text": text,
            }
        )

    return sanity_rows, dict(issue_counter)


def build_model(config: Config_base) -> torch.nn.Module:
    model_name = config.model_name
    if model_name == "clip":
        return MHKE_CLIP(config).to(config.device)
    if model_name == "CLIP":
        return CLIPMemesClassifier(config).to(config.device)
    if model_name == "vit-roberta":
        return VitRobertaMemesClassifier(config).to(config.device)
    if model_name == "vit":
        return VitClassifier(config).to(config.device)
    if model_name == "resnet":
        return ResNetClassifier(config).to(config.device)
    if model_name == "roberta":
        return RobertaClassifier(config).to(config.device)
    if model_name == "bert":
        return BertClassifier(config).to(config.device)
    if model_name == "MHKE":
        return MHKE(config).to(config.device)
    if model_name == "MHKE_CrossAttention":
        return MHKE_CrossAttention(config).to(config.device)
    if model_name == "MHKE_CrossAttention_V2":
        return MHKE_CrossAttention_V2(config).to(config.device)
    if model_name == "MHKE_ISSUES":
        return MHKE_ISSUES(config).to(config.device)
    raise ValueError(f"Unsupported model name: {model_name}")


def compute_margin(scores: torch.Tensor) -> torch.Tensor:
    sorted_scores, _ = torch.sort(scores, dim=1, descending=True)
    if sorted_scores.size(1) < 2:
        return sorted_scores[:, 0]
    return sorted_scores[:, 0] - sorted_scores[:, 1]


def run_inference(
    config: Config_base,
    records: List[Dict[str, Any]],
    sanity_rows: List[Dict[str, Any]],
    checkpoint_path: Path,
    task_name: str,
) -> List[Dict[str, Any]]:
    dataset = MemeDataset(config, training=False)
    data_loader = DataLoader(dataset, batch_size=int(config.batch_size), shuffle=False)

    model = build_model(config)
    checkpoint = torch.load(str(checkpoint_path), map_location=config.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prediction_rows: List[Dict[str, Any]] = []
    sanity_by_index = {row["index"]: row for row in sanity_rows}
    offset = 0

    for batch in tqdm(data_loader, desc="Diagnosing", colour="CYAN"):
        with torch.no_grad():
            logits = model(**batch).cpu()

        if task_name == "task_1":
            scores = torch.softmax(logits, dim=1)
            true_indices = torch.argmax(batch["label"], dim=1).cpu()
        else:
            scores = torch.sigmoid(logits)
            true_indices = torch.argmax(batch["type_label"], dim=1).cpu()

        pred_indices = torch.argmax(scores, dim=1).cpu()
        confidences = scores.gather(1, pred_indices.unsqueeze(1)).squeeze(1)
        margins = compute_margin(scores)

        batch_size = pred_indices.size(0)
        for local_idx in range(batch_size):
            record = records[offset + local_idx]
            sanity_row = sanity_by_index[offset + local_idx]
            pred_idx = int(pred_indices[local_idx].item())
            true_idx = int(true_indices[local_idx].item())
            pred_confidence = float(confidences[local_idx].item())
            margin = float(margins[local_idx].item())
            modal_tuple = get_modal_tuple(record)

            prediction_rows.append(
                {
                    "index": offset + local_idx,
                    "image_path": safe_text(record.get("new_path") or record.get("path")),
                    "true_idx": true_idx,
                    "true_name": label_name(task_name, true_idx),
                    "pred_idx": pred_idx,
                    "pred_name": label_name(task_name, pred_idx),
                    "correct": int(pred_idx == true_idx),
                    "error_type": get_error_type(task_name, true_idx, pred_idx),
                    "pred_confidence": round(pred_confidence, 6),
                    "score_margin": round(margin, 6),
                    "target": record.get("target"),
                    "modal": list(modal_tuple),
                    "text_modal": record.get("text_modal"),
                    "image_modal": record.get("image_modal"),
                    "text_len": len(safe_text(record.get("text")).strip()),
                    "text_discription_len": len(safe_text(record.get("text_discription")).strip()),
                    "meme_discription_len": len(safe_text(record.get("meme_discription")).strip()),
                    "cot_fusion_analysis_len": len(safe_text(record.get("cot_fusion_analysis")).strip()),
                    "sanity_issue_count": sanity_row["issue_count"],
                    "sanity_issues": sanity_row["issues"],
                    "text": safe_text(record.get("text")).strip(),
                    "text_discription": safe_text(record.get("text_discription")).strip(),
                    "meme_discription": safe_text(record.get("meme_discription")).strip(),
                    "cot_fusion_analysis": safe_text(record.get("cot_fusion_analysis")).strip(),
                }
            )

        offset += batch_size

    return prediction_rows


def get_error_type(task_name: str, true_idx: int, pred_idx: int) -> str:
    if pred_idx == true_idx:
        return "correct"
    if task_name == "task_1":
        if true_idx == 0 and pred_idx == 1:
            return "false_positive"
        if true_idx == 1 and pred_idx == 0:
            return "false_negative"
    return f"confused_{true_idx}_to_{pred_idx}"


def counter_from_rows(rows: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        value = row.get(key)
        if isinstance(value, list):
            counter[str(tuple(value))] += 1
        else:
            counter[str(value)] += 1
    return dict(counter)


def build_summary(
    task_name: str,
    dataset_path: Path,
    checkpoint_path: Optional[Path],
    sanity_rows: List[Dict[str, Any]],
    sanity_issue_counts: Dict[str, int],
    prediction_rows: Optional[List[Dict[str, Any]]],
    topk: int,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "task": task_name,
        "dataset_path": str(dataset_path),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else "",
        "total_samples": len(sanity_rows),
        "sanity_issue_counts": sanity_issue_counts,
        "samples_with_any_sanity_issue": sum(1 for row in sanity_rows if row["issue_count"] > 0),
    }

    if prediction_rows is None:
        return summary

    error_rows = [row for row in prediction_rows if row["correct"] == 0]
    summary.update(
        {
            "total_errors": len(error_rows),
            "error_rate": round(len(error_rows) / max(len(prediction_rows), 1), 6),
            "errors_with_sanity_issue": sum(1 for row in error_rows if row["sanity_issue_count"] > 0),
            "errors_by_type": counter_from_rows(error_rows, "error_type"),
            "errors_by_true_name": counter_from_rows(error_rows, "true_name"),
            "errors_by_pred_name": counter_from_rows(error_rows, "pred_name"),
            "errors_by_modal": counter_from_rows(error_rows, "modal"),
            "errors_by_target": counter_from_rows(error_rows, "target"),
            "top_hard_errors": sorted(
                error_rows,
                key=lambda row: (row["pred_confidence"], row["score_margin"]),
                reverse=True,
            )[:topk],
        }
    )
    return summary


def print_console_summary(summary: Dict[str, Any], skip_inference: bool) -> None:
    print("\n================ Error Diagnosis Summary ================")
    print(f"task: {summary['task']}")
    print(f"dataset: {summary['dataset_path']}")
    if summary.get("checkpoint_path"):
        print(f"checkpoint: {summary['checkpoint_path']}")
    print(f"total_samples: {summary['total_samples']}")
    print(f"samples_with_any_sanity_issue: {summary['samples_with_any_sanity_issue']}")

    if summary["sanity_issue_counts"]:
        print("sanity_issue_counts:")
        for issue_name, count in sorted(summary["sanity_issue_counts"].items(), key=lambda item: (-item[1], item[0])):
            print(f"  {issue_name}: {count}")
    else:
        print("sanity_issue_counts: none")

    if skip_inference:
        print("inference: skipped")
        return

    print(f"total_errors: {summary['total_errors']}")
    print(f"error_rate: {summary['error_rate']}")
    print(f"errors_with_sanity_issue: {summary['errors_with_sanity_issue']}")
    if summary["errors_by_type"]:
        print("errors_by_type:")
        for error_type, count in sorted(summary["errors_by_type"].items(), key=lambda item: (-item[1], item[0])):
            print(f"  {error_type}: {count}")


def save_reports(
    output_dir: Path,
    task_name: str,
    split_name: str,
    sanity_rows: List[Dict[str, Any]],
    prediction_rows: Optional[List[Dict[str, Any]]],
    summary: Dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    sanity_file = output_dir / f"{task_name}_{split_name}_sanity.csv"
    pd.DataFrame(sanity_rows).to_csv(sanity_file, index=False, encoding="utf-8-sig")

    if prediction_rows is not None:
        prediction_file = output_dir / f"{task_name}_{split_name}_predictions.csv"
        pd.DataFrame(prediction_rows).to_csv(prediction_file, index=False, encoding="utf-8-sig")

        error_rows = [row for row in prediction_rows if row["correct"] == 0]
        error_file = output_dir / f"{task_name}_{split_name}_errors.csv"
        pd.DataFrame(error_rows).to_csv(error_file, index=False, encoding="utf-8-sig")

    summary_file = output_dir / f"{task_name}_{split_name}_summary.json"
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nSaved files:")
    print(f"  {sanity_file}")
    if prediction_rows is not None:
        print(f"  {output_dir / f'{task_name}_{split_name}_predictions.csv'}")
        print(f"  {output_dir / f'{task_name}_{split_name}_errors.csv'}")
    print(f"  {summary_file}")


def main() -> None:
    args = parse_args()
    config, dataset_path = build_config(args)
    records = load_records(dataset_path)
    meme_dir = Path(config.meme_path)

    sanity_rows, sanity_issue_counts = sanity_check_records(records, meme_dir, args.task)

    checkpoint_path: Optional[Path] = None
    prediction_rows: Optional[List[Dict[str, Any]]] = None

    if not args.skip_inference:
        if args.checkpoint:
            checkpoint_path = Path(args.checkpoint)
        else:
            checkpoint_path = find_checkpoint(Path(config.checkpoint_path), args.model, args.task)
        if checkpoint_path is None or not checkpoint_path.exists():
            raise FileNotFoundError("No checkpoint found. Please pass --checkpoint explicitly.")

        prediction_rows = run_inference(config, records, sanity_rows, checkpoint_path, args.task)

    split_name = Path(args.data_file).stem if args.data_file else args.split
    summary = build_summary(
        task_name=args.task,
        dataset_path=dataset_path,
        checkpoint_path=checkpoint_path,
        sanity_rows=sanity_rows,
        sanity_issue_counts=sanity_issue_counts,
        prediction_rows=prediction_rows,
        topk=args.topk,
    )

    save_reports(
        output_dir=Path(args.output_dir),
        task_name=args.task,
        split_name=split_name,
        sanity_rows=sanity_rows,
        prediction_rows=prediction_rows,
        summary=summary,
    )
    print_console_summary(summary, args.skip_inference)


if __name__ == "__main__":
    main()
