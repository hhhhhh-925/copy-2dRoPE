#!/usr/bin/env python3
"""Evaluate OpenAI API models on zhangyir/Copy_Benchmark.

This script calls the OpenAI Responses API and scores model outputs with the
same rules used by the benchmark eval code:

- binary-copy-recursive-flip: strict string match after strip().
- binary-copy-imbalanced: extract a/A/b/B, map a/A -> 1 and b/B -> 0, then compare.
- python-list-conversion: extract numbers and compare the full number sequence.

Example:
    python eval_openai_copy_benchmark.py \
        --model gpt-5.5 \
        --dataset zhangyir/Copy_Benchmark \
        --subset binary-copy-recursive-flip \
        --limit 5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm

BENCHMARK_DATASET = "zhangyir/Copy_Benchmark"
SUBSETS = ["binary-copy-recursive-flip", "binary-copy-imbalanced", "python-list-conversion"]

AB_RE = re.compile(r"[abAB]")
NUMBER_RE = re.compile(r"-?\d+")


# -----------------------------------------------------------------------------
# JSON / dataset loading
# -----------------------------------------------------------------------------


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL line {line_no} in {path}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL line {line_no} in {path} is not an object")
            rows.append(obj)
    return rows


def load_hf_subset(dataset_name: str, subset: str, split: str, revision: Optional[str]) -> List[Dict[str, Any]]:
    """Load one subset/config from Hugging Face.

    The normal path is load_dataset(dataset, subset, split=...).  The fallback
    supports repos whose data files are directly exposed as JSONL files.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("Please install datasets: pip install datasets") from e

    load_kwargs: Dict[str, Any] = {}
    if revision:
        load_kwargs["revision"] = revision

    try:
        ds = load_dataset(dataset_name, subset, split=split, **load_kwargs)
    except Exception as first_error:
        # Fallback for a repo layout like data/binary-copy-recursive-flip.jsonl.
        data_file = f"hf://datasets/{dataset_name}/data/{subset}.jsonl"
        try:
            ds = load_dataset("json", data_files=data_file, split="train", **load_kwargs)
        except Exception as second_error:
            raise RuntimeError(
                f"Failed to load subset {subset!r} from {dataset_name!r}.\n"
                f"First error: {first_error!r}\nSecond error: {second_error!r}"
            ) from second_error

    return [dict(x) for x in ds]


def load_records_for_subset(args: argparse.Namespace, subset: str) -> List[Dict[str, Any]]:
    if args.data_file:
        if args.subset == "all":
            raise ValueError("--subset all is only supported with --dataset, not --data-file")
        return read_jsonl(args.data_file)
    return load_hf_subset(args.dataset, subset, args.split, args.revision)


# -----------------------------------------------------------------------------
# Record parsing
# -----------------------------------------------------------------------------


def maybe_json_loads(x: Any) -> Any:
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("{") or s.startswith("["):
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                return x
    return x


def content_to_text(content: Any) -> str:
    """Convert a dataset/OpenAI-style message content field to plain text."""
    content = maybe_json_loads(content)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: List[str] = []
        for item in content:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict):
                # Common formats: {"type":"text", "text":"..."} or
                # {"type":"input_text", "text":"..."}.
                if "text" in item:
                    pieces.append(str(item["text"]))
                elif "content" in item:
                    pieces.append(content_to_text(item["content"]))
            else:
                pieces.append(str(item))
        return "".join(pieces)
    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])
        if "content" in content:
            return content_to_text(content["content"])
    return str(content)


def parse_input_obj(record: Dict[str, Any]) -> Dict[str, Any]:
    input_obj = maybe_json_loads(record.get("input"))
    if not isinstance(input_obj, dict) or "messages" not in input_obj:
        raise ValueError(f"Record {record.get('id')} has no input.messages")
    return input_obj


def get_prompt_messages_and_gold(record: Dict[str, Any]) -> Tuple[List[Dict[str, str]], str]:
    """Use messages before the first assistant message as prompt; first assistant as gold."""
    input_obj = parse_input_obj(record)
    messages = input_obj.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"Record {record.get('id')} input.messages is not a list")

    prompt_messages: List[Dict[str, str]] = []
    gold: Optional[str] = None

    for raw_msg in messages:
        if not isinstance(raw_msg, dict):
            continue
        role = str(raw_msg.get("role", "user"))
        content = content_to_text(raw_msg.get("content", ""))

        if role == "assistant" and gold is None:
            gold = content
            break

        prompt_messages.append({"role": role, "content": content})

    if gold is None:
        raise ValueError(f"Record {record.get('id')} has no assistant gold answer")
    if not prompt_messages:
        raise ValueError(f"Record {record.get('id')} has no prompt messages before gold answer")

    return prompt_messages, gold


def get_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    metadata = record.get("metadata", record.get("meta", {}))
    metadata = maybe_json_loads(metadata)
    return metadata if isinstance(metadata, dict) else {}


def get_record_id(record: Dict[str, Any], subset: str, absolute_index: int) -> str:
    rid = record.get("id", None)
    if rid is None or str(rid) == "":
        return f"{subset}:{absolute_index}"
    return str(rid)


def select_records(records: List[Dict[str, Any]], start: int, limit: Optional[int]) -> List[Tuple[int, Dict[str, Any]]]:
    if start < 0:
        raise ValueError("--start must be non-negative")
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive")

    indexed = list(enumerate(records))
    indexed = indexed[start:]
    if limit is not None:
        indexed = indexed[:limit]
    return indexed


# -----------------------------------------------------------------------------
# Benchmark scoring, matching eval_benchmark.py logic
# -----------------------------------------------------------------------------


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_+-]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def normalize_ab_to_binary(text: str) -> str:
    """Extract a/A/b/B and map a/A -> 1, b/B -> 0."""
    chars = AB_RE.findall(text)
    return "".join("1" if c.lower() == "a" else "0" for c in chars)


def extract_numbers(text: str) -> List[str]:
    return NUMBER_RE.findall(strip_code_fences(text))


def score_01_copy(prediction: str, gold: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    target = gold.strip()
    parsed = prediction.strip()
    return {
        "metric": "strict_string_match",
        "match": parsed == target,
        "parsed_output": parsed,
        "target": target,
    }


def score_ab_copy(prediction: str, gold: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    # Prefer the benchmark metadata target when available; otherwise derive it
    # from the assistant gold answer.
    target_binary = metadata.get("target_binary")
    if not isinstance(target_binary, str):
        target_binary = normalize_ab_to_binary(gold)

    parsed = normalize_ab_to_binary(prediction)
    return {
        "metric": "ab_extracted_binary_match",
        "match": parsed == target_binary,
        "parsed_output": parsed,
        "target": target_binary,
    }


def score_python_list_conversion(prediction: str, gold: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    pred_nums = extract_numbers(prediction)
    gold_nums = extract_numbers(gold)
    return {
        "metric": "number_sequence_match",
        "match": pred_nums == gold_nums,
        "parsed_output": pred_nums,
        "target": gold_nums,
        "pred_num_count": len(pred_nums),
        "gold_num_count": len(gold_nums),
    }


def score_prediction(prediction: str, gold: str, subset: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    if subset == "binary-copy-recursive-flip":
        return score_01_copy(prediction, gold, metadata)
    if subset == "binary-copy-imbalanced":
        return score_ab_copy(prediction, gold, metadata)
    if subset == "python-list-conversion":
        return score_python_list_conversion(prediction, gold, metadata)
    raise ValueError(f"Unknown subset: {subset}")


# -----------------------------------------------------------------------------
# OpenAI API call
# -----------------------------------------------------------------------------


def convert_messages_for_openai(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    converted: List[Dict[str, str]] = []
    for msg in messages:
        role = str(msg.get("role", "user"))
        if role not in {"system", "developer", "user", "assistant"}:
            role = "user"
        converted.append({"role": role, "content": str(msg.get("content", ""))})
    return converted


def extract_response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str):
        return text

    data = response.model_dump() if hasattr(response, "model_dump") else dict(response)
    pieces: List[str] = []
    for item in data.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if content.get("type") in {"output_text", "text"}:
                pieces.append(str(content.get("text", "")))
    return "".join(pieces)


def extract_usage(response: Any) -> Dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "reasoning_tokens": None,
            "total_tokens": None,
        }

    usage_dict = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
    output_details = usage_dict.get("output_tokens_details") or {}
    return {
        "prompt_tokens": usage_dict.get("input_tokens"),
        "completion_tokens": usage_dict.get("output_tokens"),
        "reasoning_tokens": output_details.get("reasoning_tokens"),
        "total_tokens": usage_dict.get("total_tokens"),
    }


def call_openai_with_retries(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    max_output_tokens: int,
    temperature: Optional[float],
    max_attempts: int,
    retry_sleep_sec: float,
    max_retry_sleep_sec: float,
) -> Dict[str, Any]:
    last_error: Optional[str] = None

    for attempt in range(1, max_attempts + 1):
        try:
            kwargs: Dict[str, Any] = {
                "model": model,
                "input": convert_messages_for_openai(messages),
                "max_output_tokens": max_output_tokens,
            }
            if temperature is not None:
                kwargs["temperature"] = temperature

            t0 = time.time()
            response = client.responses.create(**kwargs)
            latency_sec = time.time() - t0

            return {
                "prediction": extract_response_text(response),
                "api_error": None,
                "attempts": attempt,
                "latency_sec": latency_sec,
                "response_id": getattr(response, "id", None),
                "response_status": getattr(response, "status", None),
                **extract_usage(response),
            }
        except Exception as e:
            last_error = repr(e)
            if attempt < max_attempts:
                sleep_sec = min(retry_sleep_sec * (2 ** (attempt - 1)), max_retry_sleep_sec)
                time.sleep(sleep_sec)

    return {
        "prediction": "",
        "api_error": last_error,
        "attempts": max_attempts,
        "latency_sec": None,
        "response_id": None,
        "response_status": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "reasoning_tokens": None,
        "total_tokens": None,
    }


# -----------------------------------------------------------------------------
# Evaluation loop and summaries
# -----------------------------------------------------------------------------


def safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name)


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def last_row_per_id(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for i, row in enumerate(rows):
        rid = str(row.get("id", f"__row_{i}"))
        if rid not in latest:
            order.append(rid)
        latest[rid] = row
    return [latest[rid] for rid in order]


def mean_ignore_none(values: Iterable[Any]) -> Optional[float]:
    nums = [float(v) for v in values if isinstance(v, (int, float))]
    return sum(nums) / len(nums) if nums else None


def make_summary(rows: List[Dict[str, Any]], args: argparse.Namespace, subset: str, predictions_path: Path) -> Dict[str, Any]:
    deduped_rows = last_row_per_id(rows)
    n = len(deduped_rows)
    correct = sum(1 for row in deduped_rows if row.get("match") is True)
    api_errors = sum(1 for row in deduped_rows if row.get("api_error"))

    return {
        "dataset": args.dataset if not args.data_file else str(args.data_file),
        "subset": subset,
        "model": args.model,
        "num_examples": n,
        "num_correct": correct,
        "num_api_errors": api_errors,
        "accuracy": correct / n if n else 0.0,
        "avg_prompt_tokens": mean_ignore_none(row.get("prompt_tokens") for row in deduped_rows),
        "avg_completion_tokens": mean_ignore_none(row.get("completion_tokens") for row in deduped_rows),
        "avg_reasoning_tokens": mean_ignore_none(row.get("reasoning_tokens") for row in deduped_rows),
        "avg_total_tokens": mean_ignore_none(row.get("total_tokens") for row in deduped_rows),
        "avg_latency_sec": mean_ignore_none(row.get("latency_sec") for row in deduped_rows),
        "predictions_path": str(predictions_path),
        "note": "If --resume --retry-errors was used, duplicate older error rows may exist in predictions.jsonl; this summary uses the last row for each id.",
    }


def evaluate_one_subset(args: argparse.Namespace, client: OpenAI, subset: str) -> Dict[str, Any]:
    records = load_records_for_subset(args, subset)
    selected = select_records(records, args.start, args.limit)
    if not selected:
        raise ValueError(f"No records selected for subset {subset}")

    out_dir = Path(args.output_dir) / safe_name(args.model) / safe_name(subset)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "predictions.jsonl"
    summary_path = out_dir / "summary.json"

    if predictions_path.exists() and not args.resume:
        predictions_path.unlink()

    existing_rows = read_jsonl(predictions_path) if predictions_path.exists() else []
    latest_rows = last_row_per_id(existing_rows)

    done_ids = set()
    for row in latest_rows:
        had_error = bool(row.get("api_error"))
        if not (args.retry_errors and had_error):
            done_ids.add(str(row.get("id")))

    for absolute_index, record in tqdm(selected, desc=f"{subset}"):
        record_id = get_record_id(record, subset, absolute_index)
        if args.resume and record_id in done_ids:
            continue

        metadata = get_metadata(record)
        prompt_messages, gold = get_prompt_messages_and_gold(record)

        api_result = call_openai_with_retries(
            client=client,
            model=args.model,
            messages=prompt_messages,
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
            max_attempts=args.max_attempts,
            retry_sleep_sec=args.retry_sleep_sec,
            max_retry_sleep_sec=args.max_retry_sleep_sec,
        )

        score = score_prediction(api_result["prediction"], gold, subset, metadata)
        matched = bool(score["match"]) and api_result["api_error"] is None

        row: Dict[str, Any] = {
            "id": record_id,
            "subset": subset,
            "model": args.model,
            "metadata": metadata,
            "metric": score["metric"],
            "match": matched,
            "prediction": api_result["prediction"],
            "gold": gold,
            "parsed_output": score.get("parsed_output"),
            "target": score.get("target"),
            **{k: v for k, v in score.items() if k not in {"metric", "match", "parsed_output", "target"}},
            **api_result,
        }
        if args.save_prompt:
            row["prompt_messages"] = prompt_messages

        append_jsonl(predictions_path, row)

        if args.request_gap_sec > 0:
            time.sleep(args.request_gap_sec)

    all_rows = read_jsonl(predictions_path)
    summary = make_summary(all_rows, args, subset, predictions_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY or pass --api-key.")

    client = OpenAI(api_key=api_key, timeout=args.timeout_sec, max_retries=0)

    subsets = SUBSETS if args.subset == "all" else [args.subset]
    summaries = []
    for subset in subsets:
        summaries.append(evaluate_one_subset(args, client, subset))

    if len(summaries) == 1:
        final_summary: Dict[str, Any] = summaries[0]
    else:
        total_n = sum(s["num_examples"] for s in summaries)
        total_correct = sum(s["num_correct"] for s in summaries)
        total_errors = sum(s["num_api_errors"] for s in summaries)
        final_summary = {
            "dataset": args.dataset,
            "model": args.model,
            "subsets": summaries,
            "overall_num_examples": total_n,
            "overall_num_correct": total_correct,
            "overall_num_api_errors": total_errors,
            "overall_accuracy": total_correct / total_n if total_n else 0.0,
        }
        out_dir = Path(args.output_dir) / safe_name(args.model)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary_all.json").write_text(
            json.dumps(final_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    print(json.dumps(final_summary, ensure_ascii=False, indent=2))
    return final_summary


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_temperature(x: str) -> Optional[float]:
    if x.lower() in {"none", "null", "default"}:
        return None
    return float(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OpenAI API models on zhangyir/Copy_Benchmark.")
    parser.add_argument("--model", default="gpt-5.5", help="OpenAI model name, e.g. gpt-5.5 or gpt-5.5-mini.")
    parser.add_argument("--api-key", default=None, help="Optional API key. If omitted, uses OPENAI_API_KEY.")

    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--dataset", default=BENCHMARK_DATASET, help="Hugging Face dataset repo.")
    source_group.add_argument("--data-file", default=None, help="Local JSONL file for one subset.")

    parser.add_argument("--subset", default="binary-copy-recursive-flip", choices=SUBSETS + ["all"], help="Subset to evaluate.")
    parser.add_argument("--split", default="train", help="Dataset split.")
    parser.add_argument("--revision", default=None, help="Optional Hugging Face dataset revision/commit.")

    parser.add_argument("--output-dir", default="openai_eval_outputs")
    parser.add_argument("--max-output-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=parse_temperature, default=0.0, help="Use 'none' to omit temperature.")
    parser.add_argument("--timeout-sec", type=float, default=300.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-sleep-sec", type=float, default=2.0)
    parser.add_argument("--max-retry-sleep-sec", type=float, default=30.0)
    parser.add_argument("--request-gap-sec", type=float, default=0.0)

    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only this many examples per subset.")
    parser.add_argument("--save-prompt", action="store_true", help="Save prompt messages in predictions.jsonl.")
    parser.add_argument("--resume", action="store_true", help="Skip ids that already have a successful or finished row.")
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="With --resume, retry ids whose latest row has api_error. Summary uses latest row per id.",
    )

    args = parser.parse_args()
    if args.data_file and args.subset == "all":
        parser.error("--subset all is not supported with --data-file")
    if args.max_attempts <= 0:
        parser.error("--max-attempts must be positive")
    return args


def main() -> None:
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
