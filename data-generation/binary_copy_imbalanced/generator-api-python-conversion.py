# make_dataset_fixed_lengths.py
import argparse
import json
import math
import random
from typing import Any, Dict, List
import os
from collections import Counter


PROMPT_TEMPLATES = [
    "Please repeat the following sequence of physical sensor measurements as a Python list:",
    "Convert this sequence of physical sensor readings into a Python list:",
    "Copy these sensor measurements into a Python-style list:",
    "Write the following physical measurement sequence as a Python list:",
    "Return this sequence of sensor values as a Python list:",
    "Reformat the following physical sensor sequence into a Python list:",
    "Please copy the following numeric sensor sequence into a Python list:",
    "Turn these physical sensor measurements into a comma-separated Python list:",
    "Repeat the sequence below as a Python list:",
    "Output the following sensor readings as a Python list:",
]


EXACT_REPEAT_SCENARIOS = [
    "simple_harmonic_motion",
    "triangle_wave_motion",
    "sawtooth_scan_motion",
    "rectified_ac_signal",
    "clipped_sensor_oscillation",
    "relaxation_oscillator",
    "fourier_periodic_waveform",
    "pulse_train",
    "pendulum_like_oscillation",
]


def fmt_float(x: float, decimals: int = 2) -> str:
    """
    Format a float as a stable numeric token.
    Avoid producing "-0.00".
    """
    if abs(x) < 0.5 * (10 ** (-decimals)):
        x = 0.0
    return f"{x:.{decimals}f}"


def triangle_wave(theta: float) -> float:
    """
    Periodic triangle wave in [-1, 1].
    """
    u = (theta / (2 * math.pi)) % 1.0
    return 4.0 * abs(u - 0.5) - 1.0


def sawtooth_wave(theta: float) -> float:
    """
    Periodic sawtooth wave in [-1, 1].
    """
    u = (theta / (2 * math.pi)) % 1.0
    return 2.0 * u - 1.0


def generate_signal(
    rng: random.Random,
    n: int,
    min_period: int,
    max_period: int,
) -> Dict[str, Any]:
    """
    Generate one exactly repeated cyclic physical-like 1D sequence.

    Important:
    We first generate one cycle, then repeat that cycle by indexing.
    Therefore ys[k] == ys[k + period] exactly at the Python object/value level,
    instead of relying on recomputing sin(theta + 2*pi).
    """
    scenario = rng.choice(EXACT_REPEAT_SCENARIOS)

    p = rng.randint(min_period, max_period)
    w = 2 * math.pi / p

    A = rng.uniform(0.5, 3.0)
    phi = rng.uniform(0, 2 * math.pi)
    offset = rng.uniform(-0.5, 0.5)

    # Scenario-specific parameters.
    rectified_mode = rng.choice(["full", "half"])
    clip_level = rng.uniform(0.45, 0.8) * A
    tau = rng.uniform(0.12, 0.45)
    pulse_width = rng.uniform(0.05, 0.18)
    pulse_center = rng.uniform(0.15, 0.85)
    harmonic_strength = rng.uniform(0.05, 0.25)

    # Fourier waveform parameters.
    num_harmonics = rng.randint(2, 4)
    fourier_coeffs = []
    for h in range(1, num_harmonics + 1):
        amp_h = rng.uniform(0.15, 1.0) / h
        phase_h = rng.uniform(0, 2 * math.pi)
        fourier_coeffs.append((h, amp_h, phase_h))

    cycle: List[float] = []

    for r in range(p):
        theta = w * r + phi

        if scenario == "simple_harmonic_motion":
            y = offset + A * math.sin(theta)

        elif scenario == "triangle_wave_motion":
            y = offset + A * triangle_wave(theta)

        elif scenario == "sawtooth_scan_motion":
            y = offset + A * sawtooth_wave(theta)

        elif scenario == "rectified_ac_signal":
            s = math.sin(theta)

            if rectified_mode == "full":
                value = abs(s)
            else:
                value = max(0.0, s)

            y = offset + A * value

        elif scenario == "clipped_sensor_oscillation":
            raw = A * math.sin(theta)
            clipped = max(-clip_level, min(clip_level, raw))
            y = offset + clipped

        elif scenario == "relaxation_oscillator":
            u = ((r / p) + phi / (2 * math.pi)) % 1.0
            value = 1.0 - math.exp(-u / tau)
            y = offset + A * value

        elif scenario == "fourier_periodic_waveform":
            value = 0.0

            for h, amp_h, phase_h in fourier_coeffs:
                value += amp_h * math.sin(h * theta + phase_h)

            y = offset + A * value

        elif scenario == "pulse_train":
            u = ((r / p) + phi / (2 * math.pi)) % 1.0
            d = min(abs(u - pulse_center), 1.0 - abs(u - pulse_center))
            pulse = math.exp(-(d ** 2) / (2 * pulse_width ** 2))
            y = offset + A * pulse

        elif scenario == "pendulum_like_oscillation":
            y = (
                offset
                + A * math.sin(theta)
                + harmonic_strength * A * math.sin(3 * theta)
            )

        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        cycle.append(y)

    ys = [cycle[k % p] for k in range(n)]

    return {
        "scenario": scenario,
        "period": p,
        "values": ys,
    }


def build_prompt(data_text: str, rng: random.Random) -> str:
    instruction = rng.choice(PROMPT_TEMPLATES)
    return f"{instruction}\n{data_text}"


def build_answer(tokens: List[str]) -> str:
    return "[" + ",".join(tokens) + ",]"


def y_sequence_generator(
    idx: int,
    n: int,
    min_period: int,
    max_period: int,
    seed: int,
    decimals: int,
) -> Dict[str, Any]:
    rng = random.Random(seed + idx)

    signal = generate_signal(
        rng=rng,
        n=n,
        min_period=min_period,
        max_period=max_period,
    )

    tokens = [fmt_float(y, decimals=decimals) for y in signal["values"]]

    # User and assistant sides both use comma-separated numeric values.
    data_text = ",".join(tokens)
    prompt = build_prompt(data_text, rng)
    answer = build_answer(tokens)

    return {
        "prompt": prompt,
        "answer": answer,
        "meta": {
            "scenario": signal["scenario"],
            "length": n,
            "period": signal["period"],
        },
    }


def parse_lengths(lengths_text: str) -> List[int]:
    lengths = []
    for part in lengths_text.split(","):
        part = part.strip()
        if not part:
            continue
        length = int(part)
        if length <= 0:
            raise ValueError(f"All lengths must be positive, got {length}.")
        lengths.append(length)

    if not lengths:
        raise ValueError("--lengths must contain at least one length.")

    return lengths


def build_dataset(
    lengths: List[int],
    samples_per_length: int,
    min_period: int,
    max_period: int,
    seed: int,
    decimals: int,
) -> List[Dict[str, Any]]:
    items = []
    idx = 0

    # This guarantees exactly `samples_per_length` samples for every length.
    for n in lengths:
        for j in range(samples_per_length):
            sample = y_sequence_generator(
                idx=idx,
                n=n,
                min_period=min_period,
                max_period=max_period,
                seed=seed,
                decimals=decimals,
            )

            items.append({
                "id": f"phys_copy_len{n}_{j:03d}",
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": sample["prompt"],
                        },
                        {
                            "role": "assistant",
                            "content": sample["answer"],
                        },
                    ]
                },
                "meta": sample["meta"],
            })
            idx += 1

    return items


def save_jsonl(items: List[Dict[str, Any]], path: str) -> None:
    out_dir = os.path.dirname(path)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def format_sft_message(role: str, content: str) -> str:
    """
    Match your SFT builder's message format:

        ### user:
        ...

        ### assistant:
        ...
    """
    return f"### {role}:\n{content}\n\n"


def extract_messages_from_item(item: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Supports both possible formats:

    1) {"messages": [...]}
    2) {"input": {"messages": [...]}}
    """
    if "messages" in item:
        return item["messages"]

    if "input" in item and "messages" in item["input"]:
        return item["input"]["messages"]

    raise KeyError(f"Cannot find messages in item keys: {item.keys()}")


def sft_char_length(item: Dict[str, Any]) -> int:
    """
    Character length under the same formatting used by the SFT builder.
    This includes:
    - role headers
    - newlines
    - user prompt
    - sequence to copy
    - assistant answer
    """
    messages = extract_messages_from_item(item)

    text = ""
    for message in messages:
        text += format_sft_message(
            role=message["role"],
            content=message["content"],
        )

    return len(text)


def compute_char_stats(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    max_sft_chars = -1
    max_sft_item_id = None

    max_jsonl_line_chars = -1
    max_jsonl_item_id = None

    max_user_content_chars = -1
    max_user_content_item_id = None

    max_assistant_content_chars = -1
    max_assistant_content_item_id = None

    max_total_content_chars = -1
    max_total_content_item_id = None

    for item in items:
        item_id = item.get("id", "unknown")

        cur_sft_chars = sft_char_length(item)
        if cur_sft_chars > max_sft_chars:
            max_sft_chars = cur_sft_chars
            max_sft_item_id = item_id

        # This is the on-disk JSONL line length, including the trailing "\n".
        cur_jsonl_line_chars = len(json.dumps(item, ensure_ascii=False)) + 1
        if cur_jsonl_line_chars > max_jsonl_line_chars:
            max_jsonl_line_chars = cur_jsonl_line_chars
            max_jsonl_item_id = item_id

        messages = extract_messages_from_item(item)

        cur_total_content_chars = 0

        for message in messages:
            role = message["role"]
            content_len = len(message["content"])
            cur_total_content_chars += content_len

            if role == "user" and content_len > max_user_content_chars:
                max_user_content_chars = content_len
                max_user_content_item_id = item_id

            elif role == "assistant" and content_len > max_assistant_content_chars:
                max_assistant_content_chars = content_len
                max_assistant_content_item_id = item_id

        if cur_total_content_chars > max_total_content_chars:
            max_total_content_chars = cur_total_content_chars
            max_total_content_item_id = item_id

    return {
        "max_sft_chars": max_sft_chars,
        "max_sft_item_id": max_sft_item_id,
        "max_total_content_chars": max_total_content_chars,
        "max_total_content_item_id": max_total_content_item_id,
        "max_user_content_chars": max_user_content_chars,
        "max_user_content_item_id": max_user_content_item_id,
        "max_assistant_content_chars": max_assistant_content_chars,
        "max_assistant_content_item_id": max_assistant_content_item_id,
        "max_jsonl_line_chars": max_jsonl_line_chars,
        "max_jsonl_item_id": max_jsonl_item_id,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lengths",
        type=str,
        default="100,200,300,400,500,600,700,800",
        help="Comma-separated sequence lengths. Default: 200,300,400,500.",
    )
    parser.add_argument(
        "--samples_per_length",
        type=int,
        default=50,
        help="Number of examples for each length. Default: 50.",
    )

    parser.add_argument("--min_period", type=int, default=2)
    parser.add_argument("--max_period", type=int, default=10)

    parser.add_argument("--decimals", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument("--out", type=str, default="physics_copy_fixed_lengths_100_800.jsonl")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    lengths = parse_lengths(args.lengths)

    if args.samples_per_length <= 0:
        raise ValueError("--samples_per_length must be positive.")

    if args.min_period <= 0:
        raise ValueError("--min_period must be positive.")

    if args.max_period < args.min_period:
        raise ValueError("--max_period must be >= --min_period.")

    if args.decimals < 0:
        raise ValueError("--decimals must be non-negative.")

    dataset = build_dataset(
        lengths=lengths,
        samples_per_length=args.samples_per_length,
        min_period=args.min_period,
        max_period=args.max_period,
        seed=args.seed,
        decimals=args.decimals,
    )

    save_jsonl(dataset, args.out)

    char_stats = compute_char_stats(dataset)
    length_counts = Counter(item["meta"]["length"] for item in dataset)

    print(f"Saved: {args.out}")
    print(f"num_samples = {len(dataset)}")
    print(f"lengths = {lengths}")
    print(f"samples_per_length = {args.samples_per_length}")
    print(f"length_counts = {dict(sorted(length_counts.items()))}")
    print(f"period range = [{args.min_period}, {args.max_period}]")
    print(f"decimals = {args.decimals}")

    print("Character length stats:")
    print(f"max_sft_chars = {char_stats['max_sft_chars']}")
    print(f"max_sft_item_id = {char_stats['max_sft_item_id']}")

    print(f"max_total_content_chars = {char_stats['max_total_content_chars']}")
    print(f"max_total_content_item_id = {char_stats['max_total_content_item_id']}")

    print(f"max_user_content_chars = {char_stats['max_user_content_chars']}")
    print(f"max_user_content_item_id = {char_stats['max_user_content_item_id']}")

    print(f"max_assistant_content_chars = {char_stats['max_assistant_content_chars']}")
    print(f"max_assistant_content_item_id = {char_stats['max_assistant_content_item_id']}")

    print(f"max_jsonl_line_chars = {char_stats['max_jsonl_line_chars']}")
    print(f"max_jsonl_item_id = {char_stats['max_jsonl_item_id']}")


if __name__ == "__main__":
    main()
