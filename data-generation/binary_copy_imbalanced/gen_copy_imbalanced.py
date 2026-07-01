#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a copy-task dataset: random binary strings over a small vocabulary,
wrapped in varied natural-language prompts, as chat-style SFT examples.

Splits are defined in ``SPLITS`` (name, length range, biased flag). Edit that
list to enable/disable splits; each active split is written to
``<out_dir>/<split_name>.jsonl``.

Output format (JSONL): one JSON object per line, each with:
  - id: stable example id
  - messages: [ {"role": "user", "content": ...}, {"role": "assistant", "content": s} ]
  - source: "copy"

For ``biased=True`` splits, token ``a`` vs ``b`` is drawn with probabilities
from ``HARD_P_VALUES``; otherwise it is fair (0.5 / 0.5).

Usage:
  python gen_copy_imbalanced.py --out_dir ./data/copy-imbalanced --seed 0
"""

from pathlib import Path
import argparse
import json
import random
from dataclasses import dataclass
from tqdm import tqdm


# ----------------------------
# Prompt templates
# ----------------------------
PROMPT_TEMPLATES: list[str] = [
    # With newline
    "Please echo the next sequence verbatim (no extra text). Here it is:\n{s}",
    "Copy the text below exactly as-is (keep every character):\n{s}",
    "Task: return the string shown below. Do not add or remove anything.\n{s}",
    "Print the following content exactly, unchanged. Output only the content:\n{s}",
    "Write the following string exactly as it is:\n{s}",
    "I want you to say the following string and nothing else:\n{s}",
    "Say the following string exactly as it is:\n{s}",
    "Here is a string:\n{s}\n\nPlease print the string exactly as it is.",
    "I will give you a string. You need to repeat it exactly as it is. Here it is:\n{s}",
    "{s}\n\nPlease print the above string exactly as it is.",
    "{s}\n\nCan you write out the above string and nothing else.",
    "Can you copy a string of characters? Here is the string I want you to copy:\n{s}",
    "Repeat this string and say nothing else:\n{s}",
    # Without newline
    "Repeat the following string after me (output only the string): {s}",
    "Echo this sequence exactly (no extra characters): {s}",
    "Return the exact same string and nothing else -> {s}",
    "Copy-paste the following payload exactly as written: {s}",
    "Output EXACTLY the string below, unchanged: {s}",
    'Just say "{s}" and nothing else.',
    "Say '{s}' and say nothing else.",
]

VOCABULARY = [
    "0", "1",
    # " a", " b",
]

HARD_P_VALUES: list[float] = [0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95]


@dataclass(frozen=True)
class SplitSpec:
    name: str
    min_len: int
    max_len: int
    imbalanced: bool


SPLITS: list[SplitSpec] = [
    SplitSpec("train_01_10_1000_unbal", 10, 1000, True),
    SplitSpec("train_01_1000_2000_unbal", 1000, 2000, True),
    SplitSpec("train_01_2000_4000_unbal", 2000, 4000, True),
    SplitSpec("train_01_4000_8000_unbal", 4000, 8000, True),
    SplitSpec("train_01_8000_16000_unbal", 8000, 16000, True),
    SplitSpec("train_01_16000_32000_unbal", 16000, 32000, True),
]


def gen_one_example(
    a: str,
    b: str,
    n_tokens: int,
    p: float = 0.5,
) -> str:
    """
    Generate s by concatenating `length_tokens` tokens, each token is either a or b.

    - If p is None: P(a)=0.5, P(b)=0.5
    - Else: P(a)=p, P(b)=1-p
    """
    if n_tokens <= 0:
        return ""

    # Biased Bernoulli
    # P(a)=p
    return "".join(a if random.random() < p else b for _ in range(n_tokens))


def generate_data(
    n_examples: int,
    biased: bool,
    min_len: int,
    max_len: int,
) -> list[str]:
    """

    Each job tuple: (token_set_index, prompt_template_index, p_value_or_None)

    We balance prompt templates within each token set:
      - 2000 samples -> each of 10 templates appears 200 times per token set.
    For hard split, we also balance p-values within each token set:
      - 2000 samples -> each of 5 p-values appears 400 times per token set.
    """
    n_templates = len(PROMPT_TEMPLATES)
    if n_templates == 0:
        raise ValueError("PROMPT_TEMPLATES is empty.")

    examples: list[str] = []
    for i in tqdm(range(n_examples)):
        length = random.randint(min_len, max_len)
        template = random.choice(PROMPT_TEMPLATES)
        a, b = random.sample(VOCABULARY, 2)

        # Balanced p-values for hard (biased) split
        p = random.choice(HARD_P_VALUES) if biased else 0.5

        s = gen_one_example(a, b, n_tokens=length, p=p)
        prompt = template.format(s=s)
        messages = [
            {
                "role": "user",
                "content": prompt,
            },
            {
                "role": "assistant",
                "content": s,
            },
        ]
        example = {
            "id": f"copy_balanced_{i}",
            "messages": messages,
            "source": "copy",
        }
        examples.append(example)

    return examples


def dump_data(
    path: Path,
    examples: list[dict],
) -> None:
    """
    Write examples to disk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for eg in tqdm(examples):
            f.write(json.dumps(eg, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        required=False,
        default=".",
        help="Output directory. Defaults to ./copy-sft.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed. Defaults to 0.",
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        default=200000,
        help="Samples per token-set template (per your spec, default=2000).",
    )
    args = parser.parse_args()

    for split in SPLITS:
        out_path = Path(args.out_dir) / f"{split.name}.jsonl"
        random.seed(args.seed)
        msgs = generate_data(
            n_examples=args.n_examples,
            biased=split.imbalanced,
            min_len=split.min_len,
            max_len=split.max_len,
        )
        print(f"Writing {len(msgs)} messages to {out_path}")
        dump_data(out_path, msgs)


if __name__ == "__main__":
    main()
