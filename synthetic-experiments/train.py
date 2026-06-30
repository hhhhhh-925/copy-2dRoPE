"""Train a small Transformer on synthetic copy tasks.

Usage:
    python train.py config.py

The config file is a regular Python file whose variables override the defaults
below. See configs/copy_12l_ada2d.py in the README for an example.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

from datagenerator import RandomGenerator, UnbalancedGenerator, vfkGenerator
from model import GPT, GPTConfig

# -----------------------------------------------------------------------------
# Defaults. Override them by running: python train.py config.py
# -----------------------------------------------------------------------------

# Output / evaluation
out_dir = "out/copy"
eval_interval = 500
log_interval = 500
train_eval_iters = 20
test_eval_iters = 250
eval_only = False
always_save_checkpoint = False
save_last = True
final_eval = True
final_eval_samples_per_interval = 10

# Optional experiment tracking. This repository does not store API keys.
# To enable SwanLab, set swanlab_log=True and export SWANLAB_API_KEY yourself.
swanlab_log = False
swanlab_project = "length-generalization"
swanlab_run_name = None

# Data / task
train_batch_size = 64
test_batch_size = 1
train_max_length = 100
test_max_length = 10000
last_test_length = 10000
train_type = "unbal"  # choices: rand, unbal, vfk
probs = [0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95]
pad_left = 2
pad_right = 4
vocab_size = 2100

# Model
n_layer = 12
n_head = 12
head_dim = 128
n_embd = n_head * head_dim
block_size = 26000
dropout = 0.0
bias = False
pe_type = "ada2d"  # choices: ada2d, 2drope, rope, nope, hrope, alibi, nrope
trainable_freqs = False
gated = True

# Optimizer / schedule
learning_rate = 1e-5
max_iters = 10000
lr_decay_iters = 10000
min_lr = 1e-6
warmup_iters = 100
weight_decay = 0.01
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True

# Reproducibility / system
seed = 42
device = "auto"  # auto, cpu, cuda, cuda:0, mps
# For CUDA: bfloat16, float16, float32. For CPU/MPS, float32 is safest.
dtype = "auto"
compile = False

# -----------------------------------------------------------------------------
# Override defaults from config file / command-line assignments.
# This follows the nanoGPT-style configurator: python train.py config.py --x=1
# -----------------------------------------------------------------------------
_config_value_types = (int, float, bool, str, list, tuple, type(None))
config_keys = [
    k for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, _config_value_types)
]
exec(open("configurator.py", encoding="utf-8").read())
config = {k: globals()[k] for k in config_keys if k in globals()}

# -----------------------------------------------------------------------------
# Validation / setup
# -----------------------------------------------------------------------------
PE_NAMES = {
    "ada2d": "Auto-2D-RoPE",
    "2drope": "2D-RoPE",
    "rope": "RoPE",
    "nope": "NoPE",
    "hrope": "Hybrid-RoPE",
    "alibi": "ALiBi",
    "nrope": "RNoPE",
}
GENERATOR_NAMES = {
    "rand": RandomGenerator,
    "unbal": UnbalancedGenerator,
    "vfk": vfkGenerator,
}

if pe_type not in PE_NAMES:
    raise ValueError(f"Unknown pe_type={pe_type!r}. Valid choices: {sorted(PE_NAMES)}")
if train_type not in GENERATOR_NAMES:
    raise ValueError(f"Unknown train_type={train_type!r}. Valid choices: {sorted(GENERATOR_NAMES)}")
if not isinstance(probs, (list, tuple)) or len(probs) == 0:
    raise ValueError("probs must be a non-empty list or tuple of probabilities.")

if device == "auto":
    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

if device.startswith("cuda"):
    device_type = "cuda"
elif device == "mps":
    device_type = "mps"
else:
    device_type = "cpu"

if dtype == "auto":
    if device_type == "cuda" and torch.cuda.is_bf16_supported():
        dtype = "bfloat16"
    elif device_type == "cuda":
        dtype = "float16"
    else:
        dtype = "float32"

if device_type != "cuda" and dtype != "float32":
    print(f"Warning: dtype={dtype!r} is usually unsafe on {device_type}; using float32 instead.")
    dtype = "float32"

Path(out_dir).mkdir(parents=True, exist_ok=True)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if device_type == "cuda":
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = (
    torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    if device_type == "cuda" and dtype in {"bfloat16", "float16"}
    else nullcontext()
)
pin_memory = device_type == "cuda"

print("Configuration:")
print(json.dumps(config, indent=2, sort_keys=True, default=str))
print(f"Using device={device}, dtype={dtype}, pe_type={pe_type} ({PE_NAMES[pe_type]})")

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

def get_batch(
    generator,
    max_length: int,
    p: list[float] | tuple[float, ...] = probs,
    noise: bool = False,
    size: int = test_batch_size,
    fixlength: bool = False,
    min_len: int = 0,
    max_len: int = 0,
):
    probabilities = random.choices(list(p), k=size)
    samples = [
        generator.generate(
            max_length,
            2 * max_length + 10,
            prob,
            noise=noise,
            fixlength=fixlength,
            min=min_len,
            max=max_len,
            pad_left=pad_left,
            pad_right=pad_right,
        )
        for prob in probabilities
    ]
    batch_x, batch_y, batch_length, batch_seq = map(list, zip(*samples))

    def as_tensor(values):
        return torch.tensor(values, dtype=torch.int64).pin_memory() if pin_memory else torch.tensor(values, dtype=torch.int64)

    x = as_tensor(batch_x).to(device, non_blocking=pin_memory)
    y = as_tensor(batch_y).to(device, non_blocking=pin_memory)
    batch_length = as_tensor(batch_length).to(device, non_blocking=pin_memory)
    batch_seq = as_tensor(batch_seq).to(device, non_blocking=pin_memory)
    return x, y, batch_length, batch_seq


def build_generator(name: str):
    return GENERATOR_NAMES[name]()

# -----------------------------------------------------------------------------
# Model / optimizer
# -----------------------------------------------------------------------------
iter_num = 0
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=vocab_size,
    dropout=dropout,
    pe_type=pe_type,
    gated=gated,
    trainable_freqs=trainable_freqs,
    pad_right=pad_right,
)
model = GPT(GPTConfig(**model_args)).to(device)
raw_model = model

scaler = torch.cuda.amp.GradScaler(enabled=(device_type == "cuda" and dtype == "float16"))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

if compile:
    print("Compiling model with torch.compile...")
    model = torch.compile(model)

# -----------------------------------------------------------------------------
# Metrics / scheduler
# -----------------------------------------------------------------------------
@torch.no_grad()
def estimate_accuracy(generator, length: int = train_max_length):
    model.eval()
    correct = torch.zeros((), device=device, dtype=torch.long)
    total = torch.zeros((), device=device, dtype=torch.long)
    losses = 0.0

    for _ in range(train_eval_iters):
        X, Y, batch_length, _ = get_batch(generator, length, probs, size=test_batch_size)
        logits, loss = model(X, Y, linebreak=batch_length.to(device))
        preds = logits.argmax(dim=-1)
        mask = Y != -1
        row_correct = ((preds == Y) | ~mask).all(dim=1)
        correct += row_correct.sum()
        total += row_correct.size(0)
        losses += loss.item()

    model.train()
    return (correct.float() / total.float()).item(), losses / train_eval_iters


def get_lr(it: int):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / max(1, lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def get_param_norm(model_to_measure):
    total = torch.zeros((), device=device)
    attn_norm = torch.zeros((), device=device)
    for name, param in model_to_measure.named_parameters():
        if param.requires_grad and param.dim() >= 2:
            value = param.data.pow(2).sum()
            total += value
            if "attn" in name:
                attn_norm += value
    return total.sqrt(), attn_norm.sqrt()


@torch.no_grad()
def estimate_each_batch_accuracy(test_length: int, generator, samples_per_interval: int = 100):
    model.eval()
    intervals = []
    start, end = 1, 25
    while start < test_length:
        intervals.append((start, min(end, test_length)))
        start = end + 1
        end *= 2

    acc_grouped = torch.zeros(len(intervals), device=device)
    for i, (min_length, max_length) in enumerate(intervals):
        correct = torch.zeros((), device=device)
        total = torch.zeros((), device=device)
        num_batches = math.ceil(samples_per_interval / test_batch_size)

        for _ in range(num_batches):
            X, Y, batch_length, _ = get_batch(
                generator,
                test_length,
                probs,
                fixlength=True,
                min_len=min_length,
                max_len=max_length,
                size=test_batch_size,
            )
            logits, _ = model(X, Y, linebreak=batch_length.to(device))
            preds = logits.argmax(dim=-1)
            valid = Y != -1
            row_correct = ((preds == Y) | ~valid).all(dim=1)
            correct += row_correct.sum()
            total += row_correct.size(0)

        acc_grouped[i] = correct.float() / total.float()

    model.train()
    return acc_grouped, intervals


def to_clean_float_list(value: Any) -> list[float]:
    tensor = torch.as_tensor(value, dtype=torch.float32).view(-1)
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0)
    return [float(v) for v in tensor.cpu().tolist()]


def save_checkpoint(name: str, current_acc: float):
    checkpoint = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": model_args,
        "iter_num": iter_num,
        "current_acc": current_acc,
        "config": config,
    }
    path = Path(out_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving checkpoint to {path}")
    torch.save(checkpoint, path)

# -----------------------------------------------------------------------------
# Optional logging
# -----------------------------------------------------------------------------
swanlab = None
if swanlab_log:
    try:
        import swanlab as _swanlab
        swanlab = _swanlab
        if swanlab_run_name is None:
            swanlab_run_name = (
                f"{pe_type}-L{n_layer}-H{n_head}-train_{train_type}-"
                f"seed{seed}-lr{learning_rate}-pad{pad_left}_{pad_right}"
            )
        swanlab.init(project=swanlab_project, name=swanlab_run_name, config=config, auto_sync_code=True)
    except Exception as exc:
        raise RuntimeError(
            "swanlab_log=True but SwanLab could not be initialized. "
            "Install swanlab and set SWANLAB_API_KEY, or set swanlab_log=False."
        ) from exc

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
train_generator = build_generator(train_type)
X, Y, batch_before, batch_seq = get_batch(train_generator, train_max_length, probs, size=train_batch_size)

t0 = time.time()
best_eval_acc = -1.0
last_acc = 0.0
last_loss = float("nan")

while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    if iter_num % eval_interval == 0:
        norm, attn_norm = get_param_norm(raw_model)
        train_acc, train_loss = estimate_accuracy(train_generator, length=train_max_length)
        ood_acc, _ = estimate_accuracy(train_generator, length=test_max_length)
        last_acc, last_loss = train_acc, train_loss

        print(
            f"step {iter_num}: train_acc={train_acc:.4f}, "
            f"ood_acc={ood_acc:.4f}, loss={train_loss:.8f}, "
            f"norm={norm.item():.2f}, attn_norm={attn_norm.item():.2f}, lr={lr:.2e}"
        )

        if swanlab is not None:
            swanlab.log({
                "iter": iter_num,
                "train/acc": train_acc,
                "ood/acc": ood_acc,
                "train/loss": train_loss,
                "lr": lr,
                "norm/total": norm.item(),
                "norm/attn": attn_norm.item(),
            }, step=iter_num)

        if always_save_checkpoint and ood_acc > best_eval_acc:
            best_eval_acc = ood_acc
            save_checkpoint("ckpt_best.pt", current_acc=ood_acc)

    if eval_only:
        break

    X, Y, batch_length, _ = get_batch(train_generator, train_max_length, probs, noise=True, size=train_batch_size)
    with ctx:
        logits, loss = model(X, Y, linebreak=batch_length.to(device))

    scaler.scale(loss).backward()
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    if iter_num % log_interval == 0:
        dt = time.time() - t0
        t0 = time.time()
        print(f"iter {iter_num}: loss={loss.item():.8f}, last_train_acc={last_acc:.4f}, time={dt * 1000:.2f}ms")

    iter_num += 1
    if iter_num > max_iters:
        break

if save_last:
    save_checkpoint("ckpt_last.pt", current_acc=last_acc)

# -----------------------------------------------------------------------------
# Final length-generalization evaluation
# -----------------------------------------------------------------------------
if final_eval:
    final_results = {}
    intervals = None
    for eval_name, generator in [
        ("unbalanced", UnbalancedGenerator()),
        ("random", RandomGenerator()),
        ("vfk", vfkGenerator()),
    ]:
        acc_tensor, intervals = estimate_each_batch_accuracy(
            last_test_length,
            generator,
            samples_per_interval=final_eval_samples_per_interval,
        )
        final_results[eval_name] = to_clean_float_list(acc_tensor)

    assert intervals is not None
    out = {
        "pe_type": pe_type,
        "pe_name": PE_NAMES[pe_type],
        "train_type": train_type,
        "seed": seed,
        "intervals": intervals,
        "samples_per_interval": final_eval_samples_per_interval,
        **final_results,
        "config": config,
    }

    result_dir = Path(out_dir) / "acc_logs"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"acc_seed_{seed}.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved final accuracy log to {result_path}")

    if swanlab is not None:
        for (start_len, end_len), a_unbal, a_rand, a_vfk in zip(
            intervals,
            final_results["unbalanced"],
            final_results["random"],
            final_results["vfk"],
        ):
            swanlab.log({
                "acc_vs_len/unbalanced": a_unbal,
                "acc_vs_len/random": a_rand,
                "acc_vs_len/vfk": a_vfk,
            }, step=end_len)
