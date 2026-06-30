# Pretraining Guide

This folder contains the training code used for pretraining experiments in this repo.
The guide below is written for open-source users running on different hardware (single GPU, multi-GPU, CPU-only debug, local workstation, or cluster).

## 1) What is in this folder

- `train.py`: main training entrypoint.
- `arguments.py`: CLI arguments (all runtime flags).
- `preparation.py`: config loading, dataloaders, accelerator setup, optimizer/scheduler setup.
- `data/`: dataset builders used by `get_data(...)` in training.
- `configs/model/*.json`: model architecture configs.
- `configs/training/*.json`: training/data/optimizer configs.
- `configs/accelerate/*.yaml`: distributed launch configs (multi-GPU, FSDP, DeepSpeed).
- `trainer/trainer.py`: training loop and checkpointing.
- `build_hf_ckpt.py`: convert saved checkpoints into Hugging Face format.

## 2) Current repository assumptions (important)

Before running, be aware of these assumptions in the current code:

- Most distributed configs assume NVIDIA GPUs + CUDA + NCCL.
- Paths in training configs are often absolute machine-local paths and must be edited for your environment.
- Run commands from the `pretraining/` directory so relative config paths resolve correctly.

## 3) Environment setup (portable)

Tested stack for this codebase:

- Python: `3.12.12`
- PyTorch: `2.9.1+cu128`
- flash-attn: `2.8.3`
- CUDA UMD: `13.3`

### Option A: venv (works on Linux/macOS)

```bash
cd pretraining
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### Option B: conda

```bash
cd pretraining
conda create -n 2drope-pretrain python=3.10 -y
conda activate 2drope-pretrain
python -m pip install --upgrade pip setuptools wheel
```

### Install Python dependencies

Install from the provided requirements file:

```bash
pip install -r requirements.txt
```

## 4) Prepare your configs

Two config files are required:

- **Model config** (`--model_config`): architecture (`hidden_size`, `num_hidden_layers`, `pos_emb_type`, etc.).
- **Training config** (`--train_config`): data path/name, batching, LR schedule, steps, logging/checkpoint intervals.

Examples in this repo:

- Model config: `configs/model/gpt-350m-2drope-x1000-y1000.json`
- Training config: `configs/training/dclm-7b.json`

You should usually copy/edit configs for your own machine and dataset paths.

### Dataset configuration (`data_name` and `data_path`)

Dataset loading is now implemented under `pretraining/data/` and routed through `data.get_data(...)`.

- `data_name` selects the dataset builder.
- `data_path` points to the dataset location for builders that require a local path.

In the current training configs under `configs/training/`, the default is `data_name: "dclm"`.
For `dclm`, `data_path` should point to your DCLM dataset root. The loader expects shard files such as `*.jsonl.zst` under that root (especially when using reverse shard order).

For validation, set:

- `validation_data_name`
- `validation_data_path`

and optionally `n_validation_examples`.

The `dclm-*.json` configs now use dummy placeholders:

- `"/path/to/datasets/dclm-baseline-1.0"`
- `"/path/to/datasets/fineweb-edu-100bt"`

Before training, download DCLM and FineWeb-Edu from Hugging Face to your machine, then replace those placeholder paths with your local dataset directories in the selected training config.

## 5) Running experiments

### Important: GPU count and batch size assumptions

The default setup in this repo assumes **8 GPUs**:

- `train.sh` requests 8 GPUs in SLURM (`#SBATCH --gres=gpu:8`) and defaults `n_gpus=8`.
- Most training configs under `configs/training/` were tuned with 8-GPU runs in mind.
- `batch_size` in training configs is **per-device (per-GPU) batch size**, not global batch size.

If you run with a different GPU count, you should update:

- SLURM resource requests in `train.sh` (especially `--gres=gpu:<N>`).
- Launch process count (`--num_processes` / `n_gpus`) so it matches visible GPUs.
- `batch_size` (and optionally `grad_accum_steps`) to keep global tokens/batch in a reasonable range.

Rule of thumb:

- Global batch is approximately `num_gpus * batch_size * grad_accum_steps`.
- If you reduce GPU count, either reduce total throughput expectations or increase `batch_size` / `grad_accum_steps` carefully (within memory limits).

### A. Single-process smoke test (portable debug)

Use this first to validate environment/config wiring:

```bash
cd pretraining
accelerate launch --cpu --num_processes=1 train.py \
  --model_name=gpt \
  --model_config=configs/model/gpt/2-256.json \
  --train_config=configs/training/dclm-7b.json \
  --proj_name=pretrain-debug \
  --run_name=cpu-smoke \
  --compile=0
```

Notes:
- `--cpu` is slow but useful for verifying imports/configs.
- On Apple Silicon (`mps`), keep `--compile=0` (the code already asserts compile is unsupported on MPS).

### B. Multi-GPU (single machine, recommended for real training)

```bash
cd pretraining
accelerate launch \
  --config_file configs/accelerate/multigpu_config.yaml \
  --num_processes 8 \
  --main_process_port 29501 \
  train.py \
  --model_name=gpt \
  --model_config=configs/model/gpt-350m-2drope-x1000-y1000.json \
  --train_config=configs/training/dclm-7b.json \
  --proj_name=2drope \
  --run_name=gpt350m-2drope-dclm \
  --compile=1
```

Adjust:
- `--num_processes` to your visible GPU count.
- `--main_process_port` if the default port is occupied.
- `--compile=0` for unsupported model families or while debugging.

### C. FSDP / DeepSpeed profiles

Swap `--config_file`:

- FSDP: `configs/accelerate/fsdp_config.yaml`
- DeepSpeed ZeRO-3: `configs/accelerate/ds_config.yaml`

These profiles are hardware-sensitive. Start with a small model and short run before long jobs.

## 6) Logging and outputs

- Output directory pattern: `results/<proj_name>/<run_name>/`
- The script saves:
  - `args.json`
  - `model_config.json`
  - checkpoints like `ckpt_<step>/`
- Logging backends are controlled by `--report_to` through HuggingFace Accelerate.
- Current default behavior is SwanLab-enabled (and `arguments.py` currently defaults to `tensorboard,swanlab`).

To log to SwanLab, set your API key before launch:

```bash
export SWANLAB_API_KEY=<your_key>
```

`train.sh` already shows this pattern with `SWANLAB_API_KEY`.

Because training is driven by HuggingFace Accelerate, changing trackers is straightforward (for example, to TensorBoard-only). You should update your run configuration accordingly (for example via `--report_to=tensorboard`), or change the default in `arguments.py` if you want a permanent default tracker.

If you do not want online logging, set:

```bash
export WANDB_MODE=offline
```

## 7) Resume training / load weights

### Resume full training state (model + optimizer + scheduler)

```bash
accelerate launch ... train.py \
  ... \
  --resume_path=results/<proj>/<run>/ckpt_<step> \
  --resume_step=<step>
```

### Load only pretrained model weights

```bash
accelerate launch ... train.py \
  ... \
  --pretrained_path=/path/to/model_weights.pt
```

Do not set both `--resume_path` and `--pretrained_path` together.

## 8) Convert checkpoints to Hugging Face format

```bash
cd pretraining
python build_hf_ckpt.py \
  --ckpt_path results/<proj>/<run> \
  --ckpt ckpt_<step> \
  --tok_path tokenizer/llama2 \
  --out_path ckpts/<export-name>
```

## 9) Cluster usage notes

- The included `train.sh` is cluster-specific (`#SBATCH` directives) and not portable as-is.
- It also contains hardcoded experiment defaults; treat it as an example template only.
- For SLURM/PBS/K8s, wrap the `accelerate launch ... train.py ...` command inside your scheduler job script.

## 10) Troubleshooting

- **`File not found` for configs**: run from `pretraining/` or pass absolute config paths.
- **Dataset loading/import errors**: make sure you are running from `pretraining/`, and that `data_name`/`data_path` in your training config match a supported builder and a valid dataset location.
- **OOM / CUDA out of memory**:
  - lower `batch_size`
  - lower `max_len`
  - increase `grad_accum_steps`
  - reduce model size
  - disable `compile` for debugging
- **NCCL/distributed startup issues**:
  - verify one process per GPU
  - try a different `--main_process_port`
  - ensure CUDA/NCCL/PyTorch versions are compatible
- **No logs in WandB/SwanLab**:
  - check `--report_to`
  - check env vars/tokens
  - use `WANDB_MODE=offline` for disconnected machines

## 11) Recommended first run checklist

1. Create environment and install dependencies.
2. Verify `python -c "import torch; print(torch.__version__)"`.
3. Confirm GPU visibility (`nvidia-smi`) if using CUDA.
4. Edit one training config with valid local `data_path`.
5. Run a short smoke test (`--cpu` or 1 GPU, very small model).
6. Scale to multi-GPU only after smoke test is stable.
