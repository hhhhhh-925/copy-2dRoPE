# Finetuning Guide

This folder contains supervised finetuning (SFT) code for HuggingFace checkpoints.

## 1) Tested environment

This finetuning code is tested on the same environment as `pretraining/`:

- Python: `3.12.12`
- PyTorch: `2.9.1+cu128`
- flash-attn: `2.8.3`
- CUDA UMD: `13.3`

## 2) What is in this folder

- `train.py`: main finetuning entrypoint.
- `train.sh`: launcher script (single GPU or multi-GPU with `accelerate`).
- `utils/args.py`: CLI arguments and defaults.
- `utils/data/builder.py`: dataset router (`data_name` selection).
- `utils/data/sft_dataset.py`: SFT dataset construction and preprocessing.

## 3) Install dependencies

Use the same environment/dependencies used by `pretraining/`, then run from this folder:

```bash
cd finetuning
```

## 4) How to start finetuning from a HuggingFace checkpoint

SFT jobs in this repo are expected to run on **one GPU** by default.

Minimal example:

```bash
cd finetuning
bash train.sh \
  pretrained_path=/path/to/hf-checkpoint \
  output_dir=results \
  run_name=my-sft-run \
  data_name=sft-dataset \
  data_path=/path/to/copy_sft.jsonl
```

Notes:

- `pretrained_path` should point to a HuggingFace-compatible checkpoint directory.
- `train.sh` auto-launches:
  - multi-GPU: `accelerate launch --num_processes=<num_gpus> train.py`
  - single GPU/CPU process: `python train.py`
- Default behavior is effectively single-GPU usage (set `num_gpus=1` explicitly for clarity).
- You can override GPU count with `num_gpus=<N>` in `train.sh` arguments.

## 5) Arguments to pass

Main arguments from `utils/args.py`:

- `pretrained_path`: checkpoint path to finetune from.
- `data_name`: currently only `sft-dataset` is supported.
- `data_path`: path to copy-SFT JSONL (see dataset notes below).
- `output_dir`: base output directory.
- `run_name`: experiment name.
- `batch_size`: **per-device** train/eval batch size.
- `gradient_accumulation_steps`: grad accumulation steps.
- `epochs`: training epochs.
- `max_steps`: if `> 0`, caps total train steps; `-1` means epoch-based.
- `lr`, `weight_decay`: optimizer hyperparameters.
- `max_len`: sequence length after truncation/padding.
- `eval_interval`: eval/save steps interval.
- `n_train_examples`, `n_eval_examples`: capped train/eval sizes.
- `copy_prop`: proportion of copy-data in the mixed dataset.
- `bf16`: enable bf16 training.
- `gradient_checkpointing`: enable model gradient checkpointing.
- `liger_fused_ce`: enable Liger fused CE path for supported models.
- `torch_ce_only`: force disable optional cut-cross-entropy path.
- `report_to`: tracker backend passed to HuggingFace `TrainingArguments` (default `tensorboard`).

## 6) Optional features supported

- **Distributed training via Accelerate**
  - Set `num_gpus=<N>` when calling `train.sh`.
- **Tracker backend switch**
  - Change with `report_to=...` (for example `tensorboard`, `swanlab`, `wandb`, or `none` depending on your installed integrations).
- **Optional SwanLab logging**
  - Set env var before launch:
  - `export SWANLAB_API_KEY=<your_key>`
- **Optional HF mirror endpoint**
  - Set env var if needed:
  - `export HF_ENDPOINT=<your_mirror_endpoint>`
- **2D position preparation**
  - If loaded model exposes `id_to_newline_count`, `train.py` calls `prepare_2dpos_with_tokenizer(...)` automatically.
- **Liger fused CE**
  - Enable with `liger_fused_ce=True` for compatible model/kernel setup.

## 7) Dataset behavior and current constraints

Current dataset pipeline in `utils/data/sft_dataset.py`:

- Loads Tulu mixture from `data/tulu-3-sft-mixture` (hardcoded in current implementation).
- Loads copy-SFT JSONL from:
  - `data_path` if explicitly passed on CLI, otherwise
  - `data/copy-sft/train_10_1000_unbal.jsonl`
- Mixes Tulu + copy data with ratio controlled by `copy_prop`.
- Preprocesses chat messages into causal-LM inputs:
  - user/system tokens masked with `-100`
  - assistant tokens contribute to loss
  - BOS/EOS handling, truncation to `max_len`, right padding

Because of this implementation, make sure the expected local dataset files are present (or pass `data_path` explicitly for copy data).

## 8) Example commands

Single GPU (recommended default):

```bash
bash train.sh \
  num_gpus=1 \
  pretrained_path=/path/to/hf-checkpoint \
  output_dir=results \
  run_name=single-gpu-test \
  report_to=tensorboard \
  bf16=True
```

Single GPU (explicit `num_gpus=1`):

```bash
bash train.sh \
  num_gpus=1 \
  pretrained_path=/path/to/hf-checkpoint \
  output_dir=results \
  run_name=single-gpu-run \
  batch_size=16 \
  gradient_accumulation_steps=1 \
  eval_interval=100 \
  report_to=tensorboard
```

Enable SwanLab:

```bash
export SWANLAB_API_KEY=<your_key>
bash train.sh \
  num_gpus=1 \
  pretrained_path=/path/to/hf-checkpoint \
  run_name=swanlab-run \
  report_to=swanlab
```

If you later scale to 8 GPUs, remember:

- `batch_size` is per-GPU, so global effective batch grows with GPU count.
- Approximate global batch:
  - `num_gpus * batch_size * gradient_accumulation_steps`
- To keep similar optimization dynamics as a 1-GPU run, reduce per-device `batch_size` and/or `gradient_accumulation_steps` accordingly when moving to 8 GPUs.

## 9) Output artifacts

`train.py` saves:

- run args JSON under `output_dir/run_name/args.json`
- trainer checkpoints according to `save_steps`
- final model/tokenizer under `output_dir/run_name/final/`

## 10) Troubleshooting

- **Model load fails**: verify `pretrained_path` is a valid HuggingFace checkpoint directory.
- **Dataset file errors**: verify local dataset paths and current `sft_dataset.py` assumptions.
- **OOM**: lower `batch_size` or `max_len`, increase `gradient_accumulation_steps`.
- **No tracker logs**: check `report_to` and related env vars (for example `SWANLAB_API_KEY`).
