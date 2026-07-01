# Synthetic Experiments

This repository trains small Transformer models on synthetic copy tasks and evaluates length generalization under several positional encodings.

## Positional encodings

Set `pe_type` in the config file to one of:

| `pe_type` | Name |
|---|---|
| `ada2d` | Auto-2D-RoPE |
| `2drope` | 2D-RoPE |
| `rope` | RoPE |
| `nope` | NoPE |
| `hrope` | Hybrid-RoPE |
| `alibi` | ALiBi |
| `nrope` | RNoPE |

## Repository layout

```text
.
├── train.py                 # training and evaluation entry point
├── configurator.py          # nanoGPT-style config override helper
├── model.py                 # GPT model and positional encodings
├── datagenerator.py         # synthetic copy-task generators
├── config.py                # experiment config, passed to train.py
└── out/                     # checkpoints and JSON accuracy logs
```

The code is intended to be run as:

```bash
python train.py config.py
```

The config file is a normal Python file. Any variable defined in `config.py` overrides the default value in `train.py`.

## Environment

Recommended environment:

- Python 3.10 or 3.11
- PyTorch 2.1 or newer
- NumPy
- CUDA-capable GPU for large-context experiments
- Optional: SwanLab, only if `swanlab_log = True`

Install the minimal dependencies:

```bash
pip install torch numpy
```

For CUDA, install the PyTorch build matching your CUDA version from the official PyTorch installation command. For optional SwanLab logging:

```bash
pip install swanlab
export SWANLAB_API_KEY="your_key_here"
```

Do not hard-code API keys in the repository.

## Quick start

Create `config.py`:

```python
out_dir = "out/copy_12l_ada2d"

# model
n_layer = 12
n_head = 12
head_dim = 128
n_embd = n_head * head_dim
block_size = 26000
pe_type = "ada2d"
trainable_freqs = False
gated = True

# data
train_type = "unbal"
probs = [0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95]
train_max_length = 100
test_max_length = 10000
last_test_length = 10000
pad_left = 2
pad_right = 4

# optimization
train_batch_size = 64
test_batch_size = 1
grad_clip = 1.0
weight_decay = 0.01
learning_rate = 1e-5
min_lr = 1e-6
max_iters = 10000
lr_decay_iters = 10000
warmup_iters = 100
beta2 = 0.95

# logging / saving
seed = 42
eval_interval = 500
log_interval = 500
train_eval_iters = 20
save_last = True
always_save_checkpoint = False
final_eval = True
final_eval_samples_per_interval = 10
swanlab_log = False

# system
device = "auto"      # auto, cuda, cuda:0, cpu, mps
dtype = "auto"       # auto, bfloat16, float16, float32
compile = False
```

Run:

```bash
python train.py config.py
```

The final length-generalization results are saved to:

```text
<out_dir>/acc_logs/acc_seed_<seed>.json
```

If `save_last = True`, the last checkpoint is saved to:

```text
<out_dir>/ckpt_last.pt
```

If `always_save_checkpoint = True`, the best checkpoint by OOD accuracy is saved to:

```text
<out_dir>/ckpt_best.pt
```

## Running on different operating systems

### Linux GPU server

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch numpy
python train.py config.py
```

To choose a GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py config.py
```

For long jobs, use `tmux`, `screen`, or your cluster scheduler, for example SLURM:

```bash
sbatch run_train.sh
```

A minimal SLURM script:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=copy_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs
python train.py config.py
```

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch numpy
python train.py config.py
```

If PowerShell blocks virtual-environment activation, run PowerShell as your user and execute:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

For serious GPU training on Windows, WSL2 or a Linux server is usually more stable than native Windows.

### macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch numpy
python train.py config.py
```

On Apple Silicon, set:

```python
device = "mps"
dtype = "float32"
compile = False
```

For debugging on CPU, set:

```python
device = "cpu"
dtype = "float32"
train_batch_size = 2
test_batch_size = 1
max_iters = 10
final_eval = False
```

## Common experiment variants

### Change positional encoding

```python
pe_type = "rope"   # or ada2d, 2drope, nope, hrope, alibi, nrope
```

### 12-layer setting

```python
n_layer = 12
n_head = 12
head_dim = 128
n_embd = n_head * head_dim
block_size = 26000
learning_rate = 1e-5
min_lr = 1e-6
max_iters = 10000
lr_decay_iters = 10000
```

### 1-layer setting

```python
n_layer = 1
n_head = 2
head_dim = 512
n_embd = n_head * head_dim
block_size = 260000
learning_rate = 5e-4
min_lr = 5e-5
max_iters = 60000
lr_decay_iters = 60000
```

## Output format

The final JSON file contains:

```json
{
  "pe_type": "ada2d",
  "pe_name": "Auto-2D-RoPE",
  "train_type": "unbal",
  "seed": 42,
  "intervals": [[1, 25], [26, 50], [51, 100]],
  "samples_per_interval": 10,
  "imbalanced": [1.0, 1.0, 0.9],
  "random": [1.0, 0.8, 0.4],
  "RepeatFlip": [1.0, 0.7, 0.2],
  "config": {}
}
```

Each accuracy is exact-sequence accuracy over the labeled copied region for that length interval.

## Notes for reproducibility

- Set `seed` explicitly.
- Keep `train_max_length`, `test_max_length`, `last_test_length`, `pad_left`, and `pad_right` in the config file.
- Keep `swanlab_log = False` for artifact-only reproduction; enable it only for online tracking.
- Use `dtype = "float32"` if you want the most conservative numerical behavior across platforms.
- Large values such as `block_size = 260000` require substantial GPU memory.
