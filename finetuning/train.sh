#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name='sft-2dr'

set -euo pipefail

# Optional mirror endpoint. Leave unset to use the default HuggingFace endpoint.
if [[ -n "${HF_ENDPOINT:-}" ]]; then
    export HF_ENDPOINT
fi

# Optional SwanLab credential for experiment tracking.
if [[ -n "${SWANLAB_API_KEY:-}" ]]; then
    export SWANLAB_API_KEY
fi
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${PWD}/.hf_modules/${SLURM_JOB_ID:-local}}"
mkdir -p "${HF_MODULES_CACHE}"

echo "STARTING TRAIN"

args="$@"

for arg in "$@"; do
    eval "$arg"
done

num_gpus="${num_gpus:-${SLURM_GPUS_ON_NODE:-8}}"

# Normalize num_gpus to an integer when possible.
if ! [[ "${num_gpus}" =~ ^[0-9]+$ ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        num_gpus="$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')"
    else
        num_gpus="1"
    fi
fi

if [[ "${num_gpus}" -gt 1 ]]; then
    echo "Launching distributed training with accelerate on ${num_gpus} GPUs"
    cmd="accelerate launch --num_processes=${num_gpus} train.py"
else
    echo "Launching single-process training"
    cmd="python train.py"
fi

# Add command-line arguments to the cmd string
for arg in "$@"; do
    key="${arg%%=*}"
    if [[ "${key}" == "num_gpus" ]]; then
        continue
    fi
    if [[ "${arg}" == *"="* ]]; then
        value="${arg#*=}"
        if [[ "${value}" == "True" || "${value}" == "true" ]]; then
            cmd+=" --${key}"
        elif [[ "${value}" == "False" || "${value}" == "false" ]]; then
            continue
        else
            cmd+=" --$arg"
        fi
    else
        cmd+=" --$arg"
    fi
done

echo "======== Final command ========"
echo "$cmd" | tr ' ' '\n'
echo "==============================="

nvidia-smi || echo "WARNING: nvidia-smi failed; continuing to Python startup"

$cmd

echo "TRAINING COMPLETE"
