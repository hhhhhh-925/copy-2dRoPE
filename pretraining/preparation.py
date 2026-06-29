from pathlib import Path
import json
from typing import Tuple
from datetime import datetime

import datasets
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from accelerate import DistributedDataParallelKwargs
from torch.optim import AdamW
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup
from optim.lr_scheduler import WSDScheduler

from arguments import Args
from data import get_data

import torch

def load_train_config(args: Args):
    '''
    Load from `args.train_config` if it exists, and use the values
    when the argument is not provided.
    '''
    if args.train_config is not None and Path(args.train_config).exists():
        config = json.load(open(args.train_config, "r"))
        for k, v in config.items():
            if getattr(args, k) is None:
                setattr(args, k, v)
    else:
        print(f"WARNING: train config {args.train_config} does not exist.")
        print("It is highly recommended to provide a train config.")


def get_accelerator(args: Args, find_unused_parameters: bool = False) -> Accelerator:
    assert args.grad_accum_steps is not None
    assert args.run_name is not None
    assert args.proj_name is not None

    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=find_unused_parameters,
    )

    project_dir = f"./results/{args.proj_name}"
    log_with = args.report_to.split(",")
    accelerator = Accelerator(
        project_dir=project_dir,
        log_with=log_with,  # type: ignore
        gradient_accumulation_steps=args.grad_accum_steps,
        # This argument allows us to step the LR scheduler manually.
        # This is needed because internally, accelerator expects the
        # LR scheduler to step more frequently when using multiple
        # processes or when using gradient accumulation (e.g.,
        # the warmup steps should be scaled by
        # num_processes * gradient_accumulation).
        step_scheduler_with_optimizer=False,
        # kwargs_handlers=[ddp_kwargs],
    )
    accelerator.print(f"Project directory: {project_dir}")
    accelerator.print(f"Reporting to: {log_with}")
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y%m%d-%H%M%S")
    hps = args.as_dict()
    run_name = f"{args.run_name}_{formatted_time}"
    accelerator.init_trackers(
        project_name=args.proj_name,
        config=hps,
        init_kwargs={
            "wandb": {
                "name": run_name,
            },
            "swanlab": {
                "experiment_name": run_name,
            }
        },
    )
    return accelerator

def cu_seqlens_collate_fn(batch):
    lens = torch.tensor([it["input_ids"].size(0) for it in batch], dtype=torch.int32)

    cu = torch.empty(lens.numel() + 1, dtype=torch.int32)
    cu[0] = 0
    torch.cumsum(lens, dim=0, out=cu[1:])

    input_ids = torch.cat([it["input_ids"] for it in batch], dim=0).to(dtype=torch.long)
    labels    = torch.cat([it["labels"]    for it in batch], dim=0).to(dtype=torch.long)

    if input_ids.numel() == 0:
        raise ValueError("input_ids is empty after concatenation, unexpected.")

    return {
        "input_ids": input_ids.view(1, -1),   # (total_T,)
        "labels": labels.view(1, -1),         # (total_T,)
        "cu_seqlens": cu.view(1, -1),         # (B+1,)
    }

def get_dataloaders(
    args: Args, tokenizer: PreTrainedTokenizerBase, accelerator: Accelerator
) -> Tuple[DataLoader, DataLoader | None]:
    assert args.data_name is not None
    assert args.data_path is not None
    assert args.max_len is not None
    assert args.batch_size is not None

    train_data_kw = {}
    if args.data_name == "dclm":
        train_data_kw["reverse_shard_order"] = bool(args.reverse_dataset_order)

    train_ds = get_data(
        tokenizer=tokenizer,
        data_name=args.data_name,
        data_path=args.data_path,
        max_len=args.max_len,
        is_main_process=accelerator.is_main_process,
        shift_labels=not bool(args.shift_labels_in_model),
        is_seq2seq=bool(args.is_seq2seq),
        **train_data_kw,
    )  # type: ignore

    train_loader = DataLoader(
        train_ds,  # type: ignore
        batch_size=args.batch_size,
        collate_fn=cu_seqlens_collate_fn if args.use_cu_seqlens else None,
    )
    if args.validation_data_path is not None and args.validation_data_name is not None:
        val_ds = get_data(
            tokenizer=tokenizer,
            data_name=args.validation_data_name,
            data_path=args.validation_data_path,
            max_len=args.max_len,
            is_main_process=accelerator.is_main_process,
            shift_labels=not bool(args.shift_labels_in_model),
        )  # type: ignore
        # val_ds = split_dataset_by_node(val_ds, rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))
        if args.n_validation_examples is not None:
            if isinstance(val_ds, datasets.IterableDataset):
                val_ds = val_ds.take(args.n_validation_examples)
            else:
                val_ds = val_ds.select(range(args.n_validation_examples))
        val_loader = DataLoader(
            val_ds,  # type: ignore
            batch_size=args.batch_size * 4,
            collate_fn=cu_seqlens_collate_fn if args.use_cu_seqlens else None,
        )
    else:
        val_loader = None

    return train_loader, val_loader


def get_args() -> Args:
    args = Args().parse_args()

    if args.compile == 1:
        if args.grad_ckpt == 1:
            print("Cannot use grad checkpoint and compile mode together, setting compile to 0.")
            args.compile = 0

        if args.model_name in ['gated_deltanet', 'gated-deltanet']:
            print("Gated DeltaNet does not support torch.compile, setting compile to 0...")
            args.compile = 0

        if args.model_name in ['rabbit']:
            print("Rabbit does not support compilation, turning it off...")
            args.compile = 0

        if args.model_name in ['mamba2']:
            print("Turning off compilation for Mamba2")
            args.compile = 0

    return args


def prepare_optimizers(model: nn.Module, args: Args):
    assert args.lr is not None
    assert args.beta1 is not None
    assert args.beta2 is not None
    assert args.weight_decay is not None
    assert args.n_warmup_steps is not None
    assert args.n_train_steps is not None
    assert args.lr_scheduler is not None

    print("Preparing optimizers")
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    if args.lr_scheduler == 'wsd':
        assert args.n_drop_steps is not None
        lr_scheduler = WSDScheduler(
            optimizer=optimizer,
            lr=args.lr,
            n_decay_iters=args.n_drop_steps,
            n_warmup_iters=args.n_warmup_steps,
            n_train_iters=args.n_train_steps,
            min_lr=args.min_lr,
        )
    elif args.lr_scheduler == 'cosine':
        lr_scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=args.n_warmup_steps,
            num_training_steps=args.n_train_steps,
            min_lr=args.min_lr,
        )
    else:
        raise ValueError(f"Invalid LR scheduler: {args.lr_scheduler}")
    return optimizer, lr_scheduler
