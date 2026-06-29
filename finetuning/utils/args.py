from tap import Tap


class Args(Tap):
    data_name: str = "sft-dataset"
    data_path: str = "/path/to/datasets/binary-copy/train.jsonl"
    pretrained_path: str = "/path/to/checkpoints/gpt-730m-rope-dclm-15b/ckpt_30000"
    output_dir: str = "results-2026-2"
    # batch_size: int = 32
    # gradient_accumulation_steps: int = 2
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    epochs: int = 1
    lr: float = 2e-5
    max_len: int = 4096
    weight_decay: float = 0.01
    dropout: float = 0.05
    run_name: str = "730m-rope"
    overwrite_output_dir: bool = True
    eval_interval: int = 100
    n_train_examples: int = 65536
    n_eval_examples: int = 1024
    copy_prop: float = 0.05
    max_steps: int = -1  # -1 = no limit (train by epochs)
    bf16: bool = False
    gradient_checkpointing: bool = False
    liger_fused_ce: bool = False
    torch_ce_only: bool = False
    report_to: str = "tensorboard"
