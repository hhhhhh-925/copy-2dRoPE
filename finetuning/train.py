import torch
import importlib
import inspect
import os
from types import MethodType
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from pathlib import Path
from datetime import datetime
from utils.data.builder import build_dataset
from utils.args import Args
import torch._dynamo
torch._dynamo.config.cache_size_limit = 256  # or higher


def load_model(
    args: Args,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    print(f"Loading model from {args.pretrained_path}")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_distributed = world_size > 1

    model_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        output_loading_info=True,
    )
    # device_map="auto" is model sharding (single-process) and conflicts with DDP.
    # For multi-process launches (torchrun), keep one full replica per rank.
    if not use_distributed:
        model_kwargs["device_map"] = "auto"

    model, loading_info = AutoModelForCausalLM.from_pretrained(
        args.pretrained_path,
        **model_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_path,
        padding_side="right",
        padding=True,
        truncation=True,
    )
    # Set pad token if it's not already defined (many LLMs use eos_token as pad token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # model = model.to(torch.float32)

    # Debug/compatibility guard:
    # Some checkpoints define optional cut_cross_entropy. Disable it explicitly when
    # users request pure PyTorch CE or when multi-GPU sharding is detected.
    model_module = importlib.import_module(model.__class__.__module__)
    linear_ce_enabled = getattr(model_module, "linear_cross_entropy", None) is not None
    if linear_ce_enabled and args.torch_ce_only:
        print("Forcing PyTorch CE: disabling cut_cross_entropy via --torch_ce_only.")
        model_module.linear_cross_entropy = None
    else:
        device_map = getattr(model, "hf_device_map", None)
        if isinstance(device_map, dict):
            used_devices = set(device_map.values())
            if len(used_devices) > 1 and linear_ce_enabled:
                print(
                    "Detected multi-GPU sharded model; disabling cut_cross_entropy "
                    "for stable cross-device loss computation."
                )
                model_module.linear_cross_entropy = None

    return model, tokenizer


def should_trainer_enable_gradient_checkpointing(args: Args) -> bool:
    """Gradient checkpointing is enabled manually to support custom model APIs."""
    if args.gradient_checkpointing:
        return False
    return False


@torch.no_grad()
def test_generation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str = "The capital of China is",
    max_new_tokens: int = 20,
    do_sample: bool = False,
):
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    model.eval()
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
    )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text


def main():
    args = Args().parse_args()
    print(f"Arguments: {args}")

    rank = int(os.environ.get("RANK", "0"))
    is_main_process = rank == 0

    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    if is_main_process:
        args.save(output_dir / "args.json")
    # run_name = args.run_name + "-" + datetime.now().strftime("%Y%m%d_%H%M%S")
    # shorter tag for easier browsing of checkpoints, e.g. "0715" instead of "2024-07-15_14-30-00"
    date_tag = datetime.now().strftime("%m%d")
    run_name = f"{args.run_name}-{date_tag}"

    model, tokenizer = load_model(args)
    if args.liger_fused_ce:
        print("Enabling Liger fused linear cross entropy for Qwen3")
        from liger_kernel.transformers.model.qwen3 import lce_forward as qwen3_liger_lce_forward

        model.forward = MethodType(qwen3_liger_lce_forward, model)
    if args.gradient_checkpointing:
        print("Enabling gradient checkpointing")
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    print("===== model =====")
    print(model)
    print('=================')
    if hasattr(model.model, "id_to_newline_count"):
        print("before:", model.model.id_to_newline_count.sum().item())
        model.prepare_2dpos_with_tokenizer(tokenizer)
        print("after :", model.model.id_to_newline_count.sum().item())
    else:
        print("model does not have id_to_newline_count")
    # print("Performing test generation for sanity check")
    # output_text = test_generation(model, tokenizer)
    # print(f"Output: {output_text}")

    train_dataset, eval_dataset = build_dataset(args, tokenizer)

    print("Preparing training arguments")
    training_arg_values = dict(
        output_dir=output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,  # Regularization to prevent overfitting
        logging_dir=output_dir / "logs",
        logging_strategy="steps",
        logging_steps=1,  # Log metrics every step
        eval_strategy="steps",  # or "epoch"
        eval_steps=args.eval_interval,               # if using "steps"
        save_strategy="steps",  # Save model checkpoint at the end of each epoch
        save_steps=args.eval_interval,
        load_best_model_at_end=True,  # Load the best model (by validation loss) at the end of training
        metric_for_best_model="eval_loss",
        report_to=args.report_to,
        lr_scheduler_type="cosine",  # Cosine scheduler
        warmup_ratio=0.03,  # 10% of training steps for warmup
        # save_total_limit=3,  # Keep only the 3 most recent checkpoints to save disk space
        seed=0,  # Reproducibility
        bf16=args.bf16,
        gradient_checkpointing=should_trainer_enable_gradient_checkpointing(args),
        run_name=run_name,
        save_safetensors=False,
        dataloader_drop_last=True,
        adam_beta2=0.95,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        ddp_find_unused_parameters=False,
    )
    training_arg_params = inspect.signature(TrainingArguments.__init__).parameters
    unsupported_args = sorted(set(training_arg_values) - set(training_arg_params))
    if unsupported_args:
        print(f"Skipping unsupported TrainingArguments: {unsupported_args}")
    training_args = TrainingArguments(
        **{
            key: value
            for key, value in training_arg_values.items()
            if key in training_arg_params
        }
    )

    print("Initializing and running trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Start finetuning
    print("========== Start finetuning ==========")
    trainer.evaluate()
    trainer.train()

    print("========== Finetuning complete ==========")

    print("last 20 logs:")
    for x in trainer.state.log_history[-20:]:
        print(x)

    # Save the final finetuned model and tokenizer
    print("Finetuning complete. Saving final model...")
    ckpt_path = output_dir / "final"
    trainer.save_model(ckpt_path)
    tokenizer.save_pretrained(ckpt_path)
    print(f"Final model saved to {ckpt_path}")


if __name__ == "__main__":
    main()
