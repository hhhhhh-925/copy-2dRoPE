from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer
import json
import sys
from ..args import Args


def load_jsonl(file_path: str) -> list:
    return [json.loads(line) for line in open(file_path)]


def _has_explicit_data_path_arg() -> bool:
    return any(arg == "--data_path" or arg.startswith("--data_path=") for arg in sys.argv[1:])


def build_dataset(
    args: Args,
    tokenizer: AutoTokenizer,
) -> tuple[Dataset, Dataset]:
    """
    Build train and eval datasets for Tulu v3 SFT training.
    Tulu v3 uses a conversational format with 'messages' containing roles.
    """
    data_path = 'data/tulu-3-sft-mixture'
    print("Loading dataset from {data_path}")
    tulu_dataset = load_dataset(data_path)['train']

    def format_message(role: str, content: str) -> str:
        """Format a single message with role tags."""
        return f"### {role}:\n{content}\n\n"
    
    def preprocess(example: dict) -> dict[str, list[int]]:
        """
        Preprocess a single example from Tulu v3 dataset.
        
        Args:
            example: A dict containing 'messages' key with list of message dicts.
                     Each message has 'role' (e.g., 'user', 'assistant') and 'content'.
        
        Returns:
            Dict with 'input_ids', 'labels', and 'attention_mask'.
            Labels are masked (-100) for user/system messages, and kept for assistant messages.
        """
        messages = example['messages']
        
        input_ids = []
        labels = []
        
        # Process each message turn by turn
        for message in messages:
            role = message['role']
            content = message['content']
            
            # Format the message with role tags
            formatted_message = format_message(role, content)
            
            # Tokenize this message
            message_ids = tokenizer(
                formatted_message,
                add_special_tokens=False,
            )["input_ids"]
            
            # Add to input_ids
            input_ids.extend(message_ids)
            
            # Mask user and system messages, keep assistant messages
            if role == 'assistant':
                # Keep these tokens for training (compute loss)
                labels.extend(message_ids)
            else:
                # Mask user/system messages (don't compute loss)
                labels.extend([-100] * len(message_ids))
        
        # Add BOS token at the beginning if tokenizer has one
        if tokenizer.bos_token_id is not None:
            input_ids = [tokenizer.bos_token_id] + input_ids
            labels = [-100] + labels  # Don't compute loss on BOS
        
        # Add EOS token at the end
        if tokenizer.eos_token_id is not None:
            input_ids = input_ids + [tokenizer.eos_token_id]
            labels = labels + [tokenizer.eos_token_id]  # Compute loss on EOS
        
        # Verify lengths match
        assert len(labels) == len(input_ids), (
            f"Length mismatch: labels={len(labels)}, input_ids={len(input_ids)}"
        )
        
        attention_mask = [1] * len(input_ids)
        
        # Truncation
        input_ids = input_ids[:args.max_len]
        labels = labels[:args.max_len]
        attention_mask = attention_mask[:args.max_len]
        
        # Add padding
        pad_len = args.max_len - len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
        labels = labels + [-100] * pad_len
        attention_mask = attention_mask + [0] * pad_len
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    copy_data_path = (
        args.data_path
        if _has_explicit_data_path_arg()
        else 'data/copy-sft/train_10_1000_unbal.jsonl'
    )
    copy_dataset = Dataset.from_list(load_jsonl(copy_data_path))
    copy_dataset = copy_dataset.shuffle(seed=0)
    # 5% of the dataset is copy data
    n_examples = args.n_train_examples + args.n_eval_examples
    copy_dataset = copy_dataset.take(int(n_examples * args.copy_prop))

    tulu_dataset = tulu_dataset.shuffle(seed=0)
    tulu_dataset = tulu_dataset.take(int(n_examples * (1 - args.copy_prop)))
    dataset = concatenate_datasets([tulu_dataset, copy_dataset])

    # Split into train and validation sets
    dataset = dataset.train_test_split(
        test_size=0.1,  # 10% for validation
        seed=0,
    )

    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    train_dataset = train_dataset.take(min(args.n_train_examples, len(train_dataset)))
    eval_dataset = eval_dataset.take(min(args.n_eval_examples, len(eval_dataset)))

    print(f"Preprocessing train dataset")
    train_dataset = train_dataset.map(
        preprocess,
        remove_columns=train_dataset.column_names,
        num_proc=4,
        load_from_cache_file=False,
    )

    print(f"Preprocessing eval dataset")
    eval_dataset = eval_dataset.map(
        preprocess,
        remove_columns=eval_dataset.column_names,
        num_proc=4,
        load_from_cache_file=False,
    )
    
    return train_dataset, eval_dataset

