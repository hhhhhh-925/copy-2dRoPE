from transformers import AutoTokenizer
from datasets import Dataset
from ..args import Args
import sys


class ImplementationError(NotImplementedError):
    """Raised when a deprecated dataset entrypoint is requested."""


def _has_explicit_data_path_arg() -> bool:
    """Return True when data_path was explicitly passed on CLI."""
    for arg in sys.argv[1:]:
        if arg == "--data_path" or arg.startswith("--data_path="):
            return True
    return False


def build_dataset(
    args: Args,
    tokenizer: AutoTokenizer,
) -> tuple[Dataset, Dataset]:
    if args.data_name == 'sft-dataset':
        from .sft_dataset import build_dataset as build_sft_dataset
        return build_sft_dataset(args, tokenizer)

    raise ImplementationError(
        f"data_name={args.data_name} is not supported. "
        "Only data_name=sft-dataset is currently supported."
    )
