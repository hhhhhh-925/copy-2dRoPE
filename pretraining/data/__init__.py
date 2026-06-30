from typing import Optional

from datasets import IterableDataset

from . import slimpj
from . import slimpj_200k
from . import wikitext
from . import openwebtext
from . import fineweb_edu_100bt
from . import dclm
from . import dclm_2d


def get_data(
    tokenizer,
    data_name: str = "slimpj",
    data_path: Optional[str] = None,
    max_len: int = 512,
    shift_labels: bool = False,
    is_seq2seq: bool = False,
    **kwargs,
) -> IterableDataset:
    '''
    Will return an IterableDataset
    '''
    if data_name == "slimpj":
        assert data_path is not None
        return slimpj.build_dataset(
            tokenizer,
            data_dir=data_path,
            max_len=max_len,
            shift_labels=shift_labels,
            **kwargs,
        )  # type: ignore
    elif data_name == "slimpj-200k":
        assert data_path is not None
        return slimpj_200k.build_dataset(
            tokenizer,
            data_dir=data_path,
            max_len=max_len,
            shift=shift_labels,
            **kwargs,
        )  # type: ignore
    elif data_name == "wikitext":
        return wikitext.build_dataset(
            tokenizer,
            max_len=max_len,
            shift=shift_labels,
            **kwargs,
        )  # type: ignore
    elif data_name == 'openwebtext':
        return openwebtext.build_dataset(
            tokenizer,
            max_len=max_len,
            shift=shift_labels,
            **kwargs,
        )  # type: ignore
    elif data_name == 'fineweb-edu-100bt':
        return fineweb_edu_100bt.build_dataset(
            tokenizer,
            data_dir=data_path,
            max_len=max_len,
            shift_labels=shift_labels,
            is_seq2seq=is_seq2seq,
            **kwargs,
        )
    elif data_name == "dclm":
        return dclm.build_dataset(
            tokenizer,
            data_dir=data_path,
            max_len=max_len,
            shift_labels=shift_labels,
            **kwargs,
        )
    elif data_name == "dclm_2d":
        return dclm_2d.build_dataset(
            tokenizer,
            data_dir=data_path,
            max_len=max_len,
            shift_labels=shift_labels,
            **kwargs,
        )

    else:
        raise ValueError(f"Unknown data name: {data_name}")
