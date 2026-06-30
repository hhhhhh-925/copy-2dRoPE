from typing import Optional, List, Dict

import torch

from datasets import load_dataset


def build_dataset(
    tokenizer,
    streaming: bool = True,
    n_workers: int = 8,
    overwrite_cache: bool = False,
    token_ids_only: bool = True,
    max_len: int = 512,
    eos_token_id: Optional[int] = None,
    do_log: bool = False,
    **kwargs,
):
    '''
    Returns an iterable of batches of token IDs.

    This will use `load_dataset` from the HuggingFace Datasets library to load the
    data from `data_dir`, tokenize each example, concatenate the input IDs, add an
    EOS token ID at the end of each sequence, then split into chunks of `max_len`
    tokens, and return a tensor of (batch_size, max_len).
    '''

    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    raw_dataset = load_dataset(
        'wikitext',
        'wikitext-103-v1',
        streaming=streaming,
    )

    text_column_name = 'text'

    # Tokenize in streaming mode
    def tokenize_function(examples: dict) -> Dict[str, torch.Tensor]:
        texts = examples[text_column_name]
        encodings = tokenizer(texts)
        batch_ids: List[List[int]] = encodings['input_ids']
        for ids in batch_ids:
            ids += [eos_token_id]
        concat_ids = sum(batch_ids, [])  # Concatenate into one long ids
        total_len = len(concat_ids)

        chunked_ids: List[List[int]] = []
        chunk_len = max_len + 1

        # Rounded down to multiple of chunk_len.
        # So the last remainder chunk is discarded.
        total_len = total_len // chunk_len * chunk_len

        for i in range(0, total_len, chunk_len):
            this_chunk: List[int] = concat_ids[i:i + chunk_len]
            chunked_ids.append(this_chunk)
        input_ids = torch.tensor(chunked_ids, dtype=torch.long)
        batch = {
            'input_ids': input_ids,
            'labels': input_ids.clone()
        }
        return batch

    if streaming:
        if do_log:
            print(">> Tokenizing data on the fly...")
        tokenized_dataset = raw_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=[text_column_name] if token_ids_only else []
        )
    else:
        if do_log:
            print("Tokenizing data...")
        tokenized_dataset = raw_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=n_workers,
            remove_columns=[text_column_name] if token_ids_only else [],
            load_from_cache_file=not overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    return tokenized_dataset
