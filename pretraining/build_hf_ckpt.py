import torch
from modeling.gpt import GPTForCausalLM, GPTConfig
from transformers import AutoTokenizer
from torch import Tensor
from pathlib import Path
from safetensors.torch import load_file
import json
from tap import Tap


def load_state_dict(ckpt_dir: Path) -> dict[str, Tensor]:  # type: ignore
    # Try to load from a safetensors checkpoint.
    safetensors_path = ckpt_dir / 'model.safetensors'
    if safetensors_path.exists():
        return load_file(safetensors_path)

    # Try to load from a FSDP checkpoint.
    fsdp_path = ckpt_dir / 'pytorch_model_fsdp.bin'
    if fsdp_path.exists():
        state_dict = torch.load(fsdp_path, map_location="cpu")
        return state_dict

    # Try to load from a DeepSpeed checkpoint.
    ds_index_path = ckpt_dir / 'output_dir/pytorch_model.bin.index.json'
    if ds_index_path.exists():
        with open(ds_index_path, "r") as f:
            index = json.load(f)

        weight_map = index["weight_map"]
        unique_files = set(weight_map.values())

        state_dict = {}

        for shard_file in sorted(unique_files):  # Sorting is optional but helpful
            shard_path = ckpt_dir / 'output_dir' / shard_file
            shard = torch.load(shard_path, map_location="cpu")
            state_dict.update(shard)

        return state_dict

    raise ValueError(f"No checkpoint found in {ckpt_dir}")


class Args(Tap):
    only_state_dict: int = 0
    tok_path = 'tokenizer/llama2'

    # config_path = "results/proj/2drope_dclm/model_config.json"

    ckpt_path = "results/2drope/e3_gpt-730m-2drope-x1000-y1000_dclm-100b_"
    out_path = "ckpts/2drope-730m-dclm-100b"

    # ckpt_path = "results/proj/rope_dclm"
    # out_path = "ckpts/rope_dclm"
    ckpt = "ckpt_100000"


args = Args().parse_args()
ckpt_path = Path(args.ckpt_path) / args.ckpt
tok_path = Path(args.tok_path)
config_path = Path(args.ckpt_path) / "model_config.json"
out_path = Path(args.out_path) / args.ckpt
out_path.mkdir(exist_ok=True, parents=True)
assert ckpt_path.exists(), f"Checkpoint {ckpt_path} does not exist"
assert config_path.exists(), f"Config {config_path} does not exist"
assert tok_path.exists(), f"Tokenizer {tok_path} does not exist"

print(f"Loading tokenizer from {tok_path}")
tokenizer = AutoTokenizer.from_pretrained(tok_path)
print(f"Loading config from {config_path}")
config = GPTConfig.from_json_file(config_path)

print(config)

print("Instantiating model")
model = GPTForCausalLM(config=config)

print(f"Loading state dict from {ckpt_path}")
state_dict = load_state_dict(ckpt_path)
print(list(state_dict.keys()))

# if bool(args.only_state_dict):
#     print(f"Saving state dict to {out_path}")
#     torch.save(state_dict, out_path / 'state_dict.pt')
#     exit()

if 'lm_head.weight' not in state_dict:
    assert config.tie_word_embeddings, "lm_head.weight is not in the state dict, so tie_word_embeddings must be True"
    # If there is no lm_head.weight, tie it to the word embeddings.
    print("Tying word embeddings...")
    state_dict['lm_head.weight'] = state_dict['model.input_emb.weight']

print("Loading parameters into model")
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
print(f"Missing keys: {missing_keys}")
print(f"Unexpected keys: {unexpected_keys}")
model = model.to(torch.bfloat16)


def prepare_2dpos():
    treat_eos_as_newline = False
    eos_token_id = tokenizer.eos_token_id

    def _build_newline_count_table(tok) -> torch.Tensor:
        """
        返回形状 (vocab_size,) 的 int32 数组：每个 token 解码字符串中的 '\n' 个数。
        """
        vocab_size = tok.vocab_size
        counts = torch.zeros((vocab_size,), dtype=torch.long)
        for i in range(vocab_size):
            try:
                s = tok.decode([i], clean_up_tokenization_spaces=False)
            except Exception:
                try:
                    s = tok.convert_ids_to_tokens(i)
                    if not isinstance(s, str):
                        s = str(s)
                except Exception:
                    s = ""
            counts[i] = s.count("\n")
        if treat_eos_as_newline and 0 <= eos_token_id < vocab_size:
            counts[eos_token_id] = max(counts[eos_token_id], 1)
        return counts


    newline_count_per_id = _build_newline_count_table(tokenizer)  # (V,)
    # Pad to 50304
    target_vocab_size = 50304
    newline_count_per_id = torch.cat([newline_count_per_id, torch.zeros((target_vocab_size - newline_count_per_id.shape[0],))])
    print(f"{newline_count_per_id.shape = }")

    model.model.set_id_to_newline_count(newline_count_per_id)


if "2d" in args.ckpt:
    prepare_2dpos()


print("Registering model")
model.config.register_for_auto_class()
model.model.register_for_auto_class("AutoModel")
model.register_for_auto_class("AutoModelForCausalLM")

print(f"Saving model to {out_path}")
model.save_pretrained(out_path, safe_serialization=False)
print(f"Saving tokenizer to {out_path}")
tokenizer.save_pretrained(out_path, safe_serialization=False)
