from typing import Optional, Tuple, List, Union
from functools import partial

from transformers.utils import logging
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

try:
    from cut_cross_entropy import linear_cross_entropy
except:
    linear_cross_entropy = None
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    BaseModelOutputWithPast,
)
from transformers.cache_utils import Cache, DynamicCache

from .ffn import MLP
from .utils import RMSNorm
from .config import GPTConfig
from .attention import Attention
from .rope import RoPESimple, RoPESimple2D


logger = logging.get_logger(__name__)


class Cache2DPos(DynamicCache):
    """
    A KV cache for supporting 2D position embeddings.

    It supports both:
    1. Rule-based 2D positions: a shared (row, col) state for the whole model.
    2. Adaptive 2D positions: a per-layer (row, col) state because each layer
       predicts its own adaptive 2D position updates.
    """

    def __init__(
        self,
        id_to_newline_count: Optional[Tensor],
        batch_size: int,
        num_layers: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.id_to_newline_count = id_to_newline_count

        # Rule-based 2D rope state: (B, 2), [row, col]
        self.cur_pos = torch.zeros((batch_size, 2), dtype=torch.long)

        # Adaptive 2D rope state: (L, B, 2), [row, col], float because the
        # adaptive position ids are continuous values.
        self.cur_pos_per_layer = torch.zeros(
            (num_layers, batch_size, 2), dtype=torch.float32
        )

    def get_cur_pos(
        self,
        layer_idx: Optional[int] = None,
        adaptive: bool = False,
    ) -> Tensor:
        if adaptive:
            assert layer_idx is not None
            return self.cur_pos_per_layer[layer_idx]
        return self.cur_pos

    def update_pos(
        self,
        value: Tensor,
        layer_idx: Optional[int] = None,
        adaptive: bool = False,
    ):
        """
        value:
        - adaptive=False: (B, 2), long, [row, col]
        - adaptive=True:  (B, 2), float, [row, col]
        """
        if adaptive:
            assert layer_idx is not None
            if (
                self.cur_pos_per_layer.device != value.device
                or self.cur_pos_per_layer.dtype != value.dtype
            ):
                self.cur_pos_per_layer = self.cur_pos_per_layer.to(
                    device=value.device,
                    dtype=value.dtype,
                    non_blocking=True,
                )
            self.cur_pos_per_layer[layer_idx] = value
        else:
            self.cur_pos = value


class Adaptive2DPositionalGenerator(nn.Module):
    """
    Learnable 2D position generator used by adaptive 2D RoPE.

    Given the current layer input h_i, predict:
        A_i = alpha * sigmoid(a_i)
        B_i = alpha * sigmoid(b_i)

    Then update internal (col, row) as:
        col_{i+1} = A_i * col_i + B_i
        row_{i+1} = row_i + (1 - A_i)

    We return the position id in the same convention as the original code:
        [row, col]
    """

    def __init__(self, hidden_size: int, alpha: float = 1.0):
        super().__init__()
        self.ab_proj = nn.Linear(hidden_size, 2, bias=True)
        self.alpha = float(alpha)
        self.reset_parameters()

    def reset_parameters(self):
        # Zero init -> sigmoid(0)=0.5.
        # If alpha=2, then A=B=1 at initialization:
        #   col_{i+1} = col_i + 1
        #   row_{i+1} = row_i
        # This gives a stable 1D-like starting point.
        nn.init.zeros_(self.ab_proj.weight)
        nn.init.zeros_(self.ab_proj.bias)

    @staticmethod
    def _affine_scan_inclusive(scale: Tensor, shift: Tensor) -> tuple[Tensor, Tensor]:
        """
        Inclusive prefix scan over affine transforms:
            x -> scale * x + shift

        If f_i(x) = scale_i * x + shift_i, then the composition rule is
            f_j o f_i = (
                scale_j * scale_i,
                scale_j * shift_i + shift_j
            )

        Args:
            scale: (B, T)
            shift: (B, T)

        Returns:
            prefix_scale: (B, T)
            prefix_shift: (B, T)

        where for each i:
            prefix(x)_i = prefix_scale_i * x + prefix_shift_i
        """
        prefix_scale = scale.clone()
        prefix_shift = shift.clone()

        T = scale.size(1)
        offset = 1
        while offset < T:
            prev_scale = prefix_scale.clone()
            prev_shift = prefix_shift.clone()

            prefix_scale[:, offset:] = (
                prev_scale[:, offset:] * prev_scale[:, :-offset]
            )
            prefix_shift[:, offset:] = (
                prev_scale[:, offset:] * prev_shift[:, :-offset]
                + prev_shift[:, offset:]
            )
            offset <<= 1

        return prefix_scale, prefix_shift

    def forward(
        self,
        hidden_states: Tensor,               # (B, T, H)
        cur_pos: Optional[Tensor] = None,   # (B, 2), [row, col]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Returns:
            pos_ids_2d: (B, T, 2), float32, [row, col]
            new_pos:    (B, 2), float32, [row, col]
            A:          (B, T)
            Bv:         (B, T)
        """
        B, T, _ = hidden_states.shape
        device = hidden_states.device
        work_dtype = torch.float32

        if cur_pos is None:
            cur_pos = torch.zeros((B, 2), device=device, dtype=work_dtype)
        else:
            cur_pos = cur_pos.to(
                device=device,
                dtype=work_dtype,
                non_blocking=True,
            )

        if T == 0:
            pos_ids_2d = torch.empty((B, 0, 2), device=device, dtype=work_dtype)
            empty = torch.empty((B, 0), device=device, dtype=work_dtype)
            return pos_ids_2d, cur_pos, empty, empty

        # (B, T, 2)
        ab = self.alpha * torch.sigmoid(
            self.ab_proj(hidden_states).to(dtype=work_dtype)
        )
        A = ab[..., 0]
        Bv = ab[..., 1]

        # cur_pos is [row, col]
        row0 = cur_pos[:, 0]
        col0 = cur_pos[:, 1]

        # For token i we need p_i, then use A_i/B_i to update p_{i+1}.
        # Therefore positions for the current tokens use an exclusive prefix.
        A_excl = torch.cat(
            [
                torch.ones((B, 1), device=device, dtype=work_dtype),
                A[:, :-1],
            ],
            dim=1,
        )
        B_excl = torch.cat(
            [
                torch.zeros((B, 1), device=device, dtype=work_dtype),
                Bv[:, :-1],
            ],
            dim=1,
        )

        prefix_A, prefix_B = self._affine_scan_inclusive(A_excl, B_excl)
        col = prefix_A * col0.unsqueeze(1) + prefix_B  # (B, T)

        row_delta = 1.0 - A
        row = row0.unsqueeze(1) + torch.cat(
            [
                torch.zeros((B, 1), device=device, dtype=work_dtype),
                row_delta[:, :-1].cumsum(dim=1),
            ],
            dim=1,
        )  # (B, T)

        pos_ids_2d = torch.stack([row, col], dim=-1)  # (B, T, 2), [row, col]

        # Final state after consuming this chunk.
        new_row = row[:, -1] + row_delta[:, -1]
        new_col = A[:, -1] * col[:, -1] + Bv[:, -1]
        new_pos = torch.stack([new_row, new_col], dim=-1)

        return pos_ids_2d, new_pos, A, Bv


class GPTBlock(nn.Module):
    """
    One layer in GPT.

    It does:
    1. x = x + Att(norm(x))
    2. x = x + FFN(norm(x))
    3. return x
    """

    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

        # RoPE type might be layer-specific
        if isinstance(config.pos_emb_type, list):
            self.pos_emb_type = config.pos_emb_type[layer_idx]
        else:
            self.pos_emb_type = config.pos_emb_type

        # Window size might be layer-specific
        if isinstance(config.att_window_size, list):
            self.window_size = config.att_window_size[layer_idx]
        else:
            self.window_size = config.att_window_size

        attn_pos_emb_type = (
            "rope_simple_2d" if self.pos_emb_type == "adap_2d_rope" else self.pos_emb_type
        )

        self.attn = Attention(
            layer_idx=layer_idx,
            hidden_size=config.hidden_size,
            head_dim=config.dim_k,
            n_heads=config.num_attention_heads,
            n_kv_heads=config.num_key_value_heads,
            use_q_norm=config.att_q_norm,
            use_k_norm=config.att_k_norm,
            window_size=self.window_size,
            bias=config.attn_bias,
            output_bias=config.attn_output_bias,
            pos_emb_type=attn_pos_emb_type,
        )
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = MLP(
            act_name=config.ffn_act_fn,
            hidden_size=config.hidden_size,
            intermediate_dim=config.ffn_d_mid,
            is_gated=config.ffn_is_gated,
            bias=config.ffn_bias,
        )

        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        position_embeddings_2d: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> Tensor | dict[str, Tensor | tuple[Tensor, Tensor]] | tuple:
        """
        Args:
        - hidden_states: (B, T, hidden_size)
        """

        if self.pos_emb_type == 'nope':
            position_embeddings = None
            position_embeddings_2d = None

        # ====== Attention ======
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,  # Position embeddings are always passed.
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            position_embeddings_2d=position_embeddings_2d,
        )
        hidden_states = residual + hidden_states

        # ====== MLP ======
        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class GPTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple
    interface for downloading and loading pretrained models.
    """

    config_class = GPTConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GPTBlock"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights with Llama 2 scheme."""
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class GPTModel(GPTPreTrainedModel):
    def __init__(self, config: GPTConfig):
        super().__init__(config=config)
        self.config = config
        self.use_grad_ckpt = True

        self.input_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [GPTBlock(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

        if isinstance(self.config.pos_emb_type, list):
            pos_types = set(self.config.pos_emb_type)
            self.has_rope = ('rope_simple' in pos_types) or ('rope' in pos_types)
            self.has_rule_2d_rope = 'rope_simple_2d' in pos_types
            self.has_adap_2d_rope = 'adap_2d_rope' in pos_types
            self.has_absolute_pos_emb = 'absolute' in pos_types
        else:
            self.has_rope = self.config.pos_emb_type in ['rope_simple', 'rope']
            self.has_rule_2d_rope = self.config.pos_emb_type in ['rope_simple_2d']
            self.has_adap_2d_rope = self.config.pos_emb_type in ['adap_2d_rope']
            self.has_absolute_pos_emb = self.config.pos_emb_type in ['absolute']

        self.has_2d_rope = self.has_rule_2d_rope or self.has_adap_2d_rope

        if self.has_rope:
            self.rope = RoPESimple(
                head_dim=config.dim_k,
                theta=config.rope_theta,
            )

        if self.has_2d_rope:
            self.rope_2d = RoPESimple2D(
                head_dim=config.dim_k,
                theta_x=config.rope_theta_x,
                theta_y=config.rope_theta_y,
            )

        if self.has_absolute_pos_emb:
            # Learned absolute position embeddings.
            self.pos_emb = nn.Embedding(config.max_len, config.hidden_size)

        self.model_parallel = False
        self.gradient_checkpointing = False

        # Initialize as a buffer (gets saved/loaded automatically)
        if self.has_rule_2d_rope:
            self.register_buffer(
                "id_to_newline_count",
                torch.zeros(config.vocab_size, dtype=torch.long),
                persistent=True,
            )

        if self.has_adap_2d_rope:
            self.adap_2d_rope_alpha = float(
                getattr(config, "adap_2d_rope_alpha", 1.0)
            )
            self.adap_2d_pos_generators = nn.ModuleList(
                [
                    Adaptive2DPositionalGenerator(
                        hidden_size=config.hidden_size,
                        alpha=self.adap_2d_rope_alpha,
                    )
                    for _ in range(config.num_hidden_layers)
                ]
            )

        self.post_init()

    def set_id_to_newline_count(self, value):
        assert self.has_rule_2d_rope, "The model does not have rule-based 2D RoPE"
        self.register_buffer("id_to_newline_count", value, persistent=True)

    def gradient_checkpointing_enable(self):
        """
        Enable gradient checkpointing for the model.
        """
        self.use_grad_ckpt = True

    def _make_pos2d_for_flat_ids(
        self,
        flat_ids: Tensor,
        nl_counts: Tensor,
    ) -> Tensor:
        """
        输入: flat_ids 形状 (L,), 为 concat 后截断到 total_len 的 token 序列
        输出: pos2d 形状 (L, 2), 每个位置是 [row, col]
        规则：
        - nl == 0: pos=(row, col); col += 1
        - nl >= 1: row += nl; col = 0; pos=(row, 0); col = 1
        """
        L = flat_ids.shape[0]
        pos = torch.empty((L, 2), dtype=torch.long)
        row = 0
        col = 0

        for i in range(L):
            tok = int(flat_ids[i])
            nl = int(nl_counts[tok]) if 0 <= tok < nl_counts.shape[0] else 0
            if nl <= 0:
                pos[i, 0] = row
                pos[i, 1] = col
                col += 1
            else:
                pos[i, 0] = row
                pos[i, 1] = col
                col = 0
                row += nl

        return pos  # (L, 2)

    def old_get_position_ids(
        self,
        input_ids: torch.Tensor,
        past_key_values: Cache2DPos | None = None,
    ) -> torch.Tensor:
        B, T = input_ids.shape
        if past_key_values is not None:
            cur_pos = past_key_values.get_cur_pos(adaptive=False)
        else:
            cur_pos = torch.zeros((B, 2), dtype=torch.long)
        # Position IDs
        pos_ids = torch.empty((B, T, 2), dtype=torch.long)
        # New cached position information
        new_pos = torch.empty((B, 2), dtype=torch.long)
        # for each example, update the position
        for bi in range(B):
            # Construct the position ids for this example
            # (T, 2)
            this_pos = self._make_pos2d_for_flat_ids(
                input_ids[bi], self.id_to_newline_count
            )
            # All col-index in the first row are shifted based on cur_pos[bi, 1]
            this_pos[this_pos[:, 0] == 0, 1] += cur_pos[bi, 1]
            # All row-index are shifted based on cur_pos[bi, 0]
            this_pos[:, 0] += cur_pos[bi, 0]
            pos_ids[bi] = this_pos

            # Update the current position in the cache
            new_pos[bi] = this_pos[-1]  # (2,)
            # The row index is incremented by 1 since it is the number of
            # rows seen so far, NOT the index of the last row.
            new_pos[bi, 1] += 1

        if past_key_values is not None:
            past_key_values.update_pos(new_pos, adaptive=False)

        return pos_ids

    def _get_position_ids(
        self,
        input_ids: torch.Tensor,
        past_key_values: Cache2DPos | None = None,
    ) -> torch.Tensor:
        """
        fast version of rule-based 2d position embedding:
        - token 先拿当前位置 (row, col)
        - 若 nl > 0，则 reset 发生在该 token 之后(对齐英发的逻辑)
        return:
            pos_ids: (B, T, 2), dtype=torch.long
        """
        device = input_ids.device
        dtype = torch.long
        B, T = input_ids.shape

        if past_key_values is None:
            cur_pos = input_ids.new_zeros((B, 2), dtype=dtype)
        else:
            cur_pos = past_key_values.get_cur_pos(adaptive=False).to(
                device=device, dtype=dtype, non_blocking=True
            )

        if T == 0:
            pos_ids = input_ids.new_empty((B, 0, 2), dtype=dtype)
            if past_key_values is not None:
                past_key_values.update_pos(cur_pos, adaptive=False)
            return pos_ids

        nl_table = self.id_to_newline_count
        if nl_table.device != device:
            # 正常情况下 register_buffer 后不会走到这里
            nl_table = nl_table.to(device=device, dtype=dtype, non_blocking=True)

        vocab_size = nl_table.numel()

        safe_ids = input_ids.clamp(0, vocab_size - 1)
        nl = nl_table[safe_ids]
        valid = (input_ids >= 0) & (input_ids < vocab_size)
        nl = nl.masked_fill(~valid, 0)  # (B, T), long

        breaks = nl > 0
        inc = nl * breaks.to(nl.dtype)  # nl<=0 时增量为 0

        # row_local[i] = sum_{j < i} inc[j]
        row_local = inc.cumsum(dim=1) - inc  # (B, T)

        # idx: 0..T-1
        idx = torch.arange(T, device=device, dtype=dtype).view(1, T).expand(B, T)

        # break_plus1[k] = k+1 if token k causes a break else 0
        break_plus1 = (idx + 1) * breaks.to(dtype)  # (B, T)

        # 到当前位置为止（包含当前位置）的最后一个 break 的位置+1
        last_break_plus1_incl = torch.cummax(break_plus1, dim=1).values  # (B, T)

        # 我们要的是“当前位置之前”的最后一个 break，所以右移一格
        prev_break_plus1 = torch.empty_like(last_break_plus1_incl)
        prev_break_plus1[:, 0] = 0
        prev_break_plus1[:, 1:] = last_break_plus1_incl[:, :-1]

        # col_local[i] = i - (last_break_before_i + 1)
        # 若此前没有 break，则 prev_break_plus1=0，col_local=i
        col_local = idx - prev_break_plus1  # (B, T)

        # 全部 row 都要加 cur_row
        row = row_local + cur_pos[:, 0:1]

        # 只有当前chunk 的第一段要续接 cur_col
        first_segment_mask = (prev_break_plus1 == 0).to(dtype)
        col = col_local + first_segment_mask * cur_pos[:, 1:2]

        pos_ids = torch.stack((row, col), dim=-1)  # (B, T, 2)

        if past_key_values is not None:
            new_row = row[:, -1] + inc[:, -1]
            new_col = torch.where(
                breaks[:, -1],
                torch.zeros_like(col[:, -1]),
                col[:, -1] + 1,
            )
            new_pos = torch.stack((new_row, new_col), dim=-1)
            past_key_values.update_pos(new_pos, adaptive=False)

        return pos_ids

    def _get_adaptive_position_ids(
        self,
        hidden_states: Tensor,
        layer_idx: int,
        past_key_values: Cache2DPos | None = None,
    ) -> Tensor:
        """
        Adaptive 2D position ids for one layer.

        Args:
            hidden_states: (B, T, H), the current layer input
            layer_idx: int
            past_key_values: Cache2DPos or None

        Returns:
            pos_ids_2d: (B, T, 2), float32, [row, col]
        """
        assert self.has_adap_2d_rope, "The model does not have adaptive 2D RoPE"

        if past_key_values is None:
            cur_pos = None
        else:
            cur_pos = past_key_values.get_cur_pos(
                layer_idx=layer_idx,
                adaptive=True,
            )

        pos_ids_2d, new_pos, _, _ = self.adap_2d_pos_generators[layer_idx](
            hidden_states=hidden_states,
            cur_pos=cur_pos,
        )

        if past_key_values is not None:
            # The cache state is only used as the starting point for the next call,
            # so it is safe to detach here.
            past_key_values.update_pos(
                new_pos.detach(),
                layer_idx=layer_idx,
                adaptive=True,
            )

        return pos_ids_2d

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[Tensor] = None,
        inputs_embeds: Tensor = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> dict[str, Tensor | List[Tuple[Tensor, Tensor]]] | BaseModelOutputWithPast:
        """
        input_ids: (B, T)
        """
        B, T = input_ids.size()

        assert inputs_embeds is None, "Current only supports input_ids"

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        # use_cache = use_cache if use_cache is not None else self.config.use_cache
        use_cache = use_cache if use_cache is not None else getattr(self.config, "use_cache", False)
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache:
            if self.has_2d_rope:
                if past_key_values is None:
                    past_key_values = Cache2DPos(
                        id_to_newline_count=getattr(
                            self, "id_to_newline_count", None
                        ),
                        batch_size=B,
                        num_layers=self.config.num_hidden_layers,
                    )
                elif not isinstance(past_key_values, Cache2DPos):
                    logger.warning_once(
                        "2D RoPE requires Cache2DPos to track row/col states. "
                        "Got a different cache type, so we re-initialize a fresh Cache2DPos."
                    )
                    past_key_values = Cache2DPos(
                        id_to_newline_count=getattr(
                            self, "id_to_newline_count", None
                        ),
                        batch_size=B,
                        num_layers=self.config.num_hidden_layers,
                    )
            else:
                if past_key_values is not None and not isinstance(past_key_values, Cache):
                    return_legacy_cache = True
                    past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                    logger.warning_once(
                        "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                        "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                        "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                    )
                elif past_key_values is None:
                    past_key_values = DynamicCache()

        if inputs_embeds is None:
            # (B, T, d_model)
            inputs_embeds = self.input_emb(input_ids)

        # ========== Position embeddings ===========
        # Now, to make the code simple, we simple ignore position_ids,
        # and always generate position_ids in the forward pass.
        rule_position_ids_2d = None
        if self.has_rule_2d_rope:
            # Rule-based 2D position IDs, shape: (B, T, 2)
            rule_position_ids_2d = self._get_position_ids(input_ids, past_key_values)

        if self.has_rope or self.has_absolute_pos_emb:
            # 1D position IDs, the shape is (T)
            if cache_position is None:
                past_seen_tokens = (
                    past_key_values.get_seq_length()
                    if past_key_values is not None
                    else 0
                )
                cache_position = torch.arange(
                    past_seen_tokens,
                    past_seen_tokens + inputs_embeds.shape[1],
                    device=inputs_embeds.device,
                )
            position_ids = cache_position.unsqueeze(0)

        hidden_states = inputs_embeds

        if self.has_absolute_pos_emb:
            # Absolute position embeddings, directly added to the hidden states.
            abs_position_embeddings = self.pos_emb(position_ids)  # (1, T, d_model)
            hidden_states = inputs_embeds + abs_position_embeddings

        # Create RoPE embeddings, shared across layers where possible
        position_embeddings = None
        if self.has_rope:
            position_embeddings = self.rope(
                hidden_states, position_ids
            )  # (B, T, dim_k/2, 2, 2)

        rule_position_embeddings_2d = None
        if self.has_rule_2d_rope:
            rule_position_embeddings_2d = self.rope_2d(
                hidden_states, rule_position_ids_2d
            )  # (B, T, dim_k/2, 2, 2)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = past_key_values if use_cache else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_position_embeddings_2d = None
            if layer.pos_emb_type == "rope_simple_2d":
                layer_position_embeddings_2d = rule_position_embeddings_2d
            elif layer.pos_emb_type == "adap_2d_rope":
                layer_position_ids_2d = self._get_adaptive_position_ids(
                    hidden_states=hidden_states,
                    layer_idx=layer.layer_idx,
                    past_key_values=(
                        past_key_values if isinstance(past_key_values, Cache2DPos) else None
                    ),
                )
                layer_position_embeddings_2d = self.rope_2d(
                    hidden_states,
                    layer_position_ids_2d,
                )

            if self.use_grad_ckpt and self.training:
                layer_forward = partial(checkpoint, layer.__call__, use_reentrant=False)
            else:
                layer_forward = layer.__call__

            layer_outputs = layer_forward(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                position_embeddings_2d=layer_position_embeddings_2d,
            )  # (B, T, d_model)

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)  # (B, T, d_model)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_decoder_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache and next_decoder_cache is not None:
            next_decoder_cache = next_decoder_cache.to_legacy_cache()

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attns,
                ]
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class GPTForCausalLM(GPTPreTrainedModel, GenerationMixin):
    def __init__(self, config: GPTConfig):
        super().__init__(config=config)
        self.config = config
        self.use_grad_ckpt = False
        self.cce_loss_impl = "torch_compile"

        self.model = GPTModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if self.config.tie_word_embeddings:
            print("Tying embeddings...")
            self.lm_head.weight = self.model.input_emb.weight

        self.post_init()

    def _build_newline_count_table(
        self,
        tokenizer,
        treat_eos_as_newline: bool = False,
    ) -> torch.Tensor:
        """
        返回形状 (vocab_size,) 的 int32 数组：每个 token 解码字符串中的 '\n' 个数。
        """
        eos_token_id = tokenizer.eos_token_id
        vocab_size = tokenizer.vocab_size
        counts = torch.zeros((vocab_size,), dtype=torch.long)
        for i in range(vocab_size):
            try:
                s = tokenizer.decode([i], clean_up_tokenization_spaces=False)
            except Exception:
                try:
                    s = tokenizer.convert_ids_to_tokens(i)
                    if not isinstance(s, str):
                        s = str(s)
                except Exception:
                    s = ""
            counts[i] = s.count("\n")
        if treat_eos_as_newline and 0 <= eos_token_id < vocab_size:
            counts[eos_token_id] = max(counts[eos_token_id], 1)
        return counts

    def prepare_2dpos_with_tokenizer(
        self,
        tokenizer,
        treat_eos_as_newline: bool = False,
    ):
        if not self.model.has_rule_2d_rope:
            logger.warning_once(
                "prepare_2dpos_with_tokenizer is only needed for rule-based 2D RoPE. Skipping."
            )
            return

        newline_count_per_id = self._build_newline_count_table(
            tokenizer=tokenizer,
            treat_eos_as_newline=treat_eos_as_newline,
        )
        self.model.set_id_to_newline_count(newline_count_per_id)

    def get_input_embeddings(self):
        return self.model.input_emb

    def set_input_embeddings(self, value):
        self.model.input_emb = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def get_num_params(
        self,
        non_embedding: bool = True,  # Whether to exclude input embedding parameters.
    ) -> int:
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            if self.model.has_absolute_pos_emb:
                n_params -= self.model.pos_emb.weight.numel()
            n_params -= self.model.input_emb.weight.numel()
        return n_params

    def gradient_checkpointing_enable(self):
        """
        Enable gradient checkpointing for the model.
        """
        self.use_grad_ckpt = True
        self.model.gradient_checkpointing_enable()

    def compute_loss(
        self,
        hidden_states: Tensor,
        labels: Tensor,
    ) -> Tensor:
        if linear_cross_entropy is not None:
            # Use Fused CE implementation
            loss = linear_cross_entropy(
                hidden_states,
                self.lm_head.weight,
                labels,
                shift=True,
                impl=self.cce_loss_impl,
            )
        else:
            # PyTorch implementation
            logits = self.lm_head(hidden_states)  # (B, T, vocab_size)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return loss

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: bool = True,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **loss_kwargs,
    ) -> tuple | CausalLMOutputWithPast:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, GPTForCausalLM

        >>> model = GPTForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        if return_dict:
            hidden_states = outputs["last_hidden_state"]  # (B, T, d_model)
        else:
            hidden_states = outputs[0]  # (B, T, d_model)

        if self.training:
            logits = None
        else:
            # For inference, only need logits for the last token
            logits = self.lm_head(
                hidden_states[:, -num_logits_to_keep:]
            )  # (B, num_logits_to_keep, vocab_size)

        if labels is not None:
            loss = self.compute_loss(hidden_states, labels)
        else:
            loss = None

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
