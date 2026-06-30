from typing import Optional

from torch import nn, Tensor
import torch
import torch.nn.functional as F
from einops import rearrange
from transformers.cache_utils import DynamicCache

from .utils import RMSNorm
from .rope import RoPESimple, RoPE, RoPESimple2D

try:
    from flash_attn import flash_attn_func  # type: ignore[import]
except Exception:
    flash_attn_func = None
    print("WARNING: flash_attn not found. Using slow attention.")


class Attention(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        use_q_norm: bool = False,
        use_k_norm: bool = False,
        max_len: int = 4096,
        device: str = "cuda",
        window_size: int = -1,
        chunk_size: int | None = None,
        bias: bool = False,
        output_bias: bool = False,
        pos_emb_type: str = 'rope_simple',
    ):
        super().__init__()

        self.device = device
        self.d_model = hidden_size
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.layer_idx = layer_idx
        self.use_q_norm = use_q_norm
        self.use_k_norm = use_k_norm
        self.window_size = window_size
        self.bias = bias
        self.pos_emb_type = pos_emb_type

        if flash_attn_func is None:
            assert (
                window_size == -1
            ), "Sliding window is only supported when we have flash attention."
        # print("Attention window size:", window_size)

        self.group_size = n_heads // n_kv_heads

        self.q_proj = nn.Linear(hidden_size, n_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, n_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, n_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(n_heads * head_dim, hidden_size, bias=output_bias)

        if self.use_q_norm:
            self.q_norm = RMSNorm(head_dim)

        if self.use_k_norm:
            self.k_norm = RMSNorm(head_dim)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,  # Not used currently.
        past_key_value: DynamicCache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: None | Tensor = None,
        position_embeddings_2d: None | Tensor = None,
    ) -> tuple:
        """
        Args:
            hidden_states: (B, T, D)
            position_embeddings: (T, dim_k/2, 2, 2), often called freqs_cis.

        ---
        B: batch size
        T: sequence length
        D: model dimension
        """
        B, T, D = hidden_states.shape
        q = self.q_proj(hidden_states)  # (B, T, H * dim_k)
        v = self.v_proj(hidden_states)  # (B, T, H * dim_v)
        k = self.k_proj(hidden_states)  # (B, T, H * dim_k)

        q = rearrange(q, "b t (h dk) -> b h t dk", h=self.n_heads)
        v = rearrange(v, "b t (h dv) -> b h t dv", h=self.n_kv_heads)
        k = rearrange(k, "b t (h dk) -> b h t dk", h=self.n_kv_heads)

        if self.use_q_norm:
            q = self.q_norm(q)

        if self.use_k_norm:
            k = self.k_norm(k)

        # Add RoPE and 2d RoPE
        
        # Add positional embedding (RoPE / 2D-RoPE)
        cos = sin = None  # for cache kwargs

        if self.pos_emb_type == "rope_simple_2d":
            # Prefer 2D RoPE if provided
            if position_embeddings_2d is None:
                raise ValueError("pos_emb_type=rope_simple_2d but position_embeddings_2d is None")
            cos, sin = position_embeddings_2d  # (B,T,Dk) or multimodal variant
            q, k = RoPESimple2D.apply_rotary_emb(q, k, cos, sin)
        else:
            if position_embeddings is not None:
                if self.pos_emb_type == "rope_simple":
                    cos, sin = position_embeddings  # (B,T,Dk) or (B,3,T,Dk) etc
                    q, k = RoPESimple.apply_rotary_emb(q, k, cos, sin)
                else:
                    # complex cis version
                    q, k = RoPE.apply_rotary_emb(q, k, seq_dim=2, freqs_cis=position_embeddings)

        # Return KV cache if needed
        if past_key_value is not None:
            # TODO: DynamicCache from transformers requires sequence length to be the
            # third dimension. We need to replace DynamicCache altogether to fix this.
            if self.pos_emb_type in ["rope_simple", "rope_simple_2d"]:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            else:
                # These kwargs are specific to RoPE models
                cache_kwargs = {"freqs_cis": position_embeddings, "cache_position": cache_position}
            k, v = past_key_value.update(k, v, self.layer_idx, cache_kwargs)

        # GQA
        if self.n_kv_heads < self.n_heads:
            num_repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(num_repeat, dim=1)  # (B, H, T, dim_k)
            v = v.repeat_interleave(num_repeat, dim=1)  # (B, H, T, dim_v)

        # Causal self-attention
        # (B, H, T, dim_v) x (B, H, dim_v, T)
        # -> (B, H, T, T)
        if flash_attn_func is not None:
            q, k, v = (rearrange(x, 'b h t d -> b t h d') for x in [q, k, v])
            attn_output = flash_attn_func(
                q=q,
                k=k,
                v=v,
                dropout_p=0.0,
                causal=True,
                window_size=(
                    (-1, -1) if self.window_size is None else (self.window_size - 1, 0)
                ),
            )  # (B, H, T, dim_v)
            attn_output = rearrange(attn_output, 'b t h d -> b t (h d)')
        else:
            attn_output = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                dropout_p=0.0,
                is_causal=True and q.shape[2] > 1,
            )
            attn_output = rearrange(attn_output, "b h t d -> b t (h d)").contiguous()
        attn_output = self.o_proj(attn_output)  # (B, T, D)
        return (attn_output, None, past_key_value)


class CausalSelfAttn(nn.Module):
    def __init__(
        self,
        d_model: int,
        dim_k: int,
        dim_v: int,
        n_head: int,
        n_kv_head: int,
        use_q_norm: bool = False,
        use_k_norm: bool = False,
        tie_kv: bool = False,
        max_len: int = 4096,
        head_mixing: int = 1,
        device: str = "cuda",
        k_to_v: bool = False,
        window_size: int = -1,
    ):
        super().__init__()

        self.device = device
        self.d_model = d_model
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.use_q_norm = use_q_norm
        self.use_k_norm = use_k_norm
        self.tie_kv = tie_kv
        self.head_mixing = head_mixing
        self.k_to_v = k_to_v
        self.window_size = window_size

        self.group_size = n_head // n_kv_head

        self.w_q = nn.Linear(d_model, n_head * dim_k, bias=False)
        if self.tie_kv:
            assert dim_k == dim_v, "tie_kv requires dim_k == dim_v"
            self.w_kv = nn.Linear(d_model, n_kv_head * dim_k, bias=False)
        else:
            self.w_v = nn.Linear(d_model, n_kv_head * dim_v, bias=False)
            if self.k_to_v:
                self.w_k = nn.Linear(dim_v, dim_k, bias=False)
            else:
                self.w_k = nn.Linear(d_model, n_kv_head * dim_k, bias=False)

        self.w_o = nn.Linear(n_head * dim_v, d_model, bias=False)

        # Will use flash attention when available
        self.use_flash_attn = True

        if self.use_q_norm:
            self.q_norm = RMSNorm(dim_k)

        if self.use_k_norm and not self.tie_kv:
            self.k_norm = RMSNorm(dim_k)

    def forward(
        self,
        x: Tensor,
        position_ids: Tensor | None = None,  # Not used currently.
        return_kvs: bool = False,
        kv_cache: None | Tensor | tuple[Tensor, Tensor] = None,
        position_embeddings: None | Tensor = None,
    ) -> dict[str, Tensor | None | tuple[Tensor, Tensor]]:
        """
        Args:
            x: (B, T, D)
            pos_embs: (T, dim_k/2, 2, 2), often called freqs_cis.

        ---
        B: batch size
        T: sequence length
        D: model dimension
        """
        B, T, D = x.shape
        q = self.w_q(x)  # (B, T, H * dim_k)
        q = rearrange(q, "b t (h dk) -> b t h dk", h=self.n_head)

        if self.use_q_norm:
            q = self.q_norm(q)

        if self.tie_kv:
            # (B, T, H * dim_k)
            kv = self.w_kv(x)
            kv = rearrange(kv, "b t (h dk) -> b t h dk", h=self.n_kv_head)

            if self.use_k_norm:
                kv = self.k_norm(kv)
        else:
            # (B, T, H * dim_v)
            v = self.w_v(x)  # (B, T, H * dim_v)
            v = rearrange(v, "b t (h dv) -> b t h dv", h=self.n_kv_head)

            if self.k_to_v:
                k = self.w_k(v)
            else:
                k = self.w_k(x)  # (B, T, H * dim_k)
                k = rearrange(k, "b t (h dk) -> b t h dk", h=self.n_kv_head)

            if self.use_k_norm:
                k = self.k_norm(k)

        # Add RoPE
        if position_embeddings is not None:
            q, k = RoPE.apply_rotary_emb(
                xq=q,
                xk=k,
                seq_dim=1,
                freqs_cis=position_embeddings,
            )

        # Concatenate with KV cache
        if kv_cache is not None:
            # Concatenate current KVs and the KV cache
            if self.tie_kv:
                assert isinstance(kv_cache, Tensor)
                kv = torch.cat((kv_cache, kv), dim=1)  # (B, T, H, dim_kv)
            else:
                assert isinstance(kv_cache, tuple)
                # kv_cache is a tuple of (k, v)
                k_cache, v_cache = kv_cache  # (B, T_ctx, H, dim_k)
                k = torch.cat((k_cache, k), dim=1)  # (B, T_ctx + T, H, dim_k)
                v = torch.cat((v_cache, v), dim=1)  # (B, T_ctx + T, H, dim_v)

        # Apply GQA and/or head mixing.
        if self.head_mixing > 1:
            if self.group_size > 1:
                """
                 Q: 0123456701234567
                KV: 0000111122223333
                """
                # TODO: implement head mixing
                raise NotImplementedError("Head mixing with GQA is not implemented")
            else:
                """
                 Q: 000111222
                KV: 012012012
                """
                q = q.repeat(
                    (1, 1, self.head_mix_size, 1)
                )  # (B, T, H * mix_size, dim_k)
                k = k.repeat_interleave(
                    self.head_mix_size, dim=2
                )  # (B, T, H * mix_size, dim_k)
                v = v.repeat_interleave(
                    self.head_mix_size, dim=2
                )  # (B, T, H * mix_size, dim_v)
        elif self.group_size > 1:
            # GQA: duplicate the keys and values for each group
            if self.tie_kv:
                kv = kv.repeat_interleave(
                    self.group_size, dim=2
                )  # (B, T, H * G, dim_kv)
            else:
                k = k.repeat_interleave(self.group_size, dim=2)  # (B, T, H * G, dim_k)
                v = v.repeat_interleave(self.group_size, dim=2)  # (B, T, H * G, dim_v)

        # Return KV cache if needed
        if return_kvs:
            if self.tie_kv:
                kvs = kv
            else:
                kvs = (k, v)
        else:
            kvs = None

        # Causal self-attention
        # (B, H, T, dim_v) x (B, H, dim_v, T)
        # -> (B, H, T, T)
        # NOTE: When dim_k != dim_v, we cannot use flash_attn
        # efficient attention using Flash Attention CUDA kernels
        if self.tie_kv:
            q, kv = map(lambda e: e.transpose(1, 2), (q, kv))  # (B, H, T, dim_kv)
            output = F.scaled_dot_product_attention(q, kv, kv, is_causal=True)
        else:
            q, k, v = map(lambda e: e.transpose(1, 2), (q, k, v))  # (B, H, T, dim_k)
            output = F.scaled_dot_product_attention(
                q, k, v, is_causal=True
            )  # (B, H, T, dim_v)

        output = rearrange(output, "b h t dv -> b t (h dv)").contiguous()
        output = self.w_o(output)  # (B, T, D)
        return {
            "hidden_states": output,
            "kvs": kvs,
        }
