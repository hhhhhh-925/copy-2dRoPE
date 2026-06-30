from typing import Optional, Tuple

import torch
from torch import nn, Tensor
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS


class RoPESimple2D(nn.Module):
    """
    A minimal 2D RoPE (row x col) in the style of Qwen2.5 RoPESimple.

    Inputs:
      - x:            (B, T, D)  just for dtype/device
      - position_ids: (B, T, 2)  int/long, [:,:,0]=row(x), [:,:,1]=col(y)

    Angle on each rotary plane k:
        angle[b,t,k] = row[b,t] * inv_freq_x[k] + col[b,t] * inv_freq_y[k]

    Returns:
      - cos, sin: both (B, T, D)

    Then call:
        q, k = RoPESimple2D.apply_rotary_emb(q, k, cos, sin)
    which matches the original broadcasting behavior.
    """

    def __init__(
        self,
        head_dim: int,
        theta_x: float,
        theta_y: Optional[float] = None,
        device=None,
    ):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.dim = head_dim
        self.theta_x = float(theta_x)
        self.theta_y = float(theta_x if theta_y is None else theta_y)
        init_inv_x, self.attention_scaling = self._precompute_inv_freqs(
            theta=self.theta_x, head_dim=head_dim // 2, device=device
        )
        init_inv_y, _ = self._precompute_inv_freqs(
            theta=self.theta_y, head_dim=head_dim // 2, device=device
        )

        z = torch.zeros(head_dim // 4, device=init_inv_x.device, dtype=init_inv_x.dtype)
        
        inv_x = torch.cat([init_inv_x, z], dim=0)
        inv_y = torch.cat([z, init_inv_y], dim=0)
        
        # Register buffers (no-grad), keep float32 for trig stability; cast later to x.dtype.
        self.register_buffer("inv_freq_x", inv_x, persistent=False)  # (D/2,)
        self.register_buffer("inv_freq_y", inv_y, persistent=False)  # (D/2,)

    def _precompute_inv_freqs(
        self, theta: float, head_dim: int, device=None
    ) -> Tuple[Tensor, float]:
        base = theta
        partial_rotary_factor = 1.0
        dim = int(head_dim * partial_rotary_factor)
        # length = dim/2 bands
        idx = torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float)
        inv_freq = 1.0 / (base ** (idx / dim))
        attention_factor = 1.0  # kept for parity with RoPESimple; not used here
        return inv_freq, attention_factor

    def forward(self, x: Tensor, position_ids: Tensor) -> Tuple[Tensor, Tensor]:
        """
        x:            (B, T, D)
        position_ids: (B, T, 2)  -> row/col
        returns: cos, sin: (B, T, D)
        """
        assert position_ids.dim() == 3 and position_ids.size(-1) == 2, \
            f"RoPESimple2D expects position_ids of shape (B,T,2), got {tuple(position_ids.shape)}"

        # print("position_ids:", position_ids)
        
        B, T, _ = position_ids.shape
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"

        # (B, D/2, 1)
        inv_x = self.inv_freq_x[None, :, None].float().expand(B, -1, 1)
        inv_y = self.inv_freq_y[None, :, None].float().expand(B, -1, 1)

        # (B, 1, T)
        row = position_ids[..., 0][:, None, :].float()
        col = position_ids[..., 1][:, None, :].float()
        # print("row:", row) # 0,0,0,0,0,0...,0
        # print("col:", col) # 0,1,2,3,4,5...2048
        with torch.autocast(device_type=device_type, enabled=False):
            # (B, D/2, T)
            freqs_x = (inv_x.to(x.device) @ row.to(x.device))
            freqs_y = (inv_y.to(x.device) @ col.to(x.device))
            # -> (B, T, D/2)
            freqs = (freqs_x + freqs_y).transpose(1, 2)

            # Duplicate to full head_dim: (B, T, D)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Keep parity with original: optional scaling hook
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling
        # print("cos:", cos) # almost all 1s
        # print("sin:", sin) # almost all 0s
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    # --- same helpers/signature as RoPESimple ---

    @staticmethod
    def apply_rotary_emb(
        q: Tensor,
        k: Tensor,
        cos: Tensor,
        sin: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        q,k: (B, H, T, D)
        cos,sin: (B, T, D)  or multi-modal variants handled below
        """
        if cos.dim() == 4:
            # shape [B, 3, T, D] -> multi-modal path (kept for drop-in parity)
            cos = RoPESimple2D._process_rotary_component(cos)
            sin = RoPESimple2D._process_rotary_component(sin)
        else:
            # text-only: (B, T, D) -> (B, 1, T, D)
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        q_embed = (q * cos) + (RoPESimple2D._rotate_half(q) * sin)
        k_embed = (k * cos) + (RoPESimple2D._rotate_half(k) * sin)
        return q_embed, k_embed

    @staticmethod
    def _process_rotary_component(x: Tensor) -> Tensor:
        """
        Keep the same multi-modal slicing trick as RoPESimple.
        """
        sections = x.split([16, 24, 24, 16, 24, 24], dim=-1)
        processed = [m[i % 3] for i, m in enumerate(sections)]
        return torch.cat(processed, dim=-1).unsqueeze(1)

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    
class RoPE(torch.nn.Module):
    """
    NOTE: This is a naive implementation. Use RoPESimple instead of this, if possible.

    Current does not support more advanced methods such as YaRN.
    """

    def __init__(self, theta: float, head_dim: int, max_seqlen: int = 1024):
        super().__init__()

        self.theta = theta
        self.head_dim = head_dim
        self.max_seqlen = max_seqlen

        self.register_buffer(
            "freqs_cis",
            self.precompute_freqs_cis(dim=head_dim, end=max_seqlen, theta=theta),
            persistent=False,
        )

    @staticmethod
    def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
        """
        Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

        This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
        and the end index 'end'. The 'theta' parameter scales the frequencies.
        The returned tensor contains complex values in complex64 data type.

        Args:
            dim (int): Dimension of the frequency tensor.
            end (int): End index for precomputing frequencies.
            theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

        Returns:
            torch.Tensor: Precomputed frequency tensor with complex exponentials.
        """
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device)
        freqs = torch.outer(t, freqs).float()

        cos, sin = freqs.cos(), freqs.sin()

        return torch.stack((cos, -sin, sin, cos), dim=-1).view(*freqs.size(), 2, 2)

    @staticmethod
    def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor, seq_dim: int):
        """
        Reshape frequency tensor for broadcasting it with another tensor.

        This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
        for the purpose of broadcasting the frequency tensor during element-wise operations.

        Args:
            freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
            x (torch.Tensor): Target tensor for broadcasting compatibility.
            seq_dim (int): Sequence dimension index.

        Returns:
            torch.Tensor: Reshaped frequency tensor.
        """
        ndim = x.ndim
        assert 0 <= seq_dim < ndim
        shape = [
            d if i == seq_dim or i == ndim - 3 else 1 for i, d in enumerate(x.shape[:-2])
        ] + [2, 2]
        return freqs_cis.view(*shape)

    @staticmethod
    def apply_rotary_emb(
        xq: torch.Tensor,  # (B S H D)
        xk: torch.Tensor,  # (B S H D)
        seq_dim: int,
        freqs_cis: torch.Tensor,  # (S D/2 2 2)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)  # B S H D -> B S H D/2 1 2
        xk_ = xk.reshape(*xk.shape[:-1], -1, 1, 2)  # B S H D -> B S H D/2 1 2
        freqs_cis = RoPE.reshape_for_broadcast(
            freqs_cis, xq_, seq_dim
        ).float()  # S D/2 2 2 -> 1 S 1 D/2 2 2
        xq_out = (xq_ * freqs_cis).sum(5).flatten(3)
        xk_out = (xk_ * freqs_cis).sum(5).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)

    def reset_parameters(self):
        self.freqs_cis[...] = RoPE.precompute_freqs_cis(
            dim=self.head_dim, end=self.max_seqlen, theta=self.theta
        )

    def forward(
        self,
        seqlen: Optional[int] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """
        Return freqs_cis corresponding to consecutive seqlen positions or the corresponding tok_idx positions
        Args:
            seqlen (int): Contiguous sequence length
            tok_idx (torch.Tensor[int]): Position indices of each token this overrides seqlen

        Returns:
            Tuple(torch.Tensor, torch.Tensor): Embedded input tensor and freqs_cis
        """
        test = (seqlen is not None) or (position_ids is not None)
        assert test, "Should provide atleast seqlen or tok_idx"
        if position_ids is not None:
            return self.freqs_cis[position_ids]
        elif seqlen is not None:
            return self.freqs_cis[0:seqlen]


class RoPESimple(nn.Module):
    '''
    Adopted from Qwen2.5.
    '''
    def __init__(self, head_dim, theta, device=None):
        super().__init__()
        self.dim = head_dim
        self.theta = theta

        inv_freq, self.attention_scaling = self._precompute_freqs_cis(
            theta=theta,
            head_dim=head_dim,
            device=device,
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _precompute_freqs_cis(self, theta, head_dim, device=None):
        base = theta
        partial_rotary_factor = 1.0
        dim = int(head_dim * partial_rotary_factor)

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
        return inv_freq, attention_factor

    def forward(self, x: Tensor, position_ids: Tensor) -> Tuple[Tensor, Tensor]:
        '''
        x: (B, T, D)
        position_ids: (B, T)
        '''
        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)  # (B, head_dim, 1)
        position_ids_expanded = position_ids[:, None, :].float()  # (B, 1, T)
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float().to(x.device) @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    @staticmethod
    def apply_rotary_emb(
        q: Tensor,
        k: Tensor,
        cos: Tensor,
        sin: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if cos.dim() == 4:
            # shape [B, 3, T, D] -> multi-modal
            cos = RoPESimple._process_rotary_component(cos)
            sin = RoPESimple._process_rotary_component(sin)
        else:
            # shape [B, T, D] -> text-only
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        q_embed = (q * cos) + (RoPESimple._rotate_half(q) * sin)
        k_embed = (k * cos) + (RoPESimple._rotate_half(k) * sin)
        return q_embed, k_embed

    @staticmethod
    def _process_rotary_component(x: Tensor) -> Tensor:
        '''
        Only used in multi-modal model.
        '''
        # Split into sections and select appropriate indices
        sections = x.split([16, 24, 24, 16, 24, 24], dim=-1)
        processed = [m[i % 3] for i, m in enumerate(sections)]
        # Combine and add dimension
        return torch.cat(processed, dim=-1).unsqueeze(1)

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="llama3",
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {
            "rope_type": rope_type,
            "factor": scaling_factor,
            "dim": dim,
            "base": base,
            "max_position_embeddings": max_position_embeddings,
        }
        self.rope_type = rope_type
        self.max_seq_len_cached = max_position_embeddings
        self.original_max_seq_len = max_position_embeddings
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(
            None, device, **self.rope_kwargs
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer(
                "inv_freq", inv_freq, persistent=False
            )  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if (
            seq_len < self.original_max_seq_len
            and self.max_seq_len_cached > self.original_max_seq_len
        ):  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
