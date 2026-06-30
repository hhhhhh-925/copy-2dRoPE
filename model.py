"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    
def _get_alibi_slopes(n_head: int) -> torch.Tensor:
    """
    Standard ALiBi slopes (as used in many reference implementations).
    Returns shape (n_head,).
    """
    def get_slopes_power_of_2(n: int):
        start = 2 ** (-2 ** -(math.log2(n) - 3))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]

    if math.log2(n_head).is_integer():
        slopes = get_slopes_power_of_2(n_head)
    else:
        closest_power_of_2 = 2 ** int(math.floor(math.log2(n_head)))
        slopes = get_slopes_power_of_2(closest_power_of_2)
        extra = _get_alibi_slopes(2 * closest_power_of_2)
        slopes += extra[0::2][: (n_head - closest_power_of_2)].tolist()
    return torch.tensor(slopes, dtype=torch.float32)


class CausalALiBiAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=getattr(config, "bias", True))
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=getattr(config, "bias", True))
        self.attn_dropout = nn.Dropout(getattr(config, "dropout", 0.0))
        self.resid_dropout = nn.Dropout(getattr(config, "dropout", 0.0))

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = getattr(config, "dropout", 0.0)

        self.flash = hasattr(F, "scaled_dot_product_attention")

        slopes = _get_alibi_slopes(self.n_head)  # (nh,)
        self.register_buffer("alibi_slopes", slopes, persistent=False)

    def forward(self, x, linebreak=None, idx=None, pos=None):
        B, T, C = x.size()
        nh = self.n_head
        hs = C // nh

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, nh, hs).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, nh, hs).transpose(1, 2)
        v = v.view(B, T, nh, hs).transpose(1, 2)

        if self.flash:
            device, dtype = q.device, q.dtype

            # IMPORTANT: flash SDPA usually requires fp16/bf16 on CUDA for speed+memory.
            # If you're in fp32, expect fallback + huge memory.
            # So make sure your model/forward is under autocast or q is half/bf16.

            pos = torch.arange(T, device=device, dtype=dtype)  # (T,)
            slopes = self.alibi_slopes.to(device=device, dtype=dtype).view(1, self.n_head, 1)  # (1,nh,1)

            # --- keep head_dim = hs, but repurpose last dim for ALiBi ---
            # scale the remaining hs-1 dims to roughly preserve dot-product magnitude
            # since we "remove" one content dim
            if hs > 1:
                gamma = math.sqrt(hs / (hs - 1))
                q_main = q[..., :-1] * gamma
                k_main = k[..., :-1] * gamma
            else:
                # hs==1 is pathological; can't repurpose
                q_main = q
                k_main = k

            # bias carrier dim
            q_last = torch.ones((B, self.n_head, T, 1), device=device, dtype=dtype)
            k_last = (slopes * pos.view(1, 1, T) * math.sqrt(hs)).expand(B, -1, -1).unsqueeze(-1)

            # rebuild q,k with same head_dim (hs)
            q_ = torch.cat([q_main, q_last], dim=-1) if hs > 1 else q_last
            k_ = torch.cat([k_main, k_last], dim=-1) if hs > 1 else k_last

            y = torch.nn.functional.scaled_dot_product_attention(
                q_, k_, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )

        else:
            # fallback: still avoid (T,T) rel matrix; use key-index bias (1,nh,1,T)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))  # (B,nh,T,T)

            causal = torch.triu(torch.ones(T, T, device=att.device, dtype=torch.bool), diagonal=1)
            att = att.masked_fill(causal.view(1, 1, T, T), float("-inf"))

            pos = torch.arange(T, device=att.device, dtype=att.dtype)  # (T,)
            slopes = self.alibi_slopes.to(device=att.device, dtype=att.dtype).view(1, nh, 1, 1)
            att = att + slopes * pos.view(1, 1, 1, T)

            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B,nh,T,hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class CausalFIREAttention(nn.Module):
    """
    FIRE (here): learnable per-head linear relative-position bias.
    Implemented with the same "bias-carrier dim" trick as your ALiBi code:
    - NO attn_mask passed to SDPA (keeps flash causal kernel)
    - Inject bias through q/k last dimension so head_dim stays hs
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=getattr(config, "bias", True))
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=getattr(config, "bias", True))
        self.attn_dropout = nn.Dropout(getattr(config, "dropout", 0.0))
        self.resid_dropout = nn.Dropout(getattr(config, "dropout", 0.0))

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = getattr(config, "dropout", 0.0)
        self.block_size = getattr(config, "block_size", None)

        self.flash = hasattr(F, "scaled_dot_product_attention")

        # FIRE slopes: learnable (nh,)
        # init small so it's near NoPE initially
        self.fire_slopes = nn.Parameter(torch.zeros(self.n_head))
        self.fire_scale  = nn.Parameter(torch.tensor(1.0))

        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            assert self.block_size is not None
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(self.block_size, self.block_size)).view(1, 1, self.block_size, self.block_size),
            )

    def forward(self, x, linebreak=None):
        B, T, C = x.size()
        nh = self.n_head
        hs = C // nh

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, nh, hs).transpose(1, 2)  # (B,nh,T,hs)
        k = k.view(B, T, nh, hs).transpose(1, 2)
        v = v.view(B, T, nh, hs).transpose(1, 2)

        if self.flash:
            device, dtype = q.device, q.dtype

            # positions
            pos = torch.arange(T, device=device, dtype=dtype)  # (T,)
            slopes = (self.fire_scale.to(dtype) * self.fire_slopes.to(dtype)).view(1, nh, 1)  # (1,nh,1)

            # same trick: use last dim to carry bias; keep head_dim = hs
            if hs > 1:
                gamma = math.sqrt(hs / (hs - 1))
                q_main = q[..., :-1] * gamma
                k_main = k[..., :-1] * gamma
            else:
                # hs==1 pathological; cannot keep content; only bias carrier
                q_main = None
                k_main = None

            # bias carrier dim:
            # q_last = 1
            # k_last = slopes * pos * sqrt(hs)
            q_last = torch.ones((B, nh, T, 1), device=device, dtype=dtype)
            k_last = (slopes * pos.view(1, 1, T) * math.sqrt(hs)).expand(B, -1, -1).unsqueeze(-1)

            if hs > 1:
                q_ = torch.cat([q_main, q_last], dim=-1)
                k_ = torch.cat([k_main, k_last], dim=-1)
            else:
                q_ = q_last
                k_ = k_last

            y = F.scaled_dot_product_attention(
                q_, k_, v,
                attn_mask=None,  # <-- crucial: keep flash kernel
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )

        else:
            # slow fallback (will allocate (B,nh,T,T) regardless)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))  # (B,nh,T,T)
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

            pos = torch.arange(T, device=att.device, dtype=att.dtype)  # (T,)
            slopes = (self.fire_scale.to(att.dtype) * self.fire_slopes.to(att.dtype)).view(1, nh, 1, 1)
            att = att + slopes * pos.view(1, 1, 1, T)  # add slope * j

            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B,nh,T,hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class CausalNoPEAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=getattr(config, "bias", True))
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=getattr(config, "bias", True))

        # regularization
        self.attn_dropout = nn.Dropout(getattr(config, "dropout", 0.0))
        self.resid_dropout = nn.Dropout(getattr(config, "dropout", 0.0))

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = getattr(config, "dropout", 0.0)

        # flash attention (PyTorch >= 2.0)
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")

        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure attention is only applied to the left
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x, linebreak=None, idx=None, pos=None):
        B, T, C = x.size()  # (batch, time, channels)

        # compute q, k, v
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        hs = C // self.n_head
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))  # (B, nh, T, T)
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class CausalRoPEAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.trainable_freqs = config.trainable_freqs
        head_dim = config.n_embd // config.n_head
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        if self.trainable_freqs:
            self.inv_freq = nn.Parameter(inv_freq)
        else:
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
            
    def apply_rope(self, x):
        # x: (B, nh, T, hs)
        B, nh, T, hs = x.size()
        device = x.device

        t = torch.arange(T, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)  # (T, hs/2)
        cos = freqs.cos()[None, None, :, :]  # (1, 1, T, hs/2)
        sin = freqs.sin()[None, None, :, :]  # (1, 1, T, hs/2)
        x1, x2 = x.chunk(2, dim=-1)  # (B, nh, T, hs/2), (B, nh, T, hs/2)
        x = torch.cat([x1 * cos - x2 * sin,
                       x1 * sin + x2 * cos], dim=-1)  # (B, nh, T, hs)
        return x

    def forward(self, x, linebreak=None, idx=None, pos=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # apply RoPE embeddings to q and k
        q = self.apply_rope(q)
        k = self.apply_rope(k)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    
class CausalAdaptive2DRoPEAttention(nn.Module):
    """
    Adaptive 2D RoPE attention.

    Main idea:
    1. From the layer input x_t, predict two scalars A_t, B_t
    2. Update a continuous 2D position state:
           c_{t+1} = A_t * c_t + B_t
           r_{t+1} = r_t + (1 - A_t)
    3. Assign token t the position (r_t, c_t)
    4. Apply 2D RoPE to q,k using these positions
    5. Run ordinary causal self-attention
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.trainable_freqs = getattr(config, "trainable_freqs", False)

        head_dim = config.n_embd // config.n_head
        assert head_dim % 4 == 0, "For 2D RoPE, head_dim should be divisible by 4."

        self.head_dim = head_dim
        self.axis_dim = head_dim // 2   # half for row, half for col

        # qkv projection
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # FlashAttention support
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                    .view(1, 1, config.block_size, config.block_size)
            )

        # ===== Adaptive 2D position generator =====
        # Predict (a_t, b_t), then:
        #   A_t = alpha * sigmoid(a_t)
        #   B_t = alpha * sigmoid(b_t)
        self.ab_proj = nn.Linear(config.n_embd, 2, bias=True)
        self.alpha = float(getattr(config, "adap_2d_rope_alpha", 1.0))

        # zero init => sigmoid(0)=0.5 => if alpha=2, initially A=B=1
        # then c_{t+1}=c_t+1, r_{t+1}=r_t, i.e. starts as a stable 1D-like path
        nn.init.zeros_(self.ab_proj.weight)
        nn.init.zeros_(self.ab_proj.bias)

        # ===== 2D RoPE frequencies =====
        # row uses first half of head_dim, col uses second half
        theta_row = float(getattr(config, "rope_theta_row", 10000.0))
        theta_col = float(getattr(config, "rope_theta_col", 10000.0))

        inv_freq_row = 1.0 / (
            theta_row ** (torch.arange(0, self.axis_dim, 2).float() / self.axis_dim)
        )
        inv_freq_col = 1.0 / (
            theta_col ** (torch.arange(0, self.axis_dim, 2).float() / self.axis_dim)
        )

        if self.trainable_freqs:
            self.inv_freq_row = nn.Parameter(inv_freq_row)
            self.inv_freq_col = nn.Parameter(inv_freq_col)
        else:
            self.register_buffer("inv_freq_row", inv_freq_row, persistent=False)
            self.register_buffer("inv_freq_col", inv_freq_col, persistent=False)

    def _affine_scan_inclusive(self, scale: torch.Tensor, shift: torch.Tensor):
        """
        Inclusive prefix scan of affine maps x -> scale * x + shift.

        Inputs:
            scale, shift: (B, T)

        Returns:
            prefix_scale, prefix_shift: (B, T)
            such that for each t,
                prefix_t(x) = prefix_scale[:, t] * x + prefix_shift[:, t]
            equals the composition up to t.
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


    def compute_adaptive_2d_positions(self, x, pos0=None):
        """
        x:    (B, T, C)
        pos0: (B, 2), optional starting position [row, col]

        Returns:
            pos:     (B, T, 2), float32, [row, col]
            new_pos: (B, 2),    float32, final state after the chunk
            A:       (B, T)
            Bv:      (B, T)
        """
        B, T, _ = x.size()
        device = x.device
        work_dtype = torch.float32

        if pos0 is None:
            row0 = torch.zeros(B, device=device, dtype=work_dtype)
            col0 = torch.zeros(B, device=device, dtype=work_dtype)
        else:
            pos0 = pos0.to(device=device, dtype=work_dtype)
            row0 = pos0[:, 0]
            col0 = pos0[:, 1]

        # (B, T, 2)
        ab = self.alpha * torch.sigmoid(self.ab_proj(x).to(dtype=work_dtype))
        A = ab[..., 0]   # (B, T)
        Bv = ab[..., 1]  # (B, T)

        # Reuse these instead of allocating multiple times
        zeros_col = torch.zeros(B, 1, device=device, dtype=work_dtype)
        ones_col = torch.ones(B, 1, device=device, dtype=work_dtype)

        # -------------------------------------------------
        # Column:
        #   col_{t+1} = A_t * col_t + Bv_t
        #
        # Token t receives col_t, so we need an EXCLUSIVE scan.
        # Build exclusive transforms by shifting right and
        # prepending identity.
        # -------------------------------------------------
        col_scale_excl = torch.cat([ones_col, A[:, :-1]], dim=1)   # (B, T)
        col_shift_excl = torch.cat([zeros_col, Bv[:, :-1]], dim=1) # (B, T)

        col_prefix_scale, col_prefix_shift = self._affine_scan_inclusive(
            col_scale_excl, col_shift_excl
        )
        col = col_prefix_scale * col0[:, None] + col_prefix_shift   # (B, T)

        # Final column state after the chunk:
        # new_col = A_{T-1} * col_{T-1} + Bv_{T-1}
        new_col = A[:, -1] * col[:, -1] + Bv[:, -1]

        # -------------------------------------------------
        # Row:
        #   row_{t+1} = row_t + (1 - A_t)
        #
        # Token t receives row_t, so again exclusive prefix sum.
        # -------------------------------------------------
        row_delta = 1.0 - A
        row = row0[:, None] + torch.cat(
            [zeros_col, torch.cumsum(row_delta[:, :-1], dim=1)],
            dim=1,
        )
        new_row = row0 + row_delta.sum(dim=1)

        pos = torch.stack([row, col], dim=-1)       # (B, T, 2)
        new_pos = torch.stack([new_row, new_col], dim=-1)  # (B, 2)

        return pos, new_pos, A, Bv

    def apply_2d_rope(self, x, pos):
        """
        x:   (B, nh, T, hs)
        pos: (B, T, 2), [row, col], float

        We split the head dimension into two halves:
            first half  -> row rotation
            second half -> col rotation
        """
        B, nh, T, hs = x.size()
        assert hs == self.head_dim

        row_pos = pos[..., 0]   # (B, T)
        col_pos = pos[..., 1]   # (B, T)

        x_row = x[..., :self.axis_dim]
        x_col = x[..., self.axis_dim:]

        # frequencies: (B, 1, T, axis_dim/2)
        freqs_row = row_pos[:, None, :, None] * self.inv_freq_row[None, None, None, :]
        freqs_col = col_pos[:, None, :, None] * self.inv_freq_col[None, None, None, :]

        cos_row = freqs_row.cos().to(dtype=x.dtype)
        sin_row = freqs_row.sin().to(dtype=x.dtype)
        cos_col = freqs_col.cos().to(dtype=x.dtype)
        sin_col = freqs_col.sin().to(dtype=x.dtype)

        # rotate row half
        x_row_1, x_row_2 = x_row.chunk(2, dim=-1)
        x_row_rot = torch.cat(
            [
                x_row_1 * cos_row - x_row_2 * sin_row,
                x_row_1 * sin_row + x_row_2 * cos_row,
            ],
            dim=-1,
        )

        # rotate col half
        x_col_1, x_col_2 = x_col.chunk(2, dim=-1)
        x_col_rot = torch.cat(
            [
                x_col_1 * cos_col - x_col_2 * sin_col,
                x_col_1 * sin_col + x_col_2 * cos_col,
            ],
            dim=-1,
        )

        return torch.cat([x_row_rot, x_col_rot], dim=-1)

    def forward(self, x, linebreak=None, idx=None, pos=None, pos0=None):
        """
        x:   (B, T, C)
        pos: optional externally provided 2D positions, shape (B, T, 2)

        If pos is None, we generate adaptive positions from x itself.
        pos0 is an optional initial [row, col] state for the whole sequence chunk.
        """
        B, T, C = x.size()

        # q, k, v
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # adaptive 2D positions
        if pos is None:
            pos, new_pos, A, Bv = self.compute_adaptive_2d_positions(x, pos0=pos0)
        else:
            pos = pos.to(device=x.device, dtype=torch.float32)
            new_pos, A, Bv = None, None, None

        # 2D RoPE on q,k
        q = self.apply_2d_rope(q, pos)
        k = self.apply_2d_rope(k, pos)

        # causal self-attention
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        # if you want to inspect the learned positions, return them here
        # return y, pos, new_pos, A, Bv
        return y
    
class CausalRePoAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn_qk = nn.Linear(config.n_embd, 2 * config.n_embd, bias=config.bias)
        self.c_attn_v = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_attn_pos = nn.Linear(128, config.n_head, bias=config.bias)
        self.pos_v_compute = nn.Linear(128, 128 * config.n_head, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.pos_proj = nn.Linear(128 * config.n_head, 128, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.trainable_freqs = config.trainable_freqs
        head_dim = config.n_embd // config.n_head
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        
        self.repo_dim = config.n_embd // 8
        self.repo_gate = nn.Linear(config.n_embd, self.repo_dim, bias=False)
        self.repo_content = nn.Linear(config.n_embd, self.repo_dim, bias=False)
        self.repo_out = nn.Linear(self.repo_dim, self.n_head, bias=False)
        
        if self.trainable_freqs:
            self.inv_freq = nn.Parameter(inv_freq)
        else:
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
    
    def compute_repo_positions(self, hidden_states):
        """
        hidden_states: (B, T, C)

        RePo:
            r = Swish(h Wg) ⊙ (h Wc)          # (B, T, repo_dim)
            z = r Wz                          # (B, T, n_head)
            return z.transpose(1, 2)          # (B, n_head, T)
        """
        r = F.silu(self.repo_gate(hidden_states)) * self.repo_content(hidden_states)
        z = self.repo_out(r)                  # (B, T, n_head)
        z = z.transpose(1, 2).contiguous()    # (B, n_head, T)
        return z
    
    def _build_rope_cos_sin(self, positions: torch.Tensor):
        """
        positions:
            - standard RoPE: (T,)
            - RePo per-head positions: (B, nh, T)
            - optional shared dynamic positions: (B, T)

        returns:
            cos, sin with shape matching x1/x2 in apply_rope:
            - standard: (1, 1, T, hs/2)
            - per-head: (B, nh, T, hs/2)
            - shared-dynamic: (B, 1, T, hs/2)
        """
        inv_freq = self.inv_freq.to(device=positions.device, dtype=torch.float32)

        if positions.dim() == 1:
            # (T,) -> (1,1,T,hs/2)
            freqs = torch.einsum("t,d->td", positions.float(), inv_freq)
            cos = freqs.cos()[None, None, :, :]
            sin = freqs.sin()[None, None, :, :]
        elif positions.dim() == 2:
            # (B,T) -> (B,1,T,hs/2)
            freqs = positions.float().unsqueeze(-1) * inv_freq.view(1, 1, -1)
            cos = freqs.cos().unsqueeze(1)
            sin = freqs.sin().unsqueeze(1)
        elif positions.dim() == 3:
            # (B,nh,T) -> (B,nh,T,hs/2)
            freqs = positions.float().unsqueeze(-1) * inv_freq.view(1, 1, 1, -1)
            cos = freqs.cos()
            sin = freqs.sin()
        else:
            raise ValueError(f"Unexpected positions shape: {positions.shape}")

        return cos, sin
            
    def apply_rope(self, x, positions=None):
        # x: (B, nh, T, hs)
        B, nh, T, hs = x.size()

        cos, sin = self._build_rope_cos_sin(positions)
        cos = cos.to(dtype=x.dtype, device=x.device)
        sin = sin.to(dtype=x.dtype, device=x.device)

        x1, x2 = x.chunk(2, dim=-1)
        out = torch.cat(
            [
                x1 * cos - x2 * sin,
                x1 * sin + x2 * cos,
            ],
            dim=-1,
        )
        return out

    def forward(self, x, linebreak=None, idx=None, pos=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k  = self.c_attn_qk(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.c_attn_v(x)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # apply RoPE embeddings to q and k
        repo_pos = self.c_attn_pos(pos)      # (B, T, n_head)
        # print(pos[0,:,0])
        # print("repo_pos")
        # print(repo_pos[0,:,1])
        # print(repo_pos[0,:,:100])
        repo_pos = repo_pos.permute(0, 2, 1)      # (B, n_head, T)
        #repo_pos = self.compute_repo_positions(x)   # (B, nh, T)
        q = self.apply_rope(q, positions=repo_pos)
        k = self.apply_rope(k, positions=repo_pos)
        # repo_pos: (B, nh, T)
        v_pos = self.pos_v_compute(pos).view(B, T, self.n_head, 128).transpose(1, 2)
        
        test_in = torch.tensor([1.0] * 128, device=self.c_attn_pos.weight.device)
        # print("debug")
        # print(self.c_attn_pos(test_in))
        # print(v_pos[0,0,:,0])

        # concatenate to value features
        v_cat = torch.cat([v, v_pos], dim=-1)   # (B, nh, T, hs+16)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            #y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
            y_cat = torch.nn.functional.scaled_dot_product_attention(q, k, v_cat, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            
        y_v = y_cat[..., :v.size(-1)]                   # (B, nh, T, hs)
        y_repo_pos = y_cat[..., v.size(-1):]           # (B, nh, T, 16)

        y = y_v.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y_repo_pos = y_repo_pos.transpose(1, 2).contiguous().view(B, T, 128 * self.n_head) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        y_repo_pos = self.pos_proj(y_repo_pos)
        return y, y_repo_pos
    
class Causal2DRoPEAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.attn_probs = None
        self.trainable_freqs = config.trainable_freqs
        # use 2D rope with trainable parameters
        head_dim = config.n_embd // config.n_head
        base = torch.arange(0, head_dim, 4).float() / head_dim
        self.pad_right = config.pad_right
        if self.trainable_freqs:
            init_freq_x = 1.0 / (100 ** (torch.arange(0, head_dim, 2).float() / head_dim))
            init_freq_y = 1.0 / (100 ** (torch.arange(0, head_dim, 2).float() / head_dim))
            self.freq_x = nn.Parameter(init_freq_x.repeat(config.n_head))
            self.freq_y = nn.Parameter(init_freq_y.repeat(config.n_head))
        else:
            freq_y = 1.0 / (100 ** base)
            freq_x = 1.0 / (100 ** base)
            pad_len = (head_dim // 2) - base.numel()
            init_freq_x = torch.cat([freq_x, torch.zeros(pad_len)], dim=0)
            init_freq_y = torch.cat([torch.zeros(pad_len), freq_y], dim=0)
            freq_x = init_freq_x.repeat(self.n_head).to(dtype=self.c_attn.weight.dtype)
            freq_y = init_freq_y.repeat(self.n_head).to(dtype=self.c_attn.weight.dtype)
            self.register_buffer("freq_x", freq_x, persistent=False)
            self.register_buffer("freq_y", freq_y, persistent=False)
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
            
    def apply_rope(self, q, k, pos):
        head_dim = self.n_embd // self.n_head
        B, nh, T, hd = q.size()
        idx_base = torch.arange(T)[None, :].to(q.device)  # (1, T)
        pos = pos[:, None]
        idx_x = torch.where(idx_base < pos - self.pad_right + 2, idx_base.expand(B, T), idx_base.expand(B, T) - pos.expand(B, T) + self.pad_right - 2)[:, None, :, None] # (B, T)
        idx_y = torch.where(idx_base < pos - self.pad_right + 2, 0, 1)[:, None, :, None] # (B, T)
        fx = self.freq_x.view(1, nh, 1, hd // 2)
        fy = self.freq_y.view(1, nh, 1, hd // 2)
        freq = idx_x * fx + idx_y * fy  # (B, nh, T, hd/2)
        cos = torch.cat([freq.cos(), freq.cos()], dim=-1)  # (B, nh, T, hd)
        sin = torch.cat([freq.sin(), freq.sin()], dim=-1)  # (B, nh, T, hd)
        q_rot = torch.cat([-q[..., head_dim//2:], q[..., :head_dim//2]], dim=-1)
        k_rot = torch.cat([-k[..., head_dim//2:], k[..., :head_dim//2]], dim=-1)
        q = q * cos + q_rot * sin
        k = k * cos + k_rot * sin
        return q, k

    def forward(self, x, linebreak, idx=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q, k = self.apply_rope(q, k, linebreak)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)

            self.attn_probs = att.detach().cpu()
            #att = F.softmax(att, dim=-1)
            self.attn_q = q.detach().cpu()
            self.attn_k = k.detach().cpu()
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            self.attn_y = y.detach().cpu()
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
class GatedMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_gate  = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.silu    = nn.SiLU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x_fc = self.c_fc(x)
        x_gate = self.c_gate(x)
        x = self.silu(x_fc) * x_gate # use SiLU activation for gating
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config, idx=None):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        if config.pe_type == 'rope':
            self.attn = CausalRoPEAttention(config)
        elif config.pe_type == '2drope':
            self.attn = Causal2DRoPEAttention(config)
        elif config.pe_type == 'nope':
            self.attn = CausalNoPEAttention(config)
        elif config.pe_type == 'alibi':
            self.attn = CausalALiBiAttention(config)
        elif config.pe_type == 'hrope':
            if idx % 2 == 0:
                self.attn = CausalRoPEAttention(config)
            else:
                self.attn = Causal2DRoPEAttention(config)
        elif config.pe_type == 'nrope':
            if idx % 2 == 0:
                self.attn = CausalRoPEAttention(config)
            else:
                self.attn = CausalNoPEAttention(config)
        elif config.pe_type == 'fire':
            self.attn = CausalFIREAttention(config)
        elif config.pe_type == 'repo':
            # if idx <= (config.n_layer // 3):
            #     print(f"idx={idx},layer={config.n_layer}")
            #     self.attn = CausalRoPEAttention(config)
            # else:
            #     print(f"idx={idx},layer={config.n_layer}")
            self.attn = CausalRePoAttention(config)
        elif config.pe_type == 'ada2d':
            self.attn = CausalAdaptive2DRoPEAttention(config)
        else:
            raise ValueError(f"Unknown pe_type: {config.pe_type}")
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        if config.gated:
            self.mlp = GatedMLP(config)
        else:
            self.mlp = MLP(config)
        self.idx = idx
        self.petype = config.pe_type

    def forward(self, x, linebreak=None, vpos=None):
        # print(self.idx)
        if self.petype == 'repo':
            delta_x, delta_pos = self.attn(self.ln_1(x), linebreak=linebreak, idx=self.idx, pos=vpos)
            x = x + delta_x
            x = x + self.mlp(self.ln_2(x))
            vpos = vpos - delta_pos
            return x, vpos
        else:
            delta_x = self.attn(self.ln_1(x), linebreak=linebreak, idx=self.idx)
            delta_pos = 0
            x = x + delta_x
            x = x + self.mlp(self.ln_2(x))
            return x, vpos

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    pe_type: str = "rope"
    gated: bool = True
    trainable_freqs: bool = True
    pad_right: int = 4

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config, idx=i) for i in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, linebreak=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        x = self.transformer.drop(tok_emb)
        B, T, C = x.size()
        pos = torch.arange(1, T + 1, device=x.device, dtype=x.dtype).view(1, T, 1)
        pos = pos.expand(B, T, 128)
        for block in self.transformer.h:
            x, pos = block(x, linebreak=linebreak, vpos=pos)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        # optimizer = torch.optim.SGD(
        #     optim_groups,
        #     lr=learning_rate,
        #     momentum=0.9,
        #     nesterov=False,   # optional, can set False if you don't want Nesterov
        # )

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx