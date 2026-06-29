from typing import Optional
from transformers.configuration_utils import PretrainedConfig


class GPTConfig(PretrainedConfig):
    def __init__(self,
        max_len: int = 512,
        num_hidden_layers: int = 12,
        hidden_size: int = 768,
        initializer_range: float = 0.02,

        # Vocab
        vocab_size: int = (
            50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
        ),
        tie_word_embeddings: bool = False,
        use_cce_loss: bool = False,

        # Attention
        num_attention_heads: int = 12,
        num_key_value_heads: int = 12,
        dim_k: int = 64,
        dim_v: int = 64,
        att_q_norm: bool = False,
        att_k_norm: bool = False,
        att_tie_kv: bool = False,
        att_window_size: int = -1,  # -1 means we don't use sliding window
        attn_bias: bool = False,
        attn_output_bias: bool = False,

        # Positional Embedding
        pos_emb_type: str = 'rope_simple_2d',  # "rope_simple" is the RoPE used by Qwen.
        rope_theta_x: int = 100,
        rope_theta_y: int = 100,
        rope_theta: int = 10000,
      
        # FFN
        ffn_bias: bool = False,
        ffn_per_layer: int = 1,
        ffn_act_fn: str = "silu",
        ffn_is_gated: bool = True,
        ffn_d_mid: Optional[int] = None,

        # RMSNorm
        norm_eps: float = 1e-6,

        # deprecated
        dropout: float = 0.0,
        
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs,
    ):
        self.max_len = max_len
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.vocab_size = vocab_size
        self.tie_word_embeddings = tie_word_embeddings
        self.use_cce_loss = use_cce_loss
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.att_q_norm = att_q_norm
        self.att_k_norm = att_k_norm
        self.att_tie_kv = att_tie_kv
        self.att_window_size = att_window_size
        self.attn_bias = attn_bias
        self.attn_output_bias = attn_output_bias
        self.pos_emb_type = pos_emb_type
        self.rope_theta_x = rope_theta_x
        self.rope_theta_y = rope_theta_y
        self.rope_theta = rope_theta
        self.ffn_bias = ffn_bias
        self.ffn_per_layer = ffn_per_layer
        self.ffn_act_fn = ffn_act_fn
        self.ffn_is_gated = ffn_is_gated
        self.ffn_d_mid = ffn_d_mid
        self.norm_eps = norm_eps
        self.dropout = dropout


        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
