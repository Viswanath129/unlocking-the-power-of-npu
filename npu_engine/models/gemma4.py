"""
Gemma 4 31B Architecture Definition & Tensor Mapping for Qualcomm Hexagon NPU
Optimized for 60-layer FastRPC layer streaming on Snapdragon X Plus (SC8380XP).
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


@dataclass
class Gemma4Config:
    """Gemma 4 31B Architectural Parameters."""
    model_name: str = "google/gemma-4-31B-it"
    num_hidden_layers: int = 60
    hidden_size: int = 5376
    intermediate_size: int = 21504
    num_attention_heads: int = 32
    num_key_value_heads: int = 16
    head_dim: int = 256
    global_head_dim: int = 512
    vocab_size: int = 262144
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    final_logit_softcapping: float = 30.0
    hidden_activation: str = "gelu_pytorch_tanh"
    tie_word_embeddings: bool = True
    
    # Hybrid attention layer configuration
    # Gemma 4 alternates sliding window attention and global full attention
    sliding_window_size: int = 4096
    full_attention_interval: int = 6  # Every 6th layer is full attention
    
    # Hexagon NPU Hardware Tiling Parameters (v73)
    crouton_tile_h: int = 64
    crouton_tile_w: int = 64
    vtcm_budget_bytes: int = 8 * 1024 * 1024  # 8MB VTCM SRAM
    hvx_vector_bytes: int = 128               # 1024-bit vector registers
    ping_pong_buffer_bytes: int = 480 * 1024 * 1024  # 480 MB per buffer in LPDDR5x

    def get_layer_type(self, layer_idx: int) -> str:
        """Returns sliding_attention or full_attention for a given layer."""
        if (layer_idx + 1) % self.full_attention_interval == 0:
            return "full_attention"
        return "sliding_attention"

    def get_layer_tensor_names(self, layer_idx: int) -> Dict[str, str]:
        """Maps logical tensor roles to exact Hugging Face safetensors tensor keys."""
        prefix = f"model.language_model.layers.{layer_idx}"
        return {
            "input_layernorm": f"{prefix}.input_layernorm.weight",
            "q_proj": f"{prefix}.self_attn.q_proj.weight",
            "q_norm": f"{prefix}.self_attn.q_norm.weight",
            "k_proj": f"{prefix}.self_attn.k_proj.weight",
            "k_norm": f"{prefix}.self_attn.k_norm.weight",
            "v_proj": f"{prefix}.self_attn.v_proj.weight",
            "o_proj": f"{prefix}.self_attn.o_proj.weight",
            "post_attention_layernorm": f"{prefix}.post_attention_layernorm.weight",
            "pre_feedforward_layernorm": f"{prefix}.pre_feedforward_layernorm.weight",
            "gate_proj": f"{prefix}.mlp.gate_proj.weight",
            "up_proj": f"{prefix}.mlp.up_proj.weight",
            "down_proj": f"{prefix}.mlp.down_proj.weight",
            "post_feedforward_layernorm": f"{prefix}.post_feedforward_layernorm.weight",
            "layer_scalar": f"{prefix}.layer_scalar",
        }

    def compute_layer_memory_bytes(self, quant_bits: int = 4) -> int:
        """
        Computes the memory footprint of a single Gemma 4 31B layer when quantized.
        - Q, K, V, O projections:
          - Q: 5376 x (32 * 256) = 5376 x 8192
          - K: 5376 x (16 * 256) = 5376 x 4096
          - V: 5376 x (16 * 256) = 5376 x 4096
          - O: (32 * 256) x 5376 = 8192 x 5376
          Total Attention Weights = 5376*(8192 + 4096 + 4096 + 8192) = 132,120,576 elements
        - MLP (SwiGLU / GeGLU):
          - Gate: 5376 x 21504 = 115,608,576 elements
          - Up: 5376 x 21504 = 115,608,576 elements
          - Down: 21504 x 5376 = 115,608,576 elements
          Total MLP Weights = 346,825,728 elements
        - Total layer elements: ~478,946,304 elements (~479M params / layer)
        - In INT4 (0.5 bytes/elem): ~239.5 MB per layer.
        - In INT8 (1 byte/elem): ~479 MB per layer.
        """
        attn_params = (
            self.hidden_size * (self.num_attention_heads * self.head_dim) +  # Q
            self.hidden_size * (self.num_key_value_heads * self.head_dim) +  # K
            self.hidden_size * (self.num_key_value_heads * self.head_dim) +  # V
            (self.num_attention_heads * self.head_dim) * self.hidden_size    # O
        )
        mlp_params = 3 * (self.hidden_size * self.intermediate_size)
        total_params = attn_params + mlp_params
        
        bytes_per_param = quant_bits / 8.0
        return int(total_params * bytes_per_param)
