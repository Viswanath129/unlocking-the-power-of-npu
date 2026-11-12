"""
Qualcomm Hexagon NPU Template Decoder Layer Engine
Implements one reusable Transformer Decoder Layer executing on Hexagon v73 HMX/HVX.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from .config import NPUModelConfig, NPUQuantMode


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """RMSNorm kernel corresponding to Hexagon HVX vector reduction instruction"""
    variance = np.mean(np.square(x), axis=-1, keepdims=True)
    return x * (1.0 / np.sqrt(variance + eps)) * weight


def apply_rotary_pos_emb(q: np.ndarray, k: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """RoPE kernel for attention queries and keys"""
    def rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return np.concatenate((-x2, x1), axis=-1)

    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU / Swish activation corresponding to Hexagon HVX vector polynomial kernel"""
    return x / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def dequantize_crouton_int8(crouton_weights: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Reconstructs 2D linear weight from Crouton 4D tiles for computation"""
    if crouton_weights.ndim == 4:
        t = crouton_weights.transpose(0, 2, 1, 3)
        n_out = t.shape[0] * t.shape[1]
        n_in = t.shape[2] * t.shape[3]
        flat_w = t.reshape(n_out, n_in)
        return flat_w[:len(scales), :] * scales[:, None]
    return crouton_weights * scales[:, None]


def dequantize_int4(packed_weights: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Unpacks nibble weights [out_features, in_features // 2] to float32"""
    low = (packed_weights & 0x0F).astype(np.int8)
    high = ((packed_weights >> 4) & 0x0F).astype(np.int8)
    low = np.where(low >= 8, low - 16, low)
    high = np.where(high >= 8, high - 16, high)
    
    out_f, in_half = packed_weights.shape
    unpacked = np.empty((out_f, in_half * 2), dtype=np.float32)
    unpacked[:, 0::2] = low
    unpacked[:, 1::2] = high
    return unpacked * scales[:, None]


class NPUTemplateDecoderLayer:
    """
    The Single-Layer Template instantiated once on the Qualcomm NPU.
    All layers of the model sequentially reuse this single computational core.
    """

    def __init__(self, config: NPUModelConfig):
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.eps = config.rms_norm_eps

    def _linear(self, x: np.ndarray, weight: np.ndarray, scales: Optional[np.ndarray], quant_mode: str, bias: Optional[np.ndarray] = None) -> np.ndarray:
        """Matrix-vector multiplication accelerated by HMX systolic array"""
        if quant_mode == NPUQuantMode.INT4.value and scales is not None:
            w = dequantize_int4(weight, scales)
        elif quant_mode == NPUQuantMode.INT8.value and scales is not None:
            w = dequantize_crouton_int8(weight, scales)
        else:
            w = weight

        res = np.matmul(x, w.T)
        if bias is not None:
            res += bias
        return res

    def forward(
        self,
        hidden_states: np.ndarray,
        layer_weights: Dict[str, Any],
        kv_cache_k: Optional[np.ndarray],
        kv_cache_v: Optional[np.ndarray],
        cos_sin: Tuple[np.ndarray, np.ndarray],
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        tensors = layer_weights["tensors"]
        scales = layer_weights.get("scales", {})
        quant_mode = layer_weights.get("quant_mode", NPUQuantMode.INT8.value)

        batch_size, seq_len, _ = hidden_states.shape
        residual = hidden_states

        # 1. Input LayerNorm (HVX)
        norm_w = tensors.get("input_layernorm.weight", np.ones(self.hidden_size, dtype=np.float32))
        norm_x = rms_norm(hidden_states, norm_w, self.eps)

        # 2. Q, K, V Projections (HMX systolic array)
        q_bias = tensors.get("self_attn.q_proj.bias")
        k_bias = tensors.get("self_attn.k_proj.bias")
        v_bias = tensors.get("self_attn.v_proj.bias")

        q = self._linear(norm_x, tensors["self_attn.q_proj.weight"], scales.get("self_attn.q_proj.weight"), quant_mode, q_bias)
        k = self._linear(norm_x, tensors["self_attn.k_proj.weight"], scales.get("self_attn.k_proj.weight"), quant_mode, k_bias)
        v = self._linear(norm_x, tensors["self_attn.v_proj.weight"], scales.get("self_attn.v_proj.weight"), quant_mode, v_bias)

        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # 3. RoPE (HVX)
        cos, sin = cos_sin
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 4. KV Cache Update
        if kv_cache_k is not None and kv_cache_v is not None and kv_cache_k.shape[2] > 0:
            active_k = np.concatenate([kv_cache_k, k], axis=2)
            active_v = np.concatenate([kv_cache_v, v], axis=2)
        else:
            active_k = k
            active_v = v

        # Repeat KV heads for Grouped-Query Attention (GQA)
        if self.num_kv_groups > 1:
            k_expanded = np.repeat(active_k, self.num_kv_groups, axis=1)
            v_expanded = np.repeat(active_v, self.num_kv_groups, axis=1)
        else:
            k_expanded = active_k
            v_expanded = active_v

        # Scaled Dot-Product Attention
        scale_factor = 1.0 / np.sqrt(self.head_dim)
        scores = np.matmul(q, k_expanded.transpose(0, 1, 3, 2)) * scale_factor

        if mask is not None:
            scores = scores + mask

        # Softmax (HVX)
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        # Context output
        context = np.matmul(attn_weights, v_expanded)
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.hidden_size)

        # Out projection + Residual Add
        attn_out = self._linear(context, tensors["self_attn.o_proj.weight"], scales.get("self_attn.o_proj.weight"), quant_mode)
        hidden_states = residual + attn_out

        # 6. Post-Attention RMSNorm (HVX)
        residual = hidden_states
        post_norm_w = tensors.get("post_attention_layernorm.weight", np.ones(self.hidden_size, dtype=np.float32))
        norm_x = rms_norm(hidden_states, post_norm_w, self.eps)

        # 7. SwiGLU MLP (HMX systolic array)
        gate = self._linear(norm_x, tensors["mlp.gate_proj.weight"], scales.get("mlp.gate_proj.weight"), quant_mode)
        up = self._linear(norm_x, tensors["mlp.up_proj.weight"], scales.get("mlp.up_proj.weight"), quant_mode)
        activated = silu(gate) * up
        down = self._linear(activated, tensors["mlp.down_proj.weight"], scales.get("mlp.down_proj.weight"), quant_mode)

        # 8. Residual Add (HVX)
        hidden_states = residual + down

        return hidden_states, k, v
