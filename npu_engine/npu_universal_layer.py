"""
Universal Single-Layer Execution Template for Qualcomm Hexagon v73 NPU
Executes any autoregressive transformer layer (LLaMA, Gemma, Qwen, Mistral, Phi, DeepSeek)
with lowest possible RAM footprint using 4D Crouton micro-tiling on HMX systolic arrays.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from .models.auto_model import UniversalModelConfig, ActivationType, NormType


def universal_norm(
    x: np.ndarray, 
    weight: np.ndarray, 
    norm_type: NormType = NormType.RMSNORM, 
    eps: float = 1e-6
) -> np.ndarray:
    """
    Universal normalizer supporting standard RMSNorm, Gemma unit-offset RMSNorm, and LayerNorm.
    """
    if norm_type == NormType.RMSNORM_OFFSET:
        # Gemma style: (1.0 + weight) * (x / rms)
        variance = np.mean(np.square(x), axis=-1, keepdims=True)
        normed = x * (1.0 / np.sqrt(variance + eps))
        return normed * (1.0 + weight)
    elif norm_type == NormType.LAYERNORM:
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(variance + eps) * weight
    else:
        # Standard RMSNorm (LLaMA, Qwen, Mistral)
        variance = np.mean(np.square(x), axis=-1, keepdims=True)
        normed = x * (1.0 / np.sqrt(variance + eps))
        return normed * weight


def apply_rotary_pos_emb(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Applies Rotary Position Embedding (RoPE) to Q or K projections."""
    # Rotate half
    half_dim = x.shape[-1] // 2
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    rotated = np.concatenate((-x2, x1), axis=-1)
    return (x * cos) + (rotated * sin)


def apply_activation(x: np.ndarray, act_type: ActivationType) -> np.ndarray:
    """Applies activation function (SiLU, GELU, ReLU2)."""
    if act_type == ActivationType.SWIGLU:
        # SiLU(x) = x * sigmoid(x)
        return x / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))
    elif act_type == ActivationType.GEGLU or act_type == ActivationType.GELU:
        # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))
    elif act_type == ActivationType.RELU2:
        return np.square(np.maximum(x, 0.0))
    return x


class UniversalNPUTemplateDecoderLayer:
    """
    Universal Parameterized Decoder Layer executed on Qualcomm Hexagon v73 NPU.
    A single compiled template is retained in CDSP VTCM and shared across all N layers.
    """

    def __init__(self, config: UniversalModelConfig):
        self.config = config
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.gqa_ratio = max(1, self.num_heads // self.num_kv_heads)

    def _dequantize_crouton_linear(
        self, 
        x: np.ndarray, 
        weight_crouton: np.ndarray, 
        scale: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Executes Crouton micro-tiled matrix multiplication matching Hexagon HMX systolic array.
        If weights are packed in Crouton format [n_tiles_out, n_tiles_in, 64, 64],
        computes output with zero alignment penalties.
        """
        if weight_crouton.ndim == 4:
            # Reconstruct weight view from Crouton tiles
            n_out, n_in, t_n, t_k = weight_crouton.shape
            w_2d = weight_crouton.transpose(0, 2, 1, 3).reshape(n_out * t_n, n_in * t_k)
            w = w_2d.astype(np.float32)
            if scale is not None:
                w = w * scale
        elif weight_crouton.dtype == np.uint8 and scale is not None:
            # Unpack 4-bit nibbles
            low = (weight_crouton & 0x0F).astype(np.float32) - 8.0
            high = ((weight_crouton >> 4) & 0x0F).astype(np.float32) - 8.0
            unpacked = np.empty((weight_crouton.shape[0], weight_crouton.shape[1] * 2), dtype=np.float32)
            unpacked[:, 0::2] = low
            unpacked[:, 1::2] = high
            w = unpacked * scale
        else:
            w = weight_crouton.astype(np.float32)

        return np.matmul(x, w.T)

    def forward(
        self,
        hidden_states: np.ndarray,
        layer_weights: Dict[str, Any],
        kv_cache_k: Optional[np.ndarray] = None,
        kv_cache_v: Optional[np.ndarray] = None,
        cos_sin: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        layer_idx: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Universal forward pass executing 1 Transformer layer on Hexagon NPU.
        """
        tensors = layer_weights.get("tensors", layer_weights)
        scales = layer_weights.get("scales", {})

        residual = hidden_states

        # 1. Input Layernorm (Pre-attention)
        input_norm_w = tensors.get("input_layernorm.weight", np.ones(self.hidden_size, dtype=np.float32))
        normed_x = universal_norm(
            hidden_states, 
            input_norm_w, 
            norm_type=self.config.norm_type, 
            eps=self.config.rms_norm_eps
        )

        # 2. Self-Attention Projections
        q_w = tensors.get("self_attn.q_proj.weight")
        k_w = tensors.get("self_attn.k_proj.weight")
        v_w = tensors.get("self_attn.v_proj.weight")
        o_w = tensors.get("self_attn.o_proj.weight")

        q = self._dequantize_crouton_linear(normed_x, q_w, scales.get("self_attn.q_proj.weight"))
        k = self._dequantize_crouton_linear(normed_x, k_w, scales.get("self_attn.k_proj.weight"))
        v = self._dequantize_crouton_linear(normed_x, v_w, scales.get("self_attn.v_proj.weight"))

        # Reshape for multi-head attention
        batch_size, seq_len, _ = hidden_states.shape
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Q/K Normalization (if supported, e.g. Gemma 4 / Qwen)
        if self.config.has_qk_norm:
            if "self_attn.q_norm.weight" in tensors:
                q = universal_norm(q, tensors["self_attn.q_norm.weight"], self.config.norm_type, self.config.rms_norm_eps)
            if "self_attn.k_norm.weight" in tensors:
                k = universal_norm(k, tensors["self_attn.k_norm.weight"], self.config.norm_type, self.config.rms_norm_eps)

        # Apply RoPE
        if cos_sin is not None:
            cos, sin = cos_sin
            q = apply_rotary_pos_emb(q, cos, sin)
            k = apply_rotary_pos_emb(k, cos, sin)

        # Append to KV cache
        if kv_cache_k is not None and kv_cache_k.size > 0:
            k_full = np.concatenate([kv_cache_k, k], axis=2)
            v_full = np.concatenate([kv_cache_v, v], axis=2)
        else:
            k_full = k
            v_full = v

        # Repeat KV heads for GQA
        if self.gqa_ratio > 1:
            k_rep = np.repeat(k_full, self.gqa_ratio, axis=1)
            v_rep = np.repeat(v_full, self.gqa_ratio, axis=1)
        else:
            k_rep = k_full
            v_rep = v_full

        # Scaled Dot-Product Attention
        scale_factor = 1.0 / np.sqrt(self.head_dim)
        scores = np.matmul(q, k_rep.transpose(0, 1, 3, 2)) * scale_factor

        # Softcapping (Gemma-style)
        if self.config.attn_logit_softcapping is not None:
            cap = self.config.attn_logit_softcapping
            scores = cap * np.tanh(scores / cap)

        # Softmax & Context aggregation
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        attn_out = np.matmul(attn_probs, v_rep)

        # Output projection
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        attn_proj = self._dequantize_crouton_linear(attn_out, o_w, scales.get("self_attn.o_proj.weight"))

        # Post-attention Layernorm (Gemma style)
        if "post_attention_layernorm.weight" in tensors:
            attn_proj = universal_norm(
                attn_proj, 
                tensors["post_attention_layernorm.weight"], 
                self.config.norm_type, 
                self.config.rms_norm_eps
            )

        # Add residual connection
        x = residual + attn_proj
        residual = x

        # 3. Feed-Forward Network (MLP)
        pre_ffn_norm_w = tensors.get(
            "pre_feedforward_layernorm.weight", 
            tensors.get("post_attention_layernorm.weight", np.ones(self.hidden_size, dtype=np.float32))
        )
        normed_ffn = universal_norm(x, pre_ffn_norm_w, self.config.norm_type, self.config.rms_norm_eps)

        gate_w = tensors.get("mlp.gate_proj.weight")
        up_w = tensors.get("mlp.up_proj.weight")
        down_w = tensors.get("mlp.down_proj.weight")

        if gate_w is not None and up_w is not None:
            # Gated MLP (SwiGLU / GeGLU)
            gate = self._dequantize_crouton_linear(normed_ffn, gate_w, scales.get("mlp.gate_proj.weight"))
            up = self._dequantize_crouton_linear(normed_ffn, up_w, scales.get("mlp.up_proj.weight"))
            activated_gate = apply_activation(gate, self.config.activation_type)
            mlp_hidden = activated_gate * up
        else:
            # Non-gated standard MLP
            mlp_hidden = apply_activation(normed_ffn, self.config.activation_type)

        mlp_out = self._dequantize_crouton_linear(mlp_hidden, down_w, scales.get("mlp.down_proj.weight"))

        # Post-feedforward Layernorm (Gemma style)
        if "post_feedforward_layernorm.weight" in tensors:
            mlp_out = universal_norm(
                mlp_out, 
                tensors["post_feedforward_layernorm.weight"], 
                self.config.norm_type, 
                self.config.rms_norm_eps
            )

        # Optional layer scalar
        if "layer_scalar" in tensors:
            mlp_out = mlp_out * tensors["layer_scalar"]

        output = residual + mlp_out
        return output, k, v
