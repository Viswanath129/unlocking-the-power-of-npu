"""
Ultra-Low-RAM INT8 Quantized Dynamic KV Cache for Qualcomm NPU
Reduces KV cache footprint by 75% to conserve unified LPDDR5x RAM on 16GB laptops.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
from .config import NPUModelConfig


class QuantizedKVCache:
    """
    Dynamically growing INT8 quantized KV cache.
    Stores Key and Value tensors in int8 format with per-channel scaling factors,
    minimizing memory usage for long conversations.
    """

    def __init__(self, config: NPUModelConfig):
        self.config = config
        self.num_layers = config.num_hidden_layers
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.max_seq_len = config.max_seq_len
        self.current_seq_len = 0
        
        self._cache: Dict[int, Dict[str, Any]] = {}
        self.reset()

    def reset(self):
        """Resets the KV cache state"""
        self.current_seq_len = 0
        self._cache.clear()
        for i in range(self.num_layers):
            self._cache[i] = {
                "k": np.empty((1, self.num_kv_heads, 0, self.head_dim), dtype=np.int8),
                "k_scale": np.empty((1, self.num_kv_heads, 0, 1), dtype=np.float32),
                "v": np.empty((1, self.num_kv_heads, 0, self.head_dim), dtype=np.int8),
                "v_scale": np.empty((1, self.num_kv_heads, 0, 1), dtype=np.float32)
            }

    def _quantize_tensor(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Symmetric INT8 quantization per token/head"""
        abs_max = np.max(np.abs(x), axis=-1, keepdims=True)
        abs_max = np.maximum(abs_max, 1e-8)
        scale = abs_max / 127.0
        q = np.clip(np.round(x / scale), -128, 127).astype(np.int8)
        return q, scale.astype(np.float32)

    def _dequantize_tensor(self, q: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """Dequantizes INT8 back to float32 for computation"""
        return q.astype(np.float32) * scale

    def update(self, layer_idx: int, new_k: np.ndarray, new_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quantizes new K and V states to INT8, appends them to the layer's cache,
        and returns the dequantized active history slice for attention.
        """
        q_k, s_k = self._quantize_tensor(new_k)
        q_v, s_v = self._quantize_tensor(new_v)

        entry = self._cache[layer_idx]
        entry["k"] = np.concatenate([entry["k"], q_k], axis=2)
        entry["k_scale"] = np.concatenate([entry["k_scale"], s_k], axis=2)
        entry["v"] = np.concatenate([entry["v"], q_v], axis=2)
        entry["v_scale"] = np.concatenate([entry["v_scale"], s_v], axis=2)

        full_k = self._dequantize_tensor(entry["k"], entry["k_scale"])
        full_v = self._dequantize_tensor(entry["v"], entry["v_scale"])
        return full_k, full_v

    def advance_step(self, seq_delta: int = 1):
        self.current_seq_len += seq_delta

    def get_memory_bytes(self) -> int:
        """Calculates total memory footprint of the KV cache across all layers"""
        total = 0
        for entry in self._cache.values():
            total += entry["k"].nbytes + entry["k_scale"].nbytes + entry["v"].nbytes + entry["v_scale"].nbytes
        return total
