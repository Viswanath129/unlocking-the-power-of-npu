"""
Qualcomm Hexagon Crouton Weight Packing & Quantization Engine
Converts standard flat tensor weights into Hexagon v73 HMX/HVX Crouton micro-tiles.
"""

import numpy as np
import struct
from typing import Tuple, Dict, Any, Optional
from .config import NPUQuantMode, NPUHardwareProfile


class CroutonWeightPacker:
    """
    Transforms 2D Linear / Convolution weights into Qualcomm Crouton micro-tiling layout.
    Crouton format arranges matrices into small 2D blocks (e.g. 64x64 or 32x128)
    aligned to 128-byte HVX cache line boundaries to guarantee zero-conflict
    SRAM bank reads when feeding the HMX systolic matrix accelerator.
    """

    def __init__(self, hardware: Optional[NPUHardwareProfile] = None):
        self.hw = hardware or NPUHardwareProfile()
        self.tile_k = self.hw.default_crouton_tile_k
        self.tile_n = self.hw.default_crouton_tile_n
        self.vector_align = self.hw.vector_width_bytes

    def quantize_to_int8(self, weight: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Per-channel symmetric INT8 quantization.
        weight shape: [out_features, in_features]
        Returns:
            quantized_weights: int8 [out_features, in_features]
            scales: float32 [out_features]
        """
        w = weight.astype(np.float32)
        abs_max = np.max(np.abs(w), axis=-1, keepdims=True)
        abs_max = np.maximum(abs_max, 1e-8)
        scale = abs_max / 127.0
        q = np.clip(np.round(w / scale), -128, 127).astype(np.int8)
        return q, scale.squeeze(-1).astype(np.float32)

    def quantize_to_int4(self, weight: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Per-channel symmetric INT4 quantization (packed 2 values per byte).
        weight shape: [out_features, in_features]
        Returns:
            packed_weights: uint8 [out_features, in_features // 2]
            scales: float32 [out_features]
        """
        w = weight.astype(np.float32)
        out_f, in_f = w.shape
        if in_f % 2 != 0:
            w = np.pad(w, ((0, 0), (0, 1)), mode='constant')
            in_f += 1

        abs_max = np.max(np.abs(w), axis=-1, keepdims=True)
        abs_max = np.maximum(abs_max, 1e-8)
        scale = abs_max / 7.0
        q = np.clip(np.round(w / scale), -8, 7).astype(np.int8)
        
        # Pack two 4-bit values (low nibble, high nibble) into 1 unsigned byte
        low = q[:, 0::2] & 0x0F
        high = (q[:, 1::2] & 0x0F) << 4
        packed = (low | high).astype(np.uint8)
        return packed, scale.squeeze(-1).astype(np.float32)

    def pack_crouton_int8(self, q_weight: np.ndarray) -> np.ndarray:
        """
        Packs [out_features, in_features] INT8 weights into 4D Crouton micro-tiles:
        [out_features // tile_n, in_features // tile_k, tile_n, tile_k]
        flattened for sequential DMA streaming into Hexagon VTCM.
        """
        out_f, in_f = q_weight.shape
        pad_n = (self.tile_n - (out_f % self.tile_n)) % self.tile_n
        pad_k = (self.tile_k - (in_f % self.tile_k)) % self.tile_k
        
        if pad_n > 0 or pad_k > 0:
            padded = np.pad(q_weight, ((0, pad_n), (0, pad_k)), mode='constant')
        else:
            padded = q_weight

        p_out, p_in = padded.shape
        n_tiles_out = p_out // self.tile_n
        n_tiles_in = p_in // self.tile_k

        # Reshape into tiles: [n_tiles_out, tile_n, n_tiles_in, tile_k]
        tiled = padded.reshape(n_tiles_out, self.tile_n, n_tiles_in, self.tile_k)
        # Permute to Crouton tile order: [n_tiles_out, n_tiles_in, tile_n, tile_k]
        crouton = np.ascontiguousarray(tiled.transpose(0, 2, 1, 3))
        return crouton

    def apply_2_4_sparsity(self, weight: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies 2:4 structured hardware sparsity for Hexagon v73 HMX engine.
        In every 4 contiguous elements along the K dimension, exactly 2 smallest
        elements are pruned to zero.
        Returns:
            sparse_weights: weights with 2:4 sparsity enforced
            mask: uint8 bitmask encoding which 2 elements are retained
        """
        orig_shape = weight.shape
        w_flat = weight.reshape(-1, 4)
        abs_w = np.abs(w_flat)
        top2_idx = np.argsort(abs_w, axis=-1)[:, 2:]
        
        sparse_w = np.zeros_like(w_flat)
        rows = np.arange(len(w_flat))[:, None]
        sparse_w[rows, top2_idx] = w_flat[rows, top2_idx]
        
        mask = np.zeros(len(w_flat), dtype=np.uint8)
        for col in range(2):
            mask |= (1 << top2_idx[:, col])
            
        return sparse_w.reshape(orig_shape), mask

    def pack_layer_weights(self, layer_state_dict: Dict[str, np.ndarray], quant_mode: NPUQuantMode) -> Dict[str, Any]:
        """
        Packs an entire transformer decoder layer's weights into Qualcomm Crouton format.
        """
        packed_layer = {
            "quant_mode": quant_mode.value,
            "tensors": {},
            "scales": {}
        }

        for name, tensor in layer_state_dict.items():
            if "weight" in name and tensor.ndim == 2:
                if quant_mode == NPUQuantMode.INT4:
                    packed, scales = self.quantize_to_int4(tensor)
                    packed_layer["tensors"][name] = packed
                    packed_layer["scales"][name] = scales
                elif quant_mode == NPUQuantMode.INT8:
                    q, scales = self.quantize_to_int8(tensor)
                    crouton = self.pack_crouton_int8(q)
                    packed_layer["tensors"][name] = crouton
                    packed_layer["scales"][name] = scales
                else: # FP16
                    packed_layer["tensors"][name] = tensor.astype(np.float16)
            else:
                packed_layer["tensors"][name] = tensor.astype(np.float32)

        return packed_layer
