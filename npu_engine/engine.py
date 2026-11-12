"""
Qualcomm Hexagon NPU Layer-Streaming Engine
Orchestrates large language models on Snapdragon X Plus / Windows on ARM.
"""

import time
import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Generator, Union, Tuple

from .config import NPUModelConfig, NPUQuantMode, NPUEngineBackend
from .crouton_packer import CroutonWeightPacker
from .fastrpc_streamer import FastRPCBufferRing
from .quantized_kv_cache import QuantizedKVCache
from .npu_layer_engine import NPUTemplateDecoderLayer, rms_norm


class NPULayerStreamingEngine:
    """
    NPULayerStreamingEngine orchestrates large language model execution on Qualcomm Hexagon NPU.
    
    Key Architectural Advantages:
    1. Single-Layer Template: Compiles and holds 1 decoder block on Hexagon CDSP,
       reusing it across all 80 layers without running out of device RAM.
    2. Zero-Copy FastRPC Streaming: Leverages unified LPDDR5x RAM with Ping-Pong
       buffers to swap layer weights asynchronously, eliminating PCIe copy stalls.
    3. Crouton Micro-Tiling: Quantizes weights into 128-byte aligned 4D Croutons
       for full 45 TOPS HMX systolic matrix acceleration.
    4. Layer-Wise Paged KV Cache: Stores KV history in unified RAM, paging each
       layer's state on demand in INT8 compressed format (75% RAM savings).
    """

    def __init__(self, config: Optional[NPUModelConfig] = None):
        self.config = config or NPUModelConfig()
        self.packer = CroutonWeightPacker(self.config.hardware)
        self.streamer = FastRPCBufferRing()
        self.kv_cache = QuantizedKVCache(self.config)
        self.layer_engine = NPUTemplateDecoderLayer(self.config)
        
        # Resident Host Modules (Embeddings, Final Norm, LM Head)
        self.embed_tokens: Optional[np.ndarray] = None
        self.final_norm_weight: Optional[np.ndarray] = None
        self.lm_head_weight: Optional[np.ndarray] = None
        
        # In-memory mock or real weight provider
        self._layer_weight_store: Dict[int, Dict[str, Any]] = {}
        self._is_initialized = False

    def load_mock_model(self, num_layers: Optional[int] = None):
        """Initializes a compact synthetic model conforming to config dimensions"""
        n_layers = num_layers or self.config.num_hidden_layers
        self.config.num_hidden_layers = n_layers
        hidden = self.config.hidden_size
        vocab = self.config.vocab_size
        inter = self.config.intermediate_size
        num_kv = self.config.num_key_value_heads
        num_heads = self.config.num_attention_heads
        head_dim = hidden // num_heads

        print(f"[NPU-Engine] Initializing {n_layers}-layer model (Hidden: {hidden}, Inter: {inter}, Heads: {num_heads}/{num_kv})...", flush=True)

        self.embed_tokens = np.random.randn(vocab, hidden).astype(np.float32) * 0.02
        self.final_norm_weight = np.ones(hidden, dtype=np.float32)
        self.lm_head_weight = np.random.randn(vocab, hidden).astype(np.float32) * 0.02

        for i in range(n_layers):
            layer_raw = {
                "input_layernorm.weight": np.ones(hidden, dtype=np.float32),
                "self_attn.q_proj.weight": np.random.randn(hidden, hidden).astype(np.float32) * 0.02,
                "self_attn.k_proj.weight": np.random.randn(num_kv * head_dim, hidden).astype(np.float32) * 0.02,
                "self_attn.v_proj.weight": np.random.randn(num_kv * head_dim, hidden).astype(np.float32) * 0.02,
                "self_attn.o_proj.weight": np.random.randn(hidden, hidden).astype(np.float32) * 0.02,
                "post_attention_layernorm.weight": np.ones(hidden, dtype=np.float32),
                "mlp.gate_proj.weight": np.random.randn(inter, hidden).astype(np.float32) * 0.02,
                "mlp.up_proj.weight": np.random.randn(inter, hidden).astype(np.float32) * 0.02,
                "mlp.down_proj.weight": np.random.randn(hidden, inter).astype(np.float32) * 0.02,
            }
            self._layer_weight_store[i] = self.packer.pack_layer_weights(layer_raw, self.config.quant_mode)

        self._is_initialized = True
        print(f"[NPU-Engine] Successfully packed {n_layers} layers in Crouton format ({self.config.quant_mode.value.upper()}) ready for NPU streaming.", flush=True)

    def _fetch_layer_weights(self, layer_idx: int) -> Dict[str, Any]:
        return self._layer_weight_store[layer_idx]

    def _compute_rope_frequencies(self, seq_len: int, start_pos: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        theta = 1000000.0  # Qwen2/2.5 1M theta
        freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2)[: (head_dim // 2)].astype(np.float32) / head_dim))
        t = np.arange(start_pos, start_pos + seq_len, dtype=np.float32)
        freqs = np.outer(t, freqs)
        freqs = np.concatenate([freqs, freqs], axis=-1)
        cos = np.cos(freqs)[None, None, :, :]
        sin = np.sin(freqs)[None, None, :, :]
        return cos, sin

    def forward_token(self, token_id: int, position: int) -> int:
        hidden_states = self.embed_tokens[token_id:token_id+1, :][None, :, :]
        cos_sin = self._compute_rope_frequencies(seq_len=1, start_pos=position)
        num_layers = self.config.num_hidden_layers

        current_layer_weights = self.streamer.load_layer_direct(self._fetch_layer_weights, 0)

        for layer_idx in range(num_layers):
            if layer_idx + 1 < num_layers:
                self.streamer.start_prefetch(self._fetch_layer_weights, layer_idx + 1)

            entry = self.kv_cache._cache[layer_idx]
            hist_k = self.kv_cache._dequantize_tensor(entry["k"], entry["k_scale"])
            hist_v = self.kv_cache._dequantize_tensor(entry["v"], entry["v_scale"])

            hidden_states, new_k, new_v = self.layer_engine.forward(
                hidden_states=hidden_states,
                layer_weights=current_layer_weights,
                kv_cache_k=hist_k,
                kv_cache_v=hist_v,
                cos_sin=cos_sin
            )

            self.kv_cache.update(layer_idx, new_k, new_v)

            if layer_idx + 1 < num_layers:
                current_layer_weights = self.streamer.get_prefetched_layer(self._fetch_layer_weights, layer_idx + 1)

        self.kv_cache.advance_step(1)

        final_x = rms_norm(hidden_states, self.final_norm_weight, self.config.rms_norm_eps)
        logits = np.matmul(final_x[0, -1, :], self.lm_head_weight.T)
        return int(np.argmax(logits))

    def generate(self, prompt_tokens: List[int], max_new_tokens: int = 8) -> List[int]:
        if not self._is_initialized:
            raise RuntimeError("Model is not initialized.")

        self.kv_cache.reset()
        generated = list(prompt_tokens)
        print(f"[NPU-Engine] Prefilling prompt ({len(prompt_tokens)} tokens)...", flush=True)

        for pos, tok in enumerate(prompt_tokens):
            self.forward_token(tok, pos)

        current_token = generated[-1]
        start_time = time.perf_counter()

        print(f"[NPU-Engine] Generating up to {max_new_tokens} tokens on Qualcomm Hexagon NPU...", flush=True)
        for step in range(max_new_tokens):
            pos = len(prompt_tokens) + step
            t0 = time.perf_counter()
            next_token = self.forward_token(current_token, pos)
            dt = time.perf_counter() - t0
            
            generated.append(next_token)
            current_token = next_token
            print(f"  Step {step+1:02d}: Token {next_token:6d} | Latency: {dt*1000:.2f} ms ({1.0/dt:.2f} tok/s)", flush=True)

        total_time = time.perf_counter() - start_time
        print(f"[NPU-Engine] Completed {max_new_tokens} tokens in {total_time:.2f}s ({max_new_tokens/total_time:.2f} tok/s average).", flush=True)
        return generated
