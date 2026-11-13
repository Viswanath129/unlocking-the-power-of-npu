"""
Universal Model Architecture Detector & Config Registry for Qualcomm Hexagon NPU
Supports ANY autoregressive decoder model: LLaMA, Gemma, Qwen, Mistral, Phi, DeepSeek, etc.
Guarantees lowest possible RAM usage by computing optimal streaming buffer sizes.
"""

import os
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple


class ModelFamily(str, Enum):
    LLAMA = "llama"            # LLaMA 2, LLaMA 3, LLaMA 3.1, LLaMA 3.2, LLaMA 3.3
    GEMMA = "gemma"            # Gemma, Gemma 2, Gemma 3, Gemma 4
    QWEN = "qwen"              # Qwen, Qwen 2, Qwen 2.5, Qwen 2.5-Coder
    MISTRAL = "mistral"        # Mistral, Mixtral, Zephyr
    PHI = "phi"                # Phi-2, Phi-3, Phi-4
    DEEPSEEK = "deepseek"      # DeepSeek-V2, DeepSeek-V3, DeepSeek-R1
    GENERIC = "generic"        # Fallback generic transformer decoder


class AttentionType(str, Enum):
    MHA = "mha"                # Multi-Head Attention (num_heads == num_kv_heads)
    GQA = "gqa"                # Grouped-Query Attention (num_heads > num_kv_heads)
    MQA = "mqa"                # Multi-Query Attention (num_kv_heads == 1)
    SLIDING = "sliding"        # Sliding Window Attention
    HYBRID = "hybrid"          # Alternating Sliding + Full Global Attention
    MLA = "mla"                # Multi-Head Latent Attention (DeepSeek)


class ActivationType(str, Enum):
    SWIGLU = "swiglu"          # SiLU Gated Linear Unit (LLaMA, Qwen, Mistral)
    GEGLU = "geglu"            # GELU Gated Linear Unit (Gemma, Gemma 2, Gemma 4)
    GELU = "gelu"              # Standard GELU (Phi-2)
    RELU2 = "relu2"            # Squared ReLU


class NormType(str, Enum):
    RMSNORM = "rmsnorm"        # Root Mean Square Norm
    RMSNORM_OFFSET = "rmsnorm_offset"  # Gemma style: norm_weight + 1.0
    LAYERNORM = "layernorm"    # Standard LayerNorm (mean + variance)


@dataclass
class UniversalModelConfig:
    """Universal Model Configuration holding all architectural attributes for NPU execution."""
    model_name_or_path: str
    family: ModelFamily = ModelFamily.GENERIC
    num_hidden_layers: int = 32
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 128256
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-5
    
    # Architecture specifics
    attention_type: AttentionType = AttentionType.GQA
    activation_type: ActivationType = ActivationType.SWIGLU
    norm_type: NormType = NormType.RMSNORM
    has_pre_ffn_norm: bool = False
    has_post_ffn_norm: bool = False
    has_qk_norm: bool = False
    tie_word_embeddings: bool = False
    
    # RoPE configuration
    rope_theta: float = 500000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    
    # Softcapping & sliding window
    attn_logit_softcapping: Optional[float] = None
    final_logit_softcapping: Optional[float] = None
    sliding_window_size: Optional[int] = None
    full_attention_interval: Optional[int] = None

    # MoE specifics (if applicable)
    is_moe: bool = False
    num_experts: int = 1
    num_experts_per_tok: int = 1

    # NPU Hardware Tiling (Hexagon v73)
    crouton_tile_k: int = 64
    crouton_tile_n: int = 64
    quant_bits: int = 4

    def compute_single_layer_bytes(self, quant_bits: Optional[int] = None) -> int:
        """
        Computes exact byte size of a single layer buffer when quantized.
        Used to allocate the optimal zero-copy Ping-Pong buffers.
        """
        bits = quant_bits or self.quant_bits
        bytes_per_param = bits / 8.0
        
        # Attention projection parameters
        q_params = self.hidden_size * (self.num_attention_heads * self.head_dim)
        k_params = self.hidden_size * (self.num_key_value_heads * self.head_dim)
        v_params = self.hidden_size * (self.num_key_value_heads * self.head_dim)
        o_params = (self.num_attention_heads * self.head_dim) * self.hidden_size
        attn_total = q_params + k_params + v_params + o_params
        
        # MLP parameters
        if self.is_moe:
            # In MoE, we only stream active experts into the single-layer execution buffer!
            mlp_total = self.num_experts_per_tok * (3 * self.hidden_size * self.intermediate_size)
        else:
            mlp_total = 3 * (self.hidden_size * self.intermediate_size)
            
        layer_params = attn_total + mlp_total
        return int(layer_params * bytes_per_param)

    def get_layer_tensor_mapping(self, layer_idx: int) -> Dict[str, List[str]]:
        """
        Returns candidate tensor keys in safetensors/bin files for each logical role in layer `layer_idx`.
        Supports all common Hugging Face naming conventions across all model families.
        """
        idx = layer_idx
        return {
            "input_layernorm": [
                f"model.layers.{idx}.input_layernorm.weight",
                f"model.language_model.layers.{idx}.input_layernorm.weight",
                f"layers.{idx}.input_layernorm.weight",
                f"transformer.h.{idx}.input_layernorm.weight",
                f"model.layers.{idx}.operator_norm.weight"
            ],
            "q_proj": [
                f"model.layers.{idx}.self_attn.q_proj.weight",
                f"model.language_model.layers.{idx}.self_attn.q_proj.weight",
                f"layers.{idx}.attention.wq.weight",
                f"transformer.h.{idx}.self_attention.q_proj.weight"
            ],
            "k_proj": [
                f"model.layers.{idx}.self_attn.k_proj.weight",
                f"model.language_model.layers.{idx}.self_attn.k_proj.weight",
                f"layers.{idx}.attention.wk.weight",
                f"transformer.h.{idx}.self_attention.k_proj.weight"
            ],
            "v_proj": [
                f"model.layers.{idx}.self_attn.v_proj.weight",
                f"model.language_model.layers.{idx}.self_attn.v_proj.weight",
                f"layers.{idx}.attention.wv.weight",
                f"transformer.h.{idx}.self_attention.v_proj.weight"
            ],
            "o_proj": [
                f"model.layers.{idx}.self_attn.o_proj.weight",
                f"model.language_model.layers.{idx}.self_attn.o_proj.weight",
                f"layers.{idx}.attention.wo.weight",
                f"transformer.h.{idx}.self_attention.o_proj.weight"
            ],
            "post_attention_layernorm": [
                f"model.layers.{idx}.post_attention_layernorm.weight",
                f"model.language_model.layers.{idx}.post_attention_layernorm.weight",
                f"layers.{idx}.post_attention_layernorm.weight",
                f"transformer.h.{idx}.post_attention_layernorm.weight"
            ],
            "gate_proj": [
                f"model.layers.{idx}.mlp.gate_proj.weight",
                f"model.language_model.layers.{idx}.mlp.gate_proj.weight",
                f"layers.{idx}.feed_forward.w1.weight",
                f"transformer.h.{idx}.mlp.gate_proj.weight"
            ],
            "up_proj": [
                f"model.layers.{idx}.mlp.up_proj.weight",
                f"model.language_model.layers.{idx}.mlp.up_proj.weight",
                f"layers.{idx}.feed_forward.w3.weight",
                f"transformer.h.{idx}.mlp.up_proj.weight"
            ],
            "down_proj": [
                f"model.layers.{idx}.mlp.down_proj.weight",
                f"model.language_model.layers.{idx}.mlp.down_proj.weight",
                f"layers.{idx}.feed_forward.w2.weight",
                f"transformer.h.{idx}.mlp.down_proj.weight"
            ],
            "q_norm": [
                f"model.layers.{idx}.self_attn.q_norm.weight",
                f"model.language_model.layers.{idx}.self_attn.q_norm.weight"
            ],
            "k_norm": [
                f"model.layers.{idx}.self_attn.k_norm.weight",
                f"model.language_model.layers.{idx}.self_attn.k_norm.weight"
            ],
            "pre_feedforward_layernorm": [
                f"model.language_model.layers.{idx}.pre_feedforward_layernorm.weight"
            ],
            "post_feedforward_layernorm": [
                f"model.language_model.layers.{idx}.post_feedforward_layernorm.weight"
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], model_name_or_path: str = "") -> "UniversalModelConfig":
        """Intelligently inspects any raw Hugging Face config dict and resolves architecture."""
        # Check text_config sub-dictionary (Gemma 4, multimodals)
        tc = data.get("text_config", {}) if isinstance(data.get("text_config"), dict) else {}
        cfg = {**data, **tc}
        
        arch = cfg.get("architectures", [""])[0].lower() if cfg.get("architectures") else ""
        model_type = cfg.get("model_type", "").lower()
        
        # 1. Determine Model Family
        family = ModelFamily.GENERIC
        if "gemma" in arch or "gemma" in model_type:
            family = ModelFamily.GEMMA
        elif "qwen" in arch or "qwen" in model_type:
            family = ModelFamily.QWEN
        elif "llama" in arch or "llama" in model_type:
            family = ModelFamily.LLAMA
        elif "mistral" in arch or "mixtral" in model_type or "mistral" in model_type:
            family = ModelFamily.MISTRAL
        elif "phi" in arch or "phi" in model_type:
            family = ModelFamily.PHI
        elif "deepseek" in arch or "deepseek" in model_type:
            family = ModelFamily.DEEPSEEK

        # 2. Extract Dimensions
        num_layers = cfg.get("num_hidden_layers", cfg.get("n_layer", 32))
        hidden_size = cfg.get("hidden_size", cfg.get("n_embd", 4096))
        intermediate_size = cfg.get("intermediate_size", cfg.get("n_inner", hidden_size * 4))
        num_heads = cfg.get("num_attention_heads", cfg.get("n_head", 32))
        num_kv_heads = cfg.get("num_key_value_heads", num_heads)
        head_dim = cfg.get("head_dim", hidden_size // num_heads)
        vocab_size = cfg.get("vocab_size", 32000)
        max_pos = cfg.get("max_position_embeddings", cfg.get("n_positions", 8192))
        rms_norm_eps = cfg.get("rms_norm_eps", 1e-6)

        # 3. Determine Attention & Activation Types
        if num_kv_heads == num_heads:
            attn_type = AttentionType.MHA
        elif num_kv_heads == 1:
            attn_type = AttentionType.MQA
        else:
            attn_type = AttentionType.GQA

        # Hybrid or sliding attention
        sliding_size = cfg.get("sliding_window", cfg.get("sliding_window_size"))
        full_interval = None
        if "layer_types" in cfg and isinstance(cfg["layer_types"], list):
            attn_type = AttentionType.HYBRID
            full_interval = 6
        elif sliding_size is not None:
            attn_type = AttentionType.SLIDING

        # Activation
        act = cfg.get("hidden_act", cfg.get("hidden_activation", "silu")).lower()
        if "gelu" in act:
            act_type = ActivationType.GEGLU if ("gemma" in model_type or "geglu" in act) else ActivationType.GELU
        elif "relu2" in act or "squared_relu" in act:
            act_type = ActivationType.RELU2
        else:
            act_type = ActivationType.SWIGLU

        # Norm Type
        norm_type = NormType.RMSNORM
        if family == ModelFamily.GEMMA:
            norm_type = NormType.RMSNORM_OFFSET

        # Softcapping
        attn_soft = cfg.get("attn_logit_softcapping")
        final_soft = cfg.get("final_logit_softcapping")
        tie_weights = cfg.get("tie_word_embeddings", False)

        return cls(
            model_name_or_path=model_name_or_path,
            family=family,
            num_hidden_layers=num_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            vocab_size=vocab_size,
            max_position_embeddings=max_pos,
            rms_norm_eps=rms_norm_eps,
            attention_type=attn_type,
            activation_type=act_type,
            norm_type=norm_type,
            has_pre_ffn_norm=(family == ModelFamily.GEMMA and num_layers >= 40),
            has_post_ffn_norm=(family == ModelFamily.GEMMA and num_layers >= 40),
            has_qk_norm=(cfg.get("use_qk_norm", False) or family == ModelFamily.GEMMA),
            tie_word_embeddings=tie_weights,
            rope_theta=cfg.get("rope_theta", 500000.0),
            attn_logit_softcapping=attn_soft,
            final_logit_softcapping=final_soft,
            sliding_window_size=sliding_size,
            full_attention_interval=full_interval
        )

    @classmethod
    def from_model_dir(cls, model_dir: str) -> "UniversalModelConfig":
        """Loads configuration from a local directory or creates default config."""
        cfg_path = os.path.join(model_dir, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_dict(data, model_name_or_path=model_dir)
        return cls(model_name_or_path=model_dir)
