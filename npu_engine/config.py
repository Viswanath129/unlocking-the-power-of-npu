"""
Qualcomm Hexagon NPU & Layer-Streaming Configuration Specifications
Target Platform: Snapdragon X Plus (SC8380XP / Hamoa) - Hexagon v73 NPU
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any


class NPUQuantMode(str, Enum):
    INT4 = "int4"        # Nibble weights (convbnb/convhnh kernels on HMX)
    INT8 = "int8"        # Byte weights (convbbb kernels on HMX)
    FP16 = "fp16"        # Half-precision float (convhhh/convf16)


class NPUEngineBackend(str, Enum):
    QNN_HTP = "qnn_htp"        # Native Qualcomm QNN HTP v73 Skeleton
    DIRECTML = "directml"      # Windows DirectML MCDM NPU Device
    ONNX_QNN = "onnx_qnn"      # ONNX Runtime QNN Execution Provider
    NATIVE_EMU = "native_emu"  # ARM64 NEON / CPU Vector Fallback


@dataclass
class NPUHardwareProfile:
    """Hardware parameters identified during reverse engineering of Hexagon v73 NPU"""
    soc_codename: str = "SC8380XP"
    device_id: str = "ACPI\\VEN_QCOM&DEV_0D0A&REV_0101"
    dsp_arch: str = "Hexagon v73"
    vector_width_bytes: int = 128         # 1024-bit HVX SIMD vector registers
    num_hvx_units: int = 2                 # STREAMING_2HVX_CLIENT_CLASS
    num_hmx_units: int = 1                 # Hexagon Matrix Systolic Array
    vtcm_total_size_mb: int = 8            # ~8MB Vector Tightly Coupled Memory SRAM
    fastrpc_sessions: int = 12             # 12 SMMU Context Banks (0x14-0x1F)
    peak_tops: int = 45                    # 45 TOPS peak INT8 compute
    supports_2_4_sparsity: bool = True     # Hardware structured sparsity acceleration
    default_crouton_tile_k: int = 64       # Crouton tile depth for HMX
    default_crouton_tile_n: int = 64       # Crouton tile width for HMX


@dataclass
class NPUModelConfig:
    """Model execution configuration for Hexagon NPU Layer-Streaming Engine"""
    model_name_or_path: str = "google/gemma-4-31B-it"
    backend: NPUEngineBackend = NPUEngineBackend.ONNX_QNN
    quant_mode: NPUQuantMode = NPUQuantMode.INT4
    
    # Layer Streaming Parameters
    enable_double_buffering: bool = True   # Asynchronous Ping-Pong FastRPC streaming
    prefetch_workers: int = 2              # Background threads streaming weights from NVMe/RAM
    use_unified_memory: bool = True        # Zero-copy buffer mapping in Windows on ARM
    
    # Context & Layer Dimensions
    max_seq_len: int = 4096
    hidden_size: int = 5376
    num_attention_heads: int = 32
    num_key_value_heads: int = 16          # Grouped-Query Attention (GQA)
    intermediate_size: int = 21504         # MLP hidden dimension
    num_hidden_layers: int = 60            # 60 layers for Gemma 4 31B
    vocab_size: int = 262144
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    final_logit_softcapping: float = 30.0
    sliding_window_size: int = 4096
    full_attention_interval: int = 6
    
    # KV Cache Parameters
    paged_kv_block_size: int = 16          # Tokens per KV memory page
    max_batch_size: int = 1
    
    hardware: NPUHardwareProfile = field(default_factory=NPUHardwareProfile)

    @classmethod
    def for_gemma4_31b(cls, quant_mode: NPUQuantMode = NPUQuantMode.INT4) -> "NPUModelConfig":
        """Factory for Google Gemma 4 31B Instruct."""
        return cls(
            model_name_or_path="google/gemma-4-31B-it",
            quant_mode=quant_mode,
            hidden_size=5376,
            num_attention_heads=32,
            num_key_value_heads=16,
            intermediate_size=21504,
            num_hidden_layers=60,
            vocab_size=262144,
            head_dim=256,
            sliding_window_size=4096,
            full_attention_interval=6
        )

    @classmethod
    def for_qwen25_72b(cls, quant_mode: NPUQuantMode = NPUQuantMode.INT4) -> "NPUModelConfig":
        """Factory for Qwen 2.5 72B Instruct."""
        return cls(
            model_name_or_path="Qwen/Qwen2.5-72B-Instruct",
            quant_mode=quant_mode,
            hidden_size=8192,
            num_attention_heads=64,
            num_key_value_heads=8,
            intermediate_size=29568,
            num_hidden_layers=80,
            vocab_size=152064,
            head_dim=128
        )
