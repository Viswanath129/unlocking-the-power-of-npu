"""
Snapdragon NPU End-to-End Execution & Throughput Benchmark
Simulates layer-wise execution on Snapdragon X Plus Hexagon v73 NPU
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from npu_engine.config import NPUModelConfig, NPUQuantMode
from npu_engine.engine import NPULayerStreamingEngine


def main():
    print("=" * 70, flush=True)
    print(" Qualcomm Snapdragon Hexagon NPU Benchmark (SC8380XP)", flush=True)
    print("=" * 70, flush=True)

    config = NPUModelConfig(
        model_name_or_path="Qwen/Qwen2.5-72B-Instruct",
        quant_mode=NPUQuantMode.INT4,      # INT4 Crouton quantized for Hexagon HMX
        num_hidden_layers=8,               # 8 layers for fast benchmark validation
        hidden_size=512,
        intermediate_size=1536,
        num_attention_heads=8,
        num_key_value_heads=2,
        vocab_size=32000,
        max_seq_len=2048
    )

    print(f"Target NPU Architecture: {config.hardware.dsp_arch} ({config.hardware.peak_tops} TOPS)", flush=True)
    print(f"Quantization Mode:       {config.quant_mode.value.upper()} (Crouton Tiled)", flush=True)
    print(f"Total Model Layers:      {config.num_hidden_layers} layers (Streaming across 1 Template)", flush=True)
    print(f"FastRPC Buffers:         Dual Ping-Pong Zero-Copy Unified Memory\n", flush=True)

    model = NPULayerStreamingEngine(config)
    model.load_mock_model(num_layers=config.num_hidden_layers)

    prompt_tokens = [1, 15043, 29892, 10567]
    print(f"\n[Benchmark] Input Prompt: {len(prompt_tokens)} tokens", flush=True)

    tokens_to_generate = 5
    output_tokens = model.generate(prompt_tokens, max_new_tokens=tokens_to_generate)

    print("\n" + "=" * 70, flush=True)
    print(" Benchmark Completed Successfully!", flush=True)
    print(f" Generated Token IDs: {output_tokens}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
