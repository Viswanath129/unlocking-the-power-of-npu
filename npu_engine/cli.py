"""
Universal Qualcomm Hexagon NPU CLI
Command-line tools for running, inspecting, and converting ANY model architecture on Snapdragon X Plus.
"""

import os
import sys
import argparse
import json

# Add repository root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from npu_engine.models.auto_model import UniversalModelConfig
from npu_engine.converters.universal_sharder import UniversalModelSharder
from npu_engine.universal_engine import UniversalNPUEngine
from npu_engine.config import NPUQuantMode


def cmd_inspect(args):
    """Inspects any model repository or config file."""
    print("=" * 70)
    print(" Qualcomm Hexagon NPU Architecture Inspector")
    print("=" * 70)
    cfg = UniversalModelConfig.from_model_dir(args.model_path)
    buf_mb = cfg.compute_single_layer_bytes(quant_bits=4) / (1024 * 1024)
    total_weights_gb = (cfg.compute_single_layer_bytes(quant_bits=4) * cfg.num_hidden_layers) / (1024 ** 3)
    
    print(f"Model ID / Path:        {cfg.model_name_or_path}")
    print(f"Detected Family:        {cfg.family.value.upper()}")
    print(f"Hidden Layers:          {cfg.num_hidden_layers}")
    print(f"Hidden Size:            {cfg.hidden_size}")
    print(f"Intermediate Size:      {cfg.intermediate_size}")
    print(f"Attention Heads:        {cfg.num_attention_heads} (Q) / {cfg.num_key_value_heads} (KV)")
    print(f"Attention Variant:      {cfg.attention_type.value.upper()}")
    print(f"Activation:             {cfg.activation_type.value.upper()}")
    print(f"Normalization:          {cfg.norm_type.value.upper()}")
    print(f"Vocab Size:             {cfg.vocab_size}")
    print(f"RoPE Theta:             {cfg.rope_theta}")
    print("-" * 70)
    print(f"Crouton INT4 Size/Layer: {buf_mb:.1f} MB")
    print(f"Total INT4 Model Size:   {total_weights_gb:.2f} GB")
    print(f"Active RAM Streaming:   < {buf_mb * 2 + 50:.1f} MB (Fits comfortably on 16GB laptop!)")
    print("=" * 70)


def cmd_convert(args):
    """Converts any model into Crouton INT4 layers for NPU execution."""
    print("=" * 70)
    print(" Qualcomm Hexagon NPU Universal Crouton Converter")
    print("=" * 70)
    sharder = UniversalModelSharder(args.model_path, args.output_dir)
    sharder.convert_all(max_layers=args.layers)


def cmd_chat(args):
    """Launches the ChatGPT Web Interface."""
    from npu_engine.launch_chat import main as launch_main
    launch_main()


def main():
    parser = argparse.ArgumentParser(description="Universal Qualcomm Hexagon NPU Layer Streaming CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Inspect
    p_inspect = subparsers.add_parser("inspect", help="Inspect model architecture and RAM budget")
    p_inspect.add_argument("model_path", help="Path to model directory or Hugging Face repo")

    # Convert
    p_convert = subparsers.add_parser("convert", help="Convert model layers to Crouton INT4 format")
    p_convert.add_argument("model_path", help="Path to model directory")
    p_convert.add_argument("--output-dir", default=None, help="Output directory for npu layers")
    p_convert.add_argument("--layers", type=int, default=None, help="Number of layers to convert")

    # Chat
    subparsers.add_parser("chat", help="Launch ChatGPT-style web interface")

    args = parser.parse_args()
    if args.command == "inspect":
        cmd_inspect(args)
    elif args.command == "convert":
        cmd_convert(args)
    elif args.command == "chat":
        cmd_chat(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
