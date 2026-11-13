import os
from setuptools import setup, find_packages

setup(
    name="unlocking-the-power-of-npu",
    version="2.0.0",
    description="Universal Qualcomm Hexagon NPU Layer-Streaming Engine for Snapdragon X Plus",
    author="Viswanath Kasi",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "psutil>=5.9.0",
        "huggingface-hub>=0.20.0",
        "tokenizers>=0.15.0",
    ],
    extras_require={
        "safetensors": ["safetensors>=0.4.0"],
        "onnx": ["onnxruntime>=1.16.0"],
    },
    entry_points={
        "console_scripts": [
            "npu-stream=npu_engine.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Microsoft :: Windows",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
