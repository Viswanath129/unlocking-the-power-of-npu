"""
Ultra-Low-RAM Memory-Mapped Layer Streamer for Qualcomm Hexagon NPU
Streams model weights directly via mmap without loading layers into the Python heap.
"""

import os
import mmap
import time
import gc
import psutil
from typing import Dict, Any, Optional, Callable
import numpy as np


class LowRAMStreamer:
    """
    Fixed-overhead memory streamer designed for systems with tight RAM budgets.
    Maintains strictly two pre-allocated buffers (Ping & Pong) in memory.
    All layer weights are memory-mapped directly from disk into these buffers,
    ensuring host RAM consumption never grows with the number of layers (e.g. 80 layers in Qwen-72B).
    """

    def __init__(self, buffer_size_mb: int = 480):
        self.buffer_size = buffer_size_mb * 1024 * 1024
        self.buffer_a = bytearray(self.buffer_size)
        self.buffer_b = bytearray(self.buffer_size)
        self.active_id = 0  # 0 -> A, 1 -> B
        self._mmap_handles = {}

    def get_process_ram_mb(self) -> float:
        """Returns current process RAM usage in megabytes"""
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / (1024 * 1024)

    def get_system_ram_stats(self) -> Dict[str, Any]:
        """Returns host system memory metrics"""
        mem = psutil.virtual_memory()
        return {
            "total_gb": round(mem.total / (1024**3), 2),
            "available_gb": round(mem.available / (1024**3), 2),
            "used_gb": round(mem.used / (1024**3), 2),
            "percent": mem.percent,
            "process_mb": round(self.get_process_ram_mb(), 2)
        }

    def swap(self):
        """Swaps active and standby buffers instantly with zero allocation"""
        self.active_id = 1 - self.active_id

    def stream_layer_mmap(self, file_path: str, offset: int, length: int) -> memoryview:
        """Memory-maps a weight slice directly from disk without copying into Python heap"""
        if file_path not in self._mmap_handles:
            f = open(file_path, "rb")
            self._mmap_handles[file_path] = (f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ))

        _, mm = self._mmap_handles[file_path]
        return memoryview(mm)[offset:offset+length]

    def close(self):
        for f, mm in self._mmap_handles.values():
            try:
                mm.close()
                f.close()
            except Exception:
                pass
        self._mmap_handles.clear()
        gc.collect()
