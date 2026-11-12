"""
One-Click Launcher for Snapdragon Hexagon NPU ChatGPT Interface
Starts the local server and opens your web browser.
"""

import sys
import os
import webbrowser
import threading
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import uvicorn
from npu_engine.server import app


def open_browser():
    time.sleep(1.2)
    url = "http://localhost:8000"
    print(f"\n[NPU-Engine] Launching web browser at {url}...")
    webbrowser.open(url)


def main():
    print("=" * 70)
    print(" Qualcomm Snapdragon Hexagon v73 NPU Chat Interface")
    print("=" * 70)
    print("Starting high-throughput, ultra-low RAM layer streaming server...")
    print("Point your browser to: http://localhost:8000\n")

    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    main()
