#!/usr/bin/env python3
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from integration_pipeline import process_video_neural

if __name__ == "__main__":
    vpath = sys.argv[1] if len(sys.argv) > 1 else "ISL_sign_videos/00384.mp4"
    print(f"Testing Neural Inference on: {vpath}")
    hamnosys_result = process_video_neural(vpath)
    print("\n==========================================")
    print(f" PREDICTED HAMNOSYS: {hamnosys_result}")
    print("==========================================\n")
