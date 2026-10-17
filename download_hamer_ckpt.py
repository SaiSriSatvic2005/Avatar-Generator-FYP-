#!/usr/bin/env python3
"""
Downloads hamer.ckpt from Hugging Face using huggingface_hub with resumable chunks.
"""

import os
import sys
from huggingface_hub import hf_hub_download

def main():
    root = os.path.dirname(os.path.abspath(__file__))
    dest_dir = os.path.join(root, "hamer", "_DATA", "hamer_ckpts", "checkpoints")
    os.makedirs(dest_dir, exist_ok=True)
    
    print("[HF] Starting resumable download of hamer.ckpt from Leo-TX/hamer...")
    ckpt_path = hf_hub_download(
        repo_id="Leo-TX/hamer",
        filename="hamer.ckpt",
        local_dir=dest_dir,
        resume_download=True
    )
    print(f"[SUCCESS] Checkpoint downloaded to {ckpt_path} ({os.path.getsize(ckpt_path)/(1024*1024):.1f} MB)")

if __name__ == "__main__":
    main()
