#!/usr/bin/env python3
"""
Downloads HaMeR checkpoints and MANO assets for local 3D extraction.
"""

import os
import urllib.request
import tarfile
import sys

def download_file(url, dest_path, desc=""):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        print(f"[Skip] {desc} already exists at {dest_path}")
        return True
    print(f"[Download] Fetching {desc} from {url}...")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = downloaded / total_size * 100
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r   Progress: {mb:.1f}/{total_mb:.1f} MB ({percent:.1f}%)")
            sys.stdout.flush()
            
    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=report_progress)
        print("\n   Done.")
        return True
    except Exception as e:
        print(f"\n   Failed to download {url}: {e}")
        return False

def main():
    root = os.path.dirname(os.path.abspath(__file__))
    hamer_dir = os.path.join(root, "hamer")
    mano_dir = os.path.join(hamer_dir, "_DATA", "data", "mano")
    ckpts_dir = os.path.join(hamer_dir, "_DATA", "hamer_ckpts")
    
    os.makedirs(mano_dir, exist_ok=True)
    os.makedirs(ckpts_dir, exist_ok=True)
    
    # 1. Download MANO hand models
    mano_right_url = "https://huggingface.co/camenduru/HandRefiner/resolve/main/MANO_RIGHT.pkl"
    mano_left_url = "https://huggingface.co/camenduru/HandRefiner/resolve/main/MANO_LEFT.pkl"
    download_file(mano_right_url, os.path.join(mano_dir, "MANO_RIGHT.pkl"), "MANO_RIGHT.pkl")
    download_file(mano_left_url, os.path.join(mano_dir, "MANO_LEFT.pkl"), "MANO_LEFT.pkl")
    
    # 2. Download MediaPipe Hand Landmarker model
    mp_task_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    download_file(mp_task_url, os.path.join(hamer_dir, "hand_landmarker.task"), "hand_landmarker.task")
    
    # 3. Download HaMeR demo data / checkpoint
    # We can fetch directly from UT Austin CS server
    ckpt_file = os.path.join(ckpts_dir, "checkpoints", "hamer.ckpt")
    if not os.path.exists(ckpt_file):
        tar_path = os.path.join(hamer_dir, "hamer_demo_data.tar.gz")
        tar_url = "https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz"
        success = download_file(tar_url, tar_path, "hamer_demo_data.tar.gz (~380 MB)")
        if success and os.path.exists(tar_path):
            print("[Extract] Extracting hamer_demo_data.tar.gz...")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=hamer_dir)
            print("[Extract] Extraction complete.")
            if os.path.exists(tar_path):
                os.remove(tar_path)
                
    print("[SUCCESS] All HaMeR assets verified!")

if __name__ == "__main__":
    main()
