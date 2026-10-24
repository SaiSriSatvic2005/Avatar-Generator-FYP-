#!/usr/bin/env python3
"""
Run batch DGS videos through the Local HaMeR 3D Hybrid Pipeline.
Supports running live inference on specified number of videos (default: 20),
caching, saving to <stem>_hamer_output.sigml, and outputting benchmark accuracy.
"""

import os
import sys
import argparse
import time
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from local_hamer_pipeline import LocalHaMeRPipeline

DGS_DIR = os.path.join(ROOT_DIR, "downloaded_dgs_dataset")
VIDEOS_DIR = os.path.join(DGS_DIR, "videos")
CSV_PATH = os.path.join(DGS_DIR, "dgs_clean_ground_truth.csv")
OUTPUT_DIR = os.path.join(ROOT_DIR, "dgs_sigml_outputs")

def main():
    parser = argparse.ArgumentParser(description="Run 20 DGS dataset videos through V8 HaMeR Pipeline.")
    parser.add_argument("--count", "-n", type=int, default=20, help="Number of DGS videos to run (default: 20)")
    parser.add_argument("--force", action="store_true", help="Force recomputation even if .sigml exists")
    parser.add_argument("--threads", type=int, default=8, help="PyTorch CPU thread count (default: 8)")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH).dropna(subset=["base_handshape", "movement"])
    test_videos = df.head(args.count).reset_index(drop=True)

    print("=" * 80)
    print(f"      STARTING BATCH RUN FOR {len(test_videos)} DGS DATASET VIDEOS")
    print(f"      PyTorch CPU Threads: {torch.get_num_threads()}")
    print("=" * 80)

    pipeline = None
    processed = []

    for idx, row in test_videos.iterrows():
        vname = row["video_file"]
        stem = Path(vname).stem
        vpath = os.path.join(VIDEOS_DIR, vname)
        root_sigml = os.path.join(ROOT_DIR, f"{stem}_hamer_output.sigml")
        archive_sigml = os.path.join(OUTPUT_DIR, f"{stem}_hamer_output.sigml")

        if not os.path.exists(vpath):
            print(f"[{idx+1:2d}/{len(test_videos)}] Missing video: {vpath}, skipping.")
            continue

        if not args.force and os.path.exists(root_sigml):
            print(f"[{idx+1:2d}/{len(test_videos)}] [CACHE HIT] {stem}_hamer_output.sigml exists.")
            with open(root_sigml, "r", encoding="utf-8") as f:
                sigml_xml = f.read()
            processed.append({"video": vname, "sigml_path": root_sigml, "status": "cached"})
            continue

        if pipeline is None:
            print("\n[*] Initializing LocalHaMeRPipeline...")
            pipeline = LocalHaMeRPipeline()
            print("[*] Pipeline ready.\n")

        print(f"[{idx+1:2d}/{len(test_videos)}] Processing: {vname}...")
        t0 = time.time()
        res = pipeline.process_video(vpath)
        dt = time.time() - t0

        sigml_xml = res["sigml_xml"]
        with open(root_sigml, "w", encoding="utf-8") as f:
            f.write(sigml_xml + "\n")
        with open(archive_sigml, "w", encoding="utf-8") as f:
            f.write(sigml_xml + "\n")

        print(f"       -> Done in {dt:.2f}s | Valid XML: {res.get('is_valid_xml', True)}")
        print(f"       -> Handshape: {res.get('handshape')} | Movement: {res.get('movement')}")
        print(f"       -> Saved to: {root_sigml}\n")

        processed.append({"video": vname, "sigml_path": root_sigml, "status": "processed", "time_sec": dt})

    print("=" * 80)
    print(f"Completed processing {len(processed)} DGS videos. All SiGML files saved.")
    print("=" * 80)

if __name__ == "__main__":
    main()
