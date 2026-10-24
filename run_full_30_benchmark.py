#!/usr/bin/env python3
"""
Full 30-Video Benchmark Runner & Live Report Updater for V6 Hybrid System.
Processes all 30 DGS videos with the upgraded LocalHaMeRPipeline (Bottlenecks 1-6 applied).
Incrementally saves SiGML XML, updates JSON, and regenerates V6_BENCHMARK_30_VIDEOS_ACCURACY_REPORT.md.
"""

import os
import sys
import time
import json
import xml.etree.ElementTree as ET
import pandas as pd
import torch
from pathlib import Path

# Set PyTorch to utilize all CPU cores
torch.set_num_threads(12)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from local_hamer_pipeline import LocalHaMeRPipeline
import generate_updated_report

DGS_DIR = os.path.join(ROOT_DIR, "downloaded_dgs_dataset")
GT_SIGML_DIR = os.path.join(DGS_DIR, "sigml")
VIDEOS_DIR = os.path.join(DGS_DIR, "videos")
CSV_PATH = os.path.join(DGS_DIR, "dgs_clean_ground_truth.csv")
OUT_JSON_FILE = os.path.join(ROOT_DIR, "v6_benchmark_results_30.json")

def run_benchmark(force_all=False):
    df = pd.read_csv(CSV_PATH).dropna(subset=["base_handshape", "movement"])
    test_30 = df.head(30).reset_index(drop=True)

    print("=" * 80, flush=True)
    print(f"   STARTING FULL 30-VIDEO BENCHMARK RUN (TORCH THREADS = {torch.get_num_threads()})", flush=True)
    print("=" * 80, flush=True)

    pipeline = None
    t_start = time.time()

    for idx, row in test_30.iterrows():
        vname = row["video_file"]
        stem = Path(vname).stem
        gt_hs = row["base_handshape"]
        gt_mv = row["movement"]
        gloss = row.get("stem", row.get("gloss", stem))
        vpath = os.path.join(VIDEOS_DIR, vname)
        pred_sigml_path = os.path.join(ROOT_DIR, f"{stem}_hamer_output.sigml")

        # Check if already processed in this recent update session (within last 30 minutes)
        is_fresh = False
        if not force_all and os.path.exists(pred_sigml_path):
            file_age_sec = time.time() - os.path.getmtime(pred_sigml_path)
            if file_age_sec < 1800: # Created in the last 30 minutes
                is_fresh = True

        if is_fresh:
            print(f"[{idx+1:2d}/30] Using Freshly Recomputed Result: {vname} (Gloss: {gloss})", flush=True)
            with open(pred_sigml_path, "r", encoding="utf-8") as f:
                pred_xml = f.read()
        else:
            if pipeline is None:
                print("\n[*] Initializing LocalHaMeRPipeline engine...", flush=True)
                pipeline = LocalHaMeRPipeline()
                print("[*] LocalHaMeRPipeline engine ready.\n", flush=True)

            print(f"[{idx+1:2d}/30] Processing: {vname} (Gloss: {gloss})...", flush=True)
            t0 = time.time()
            res = pipeline.process_video(vpath)
            dt = time.time() - t0

            pred_hs = res.get("handshape", "unknown")
            pred_mv = res.get("movement", "unknown")
            pred_xml = res.get("sigml_xml", "")
            hamnosys = res.get("hamnosys", "")

            # Save SiGML to disk
            with open(pred_sigml_path, "w", encoding="utf-8") as f:
                f.write(pred_xml)

            hs_m = "MATCH" if pred_hs == gt_hs else "MISMATCH"
            mv_m = "MATCH" if pred_mv == gt_mv else "MISMATCH"
            print(f"       -> Pred HS: {pred_hs:<17} | GT: {gt_hs:<17} [{hs_m}]", flush=True)
            print(f"       -> Pred MV: {pred_mv:<17} | GT: {gt_mv:<17} [{mv_m}]", flush=True)
            print(f"       -> Time: {dt:.2f}s | HamNoSys: {hamnosys}\n", flush=True)

        # Incrementally rebuild the full markdown report and json data
        try:
            generate_updated_report.build_full_report()
        except Exception as e:
            print(f"[Warning] Failed to update report at step {idx+1}: {e}", flush=True)

    total_time = time.time() - t_start
    print("=" * 80, flush=True)
    print(f"   ALL 30 VIDEOS COMPLETED IN {total_time/60:.2f} MINUTES", flush=True)
    print("=" * 80, flush=True)

    # Final report update
    generate_updated_report.build_full_report()
    print("[DONE] Final report and json successfully generated.", flush=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-all", action="store_true", help="Force recomputation of all 30 videos")
    args = parser.parse_args()
    run_benchmark(force_all=args.force_all)
