# --- Fix Windows DLL conflict: MediaPipe must be first ---
import mediapipe as mp
# ---------------------------------------------------------
#!/usr/bin/env python3
"""
Benchmark Evaluation Script for V8 Hybrid 3D Sign Language Recognition Pipeline.
Evaluates:
  1. Handshape Classification Accuracy (Model A: SkeletalHandResNet_V7 with 128-D local-canonical features)
  2. Movement Direction Accuracy (Model B: AttentionBiGRU with 18-D kinematics)
  3. Body Collision Prevention & Contact State (No clipping into torso)
  4. SiGML XML Syntactic and Parser Validity
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from local_hamer_pipeline import LocalHaMeRPipeline

def run_benchmark(num_videos=25):
    csv_path = os.path.join(ROOT_DIR, "downloaded_dgs_dataset", "dgs_clean_ground_truth.csv")
    video_dir = os.path.join(ROOT_DIR, "downloaded_dgs_dataset", "videos")

    df = pd.read_csv(csv_path).dropna(subset=["base_handshape", "movement"])
    df["full_path"] = df["video_file"].apply(lambda f: os.path.join(video_dir, str(f)))
    df = df[df["full_path"].apply(os.path.exists)].reset_index(drop=True)

    test_subset = df.head(num_videos)

    print("=" * 80)
    print(f"   STARTING V8 HYBRID BENCHMARK EVALUATION ON {len(test_subset)} DGS SIGN VIDEOS")
    print("=" * 80)
    print(f"Device: CPU (PyTorch {os.environ.get('TORCH_VERSION', 'local')})")
    print(f"Model A: SkeletalHandResNet_V7 (128-D Invariant Local-Canonical Features)")
    print(f"Model B: AttentionBiGRU (18-D Spatiotemporal Kinematic Trajectories)")
    print(f"Collision Protection: Active (Spurious hamtouch eliminated, torso clearance guarded)")
    print("-" * 80)

    t_init_start = time.time()
    pipeline = LocalHaMeRPipeline()
    t_init = time.time() - t_init_start
    print(f"[*] Pipeline initialized (Version: {getattr(pipeline, 'version', 'unknown').upper()}) in {t_init:.2f}s\n")

    results = []
    hs_correct = 0
    mv_correct = 0
    both_correct = 0
    xml_valid_count = 0
    collision_free_count = 0

    for idx, row in test_subset.iterrows():
        vpath = row["full_path"]
        vname = row["video_file"]
        gt_hs = row["base_handshape"]
        gt_mv = row["movement"]
        gloss = row.get("stem", row.get("gloss", "N/A"))

        print(f"[{idx+1:2d}/{len(test_subset)}] Processing: {vname} (Gloss: {gloss})...")
        t0 = time.time()
        try:
            pred = pipeline.process_video(vpath)
            dt = time.time() - t0

            pred_hs = pred.get("handshape", "unknown")
            pred_mv = pred.get("movement", "unknown")
            is_valid_xml = pred.get("is_valid_xml", False)
            hns = pred.get("hamnosys", "")

            # Collision check: no hamtouch at chest/shoulders
            is_collision_free = not ("hamtouch" in hns and ("hamshoulders" in hns or "hamchest" in hns))
            if is_collision_free:
                collision_free_count += 1

            match_hs = (pred_hs == gt_hs)
            match_mv = (pred_mv == gt_mv)
            match_both = (match_hs and match_mv)

            if match_hs:
                hs_correct += 1
            if match_mv:
                mv_correct += 1
            if match_both:
                both_correct += 1
            if is_valid_xml:
                xml_valid_count += 1

            status_hs = "MATCH" if match_hs else "MISMATCH"
            status_mv = "MATCH" if match_mv else "MISMATCH"

            print(f"       Handshape: Pred={pred_hs:<17} | GT={gt_hs:<17} -> [{status_hs}]")
            print(f"       Movement:  Pred={pred_mv:<17} | GT={gt_mv:<17} -> [{status_mv}]")
            print(f"       SiGML XML: Valid={is_valid_xml} | CollisionFree={is_collision_free} | Time={dt:.2f}s\n")

            results.append({
                "video_file": vname,
                "gloss": gloss,
                "gt_handshape": gt_hs,
                "pred_handshape": pred_hs,
                "hs_match": match_hs,
                "gt_movement": gt_mv,
                "pred_movement": pred_mv,
                "mv_match": match_mv,
                "both_match": match_both,
                "is_valid_xml": is_valid_xml,
                "collision_free": is_collision_free,
                "time_sec": round(dt, 2),
                "hamnosys": hns,
                "sigml_xml": pred.get("sigml_xml", "")
            })

        except Exception as e:
            print(f"       ERROR processing {vname}: {e}\n")
            results.append({
                "video_file": vname,
                "gloss": gloss,
                "error": str(e)
            })

    total_eval = len(results)
    hs_acc = (hs_correct / total_eval) * 100 if total_eval > 0 else 0
    mv_acc = (mv_correct / total_eval) * 100 if total_eval > 0 else 0
    both_acc = (both_correct / total_eval) * 100 if total_eval > 0 else 0
    xml_rate = (xml_valid_count / total_eval) * 100 if total_eval > 0 else 0
    cf_rate = (collision_free_count / total_eval) * 100 if total_eval > 0 else 0

    print("=" * 80)
    print("                      V8 BENCHMARK ACCURACY SUMMARY                        ")
    print("=" * 80)
    print(f"  Total Videos Tested:               {total_eval}")
    print(f"  Handshape Accuracy (Top-1):        {hs_correct}/{total_eval} ({hs_acc:.1f}%)")
    print(f"  Movement Accuracy (Top-1):         {mv_correct}/{total_eval} ({mv_acc:.1f}%)")
    print(f"  Full Sign Compound Match:          {both_correct}/{total_eval} ({both_acc:.1f}%)")
    print(f"  Collision-Free Posture Rate:       {collision_free_count}/{total_eval} ({cf_rate:.1f}%)")
    print(f"  Valid SiGML XML Rate:              {xml_valid_count}/{total_eval} ({xml_rate:.1f}%)")
    print("=" * 80)

    output_summary = {
        "version": "V8",
        "total_evaluated": total_eval,
        "handshape_accuracy_pct": round(hs_acc, 2),
        "movement_accuracy_pct": round(mv_acc, 2),
        "compound_accuracy_pct": round(both_acc, 2),
        "collision_free_rate_pct": round(cf_rate, 2),
        "xml_validity_rate_pct": round(xml_rate, 2),
        "detailed_results": results
    }

    out_file = os.path.join(ROOT_DIR, "v8_benchmark_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_summary, f, indent=2)

    print(f"\n[SAVED] V8 Benchmark evaluation exported to: {out_file}")

if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    run_benchmark(num_videos=count)
