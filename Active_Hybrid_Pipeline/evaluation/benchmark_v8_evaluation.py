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

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(EVAL_DIR) if os.path.basename(EVAL_DIR) == "evaluation" else EVAL_DIR

PIPELINE_ENGINE_DIR = os.path.join(BASE_DIR, "pipeline_engine")
HEURISTICS_DIR = os.path.join(BASE_DIR, "heuristics")
HAMER_DIR = os.path.join(BASE_DIR, "hamer_3d_engine")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

for p in [BASE_DIR, PIPELINE_ENGINE_DIR, HEURISTICS_DIR, HAMER_DIR, WEIGHTS_DIR]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from local_hamer_pipeline import LocalHaMeRPipeline

def run_benchmark(num_videos=25):
    csv_candidates = [
        os.path.join(EVAL_DIR, "dgs_clean_ground_truth.csv"),
        os.path.join(BASE_DIR, "FYP-DataSet", "DGS", "dgs_clean_ground_truth.csv"),
        os.path.join(os.path.dirname(BASE_DIR), "downloaded_dgs_dataset", "dgs_clean_ground_truth.csv"),
    ]
    csv_path = next((p for p in csv_candidates if os.path.exists(p)), None)

    video_candidates = [
        os.path.join(EVAL_DIR, "sample_videos"),
        os.path.join(BASE_DIR, "FYP-DataSet", "DGS", "videos"),
        os.path.join(os.path.dirname(BASE_DIR), "downloaded_dgs_dataset", "videos"),
    ]
    video_dir = next((p for p in video_candidates if os.path.exists(p) and len(os.listdir(p)) > 0), None)

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
    print(f"Heuristics: SVD Palm Normal, Depth Pointing, 7-Zone Location, Clearance Guard")
    print("-" * 80)

    t_init_start = time.time()
    pipeline = LocalHaMeRPipeline()
    t_init = time.time() - t_init_start
    print(f"[*] Pipeline initialized (Version: {getattr(pipeline, 'version', 'unknown').upper()}) in {t_init:.2f}s\n")

    results = []
    
    # Module-wise metric counters
    hs_correct = 0
    ext_correct = 0
    palm_correct = 0
    loc_correct = 0
    contact_correct = 0
    sym_correct = 0
    mv_correct = 0
    
    both_correct = 0
    full_compound_correct = 0
    total_tokens_correct = 0
    total_tokens_evaluated = 0
    
    xml_valid_count = 0
    collision_free_count = 0

    for idx, row in test_subset.iterrows():
        vpath = row["full_path"]
        vname = row["video_file"]
        
        gt_hs = str(row.get("base_handshape", "")).strip()
        gt_ext = str(row.get("extended_finger", "")).strip()
        gt_palm = str(row.get("palm_orientation", "")).strip()
        gt_loc = str(row.get("location", "")).strip()
        gt_contact = bool(row.get("has_touch", False))
        gt_sym = str(row.get("symmetry_type", "single")).strip()
        gt_mv = str(row.get("movement", "")).strip()
        gloss = row.get("stem", row.get("gloss", "N/A"))

        print(f"[{idx+1:2d}/{len(test_subset)}] Processing: {vname} (Gloss: {gloss})...")
        t0 = time.time()
        try:
            pred = pipeline.process_video(vpath)
            dt = time.time() - t0

            pred_hs = str(pred.get("handshape", "unknown")).strip()
            pred_ext = str(pred.get("extended_finger", "unknown")).strip()
            pred_palm = str(pred.get("palm_orientation", "unknown")).strip()
            pred_loc = str(pred.get("location", "unknown")).strip()
            pred_contact = bool(pred.get("has_touch", False))
            pred_sym = str(pred.get("symmetry", "single")).strip()
            pred_mv = str(pred.get("movement", "unknown")).strip()
            is_valid_xml = pred.get("is_valid_xml", False)
            hns = pred.get("hamnosys", "")

            # Collision check: no hamtouch at chest/shoulders
            is_collision_free = not ("hamtouch" in hns and ("hamshoulders" in hns or "hamchest" in hns))
            if is_collision_free:
                collision_free_count += 1

            # Module-wise matching
            match_hs = (pred_hs == gt_hs)
            match_ext = (pred_ext == gt_ext)
            match_palm = (pred_palm == gt_palm)
            match_loc = (pred_loc == gt_loc)
            match_contact = (pred_contact == gt_contact)
            match_sym = (pred_sym == gt_sym or (gt_sym in ["single", "none"] and pred_sym in ["single", "none"]))
            match_mv = (pred_mv == gt_mv)

            if match_hs: hs_correct += 1
            if match_ext: ext_correct += 1
            if match_palm: palm_correct += 1
            if match_loc: loc_correct += 1
            if match_contact: contact_correct += 1
            if match_sym: sym_correct += 1
            if match_mv: mv_correct += 1

            # Token precision (6 primary slots)
            slot_matches = [match_hs, match_ext, match_palm, match_loc, match_contact, match_mv]
            total_tokens_correct += sum(slot_matches)
            total_tokens_evaluated += len(slot_matches)

            match_both = (match_hs and match_mv)
            if match_both: both_correct += 1
            
            match_full_compound = all(slot_matches)
            if match_full_compound: full_compound_correct += 1

            if is_valid_xml: xml_valid_count += 1

            print(f"       Handshape:  Pred={pred_hs:<17} | GT={gt_hs:<17} -> [{'MATCH' if match_hs else 'MISMATCH'}]")
            print(f"       Ext Finger: Pred={pred_ext:<17} | GT={gt_ext:<17} -> [{'MATCH' if match_ext else 'MISMATCH'}]")
            print(f"       Palm Ori:   Pred={pred_palm:<17} | GT={gt_palm:<17} -> [{'MATCH' if match_palm else 'MISMATCH'}]")
            print(f"       Location:   Pred={pred_loc:<17} | GT={gt_loc:<17} -> [{'MATCH' if match_loc else 'MISMATCH'}]")
            print(f"       Movement:   Pred={pred_mv:<17} | GT={gt_mv:<17} -> [{'MATCH' if match_mv else 'MISMATCH'}]")
            print(f"       SiGML XML:  Valid={is_valid_xml} | CollisionFree={is_collision_free} | Time={dt:.2f}s\n")

            results.append({
                "video_file": vname,
                "gloss": gloss,
                "gt_handshape": gt_hs,
                "pred_handshape": pred_hs,
                "hs_match": match_hs,
                "gt_extended_finger": gt_ext,
                "pred_extended_finger": pred_ext,
                "ext_match": match_ext,
                "gt_palm_orientation": gt_palm,
                "pred_palm_orientation": pred_palm,
                "palm_match": match_palm,
                "gt_location": gt_loc,
                "pred_location": pred_loc,
                "loc_match": match_loc,
                "gt_contact": gt_contact,
                "pred_contact": pred_contact,
                "contact_match": match_contact,
                "gt_movement": gt_mv,
                "pred_movement": pred_mv,
                "mv_match": match_mv,
                "both_match": match_both,
                "full_compound_match": match_full_compound,
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
    ext_acc = (ext_correct / total_eval) * 100 if total_eval > 0 else 0
    palm_acc = (palm_correct / total_eval) * 100 if total_eval > 0 else 0
    loc_acc = (loc_correct / total_eval) * 100 if total_eval > 0 else 0
    contact_acc = (contact_correct / total_eval) * 100 if total_eval > 0 else 0
    sym_acc = (sym_correct / total_eval) * 100 if total_eval > 0 else 0
    mv_acc = (mv_correct / total_eval) * 100 if total_eval > 0 else 0
    
    token_prec = (total_tokens_correct / total_tokens_evaluated) * 100 if total_tokens_evaluated > 0 else 0
    both_acc = (both_correct / total_eval) * 100 if total_eval > 0 else 0
    full_comp_acc = (full_compound_correct / total_eval) * 100 if total_eval > 0 else 0
    xml_rate = (xml_valid_count / total_eval) * 100 if total_eval > 0 else 0
    cf_rate = (collision_free_count / total_eval) * 100 if total_eval > 0 else 0

    print("=" * 80)
    print("                 V8 COMPREHENSIVE MODULE-WISE ACCURACY SUMMARY             ")
    print("=" * 80)
    print(f"  Total Videos Evaluated:            {total_eval}")
    print("-" * 80)
    print(f"  [Module A] Handshape Accuracy:     {hs_correct}/{total_eval} ({hs_acc:.1f}%)")
    print(f"  [Module B] Movement Accuracy:      {mv_correct}/{total_eval} ({mv_acc:.1f}%)")
    print(f"  [Heuristic 1] Palm Orientation:    {palm_correct}/{total_eval} ({palm_acc:.1f}%)")
    print(f"  [Heuristic 2] Ext. Finger Vector:  {ext_correct}/{total_eval} ({ext_acc:.1f}%)")
    print(f"  [Heuristic 3] Body Location:       {loc_correct}/{total_eval} ({loc_acc:.1f}%)")
    print(f"  [Heuristic 4] Contact State:       {contact_correct}/{total_eval} ({contact_acc:.1f}%)")
    print(f"  [Heuristic 5] Dual-Hand Symmetry:  {sym_correct}/{total_eval} ({sym_acc:.1f}%)")
    print("-" * 80)
    print(f"  Overall Phonetic Token Precision:  {total_tokens_correct}/{total_tokens_evaluated} ({token_prec:.1f}%)")
    print(f"  Compound Match (Handshape + Move): {both_correct}/{total_eval} ({both_acc:.1f}%)")
    print(f"  Full 6-Slot Compound Exact Match:  {full_compound_correct}/{total_eval} ({full_comp_acc:.1f}%)")
    print(f"  Collision-Free Posture Rate:       {collision_free_count}/{total_eval} ({cf_rate:.1f}%)")
    print(f"  Valid SiGML XML Generation Rate:   {xml_valid_count}/{total_eval} ({xml_rate:.1f}%)")
    print("=" * 80)

    output_summary = {
        "version": "V8",
        "total_evaluated": total_eval,
        "module_accuracy": {
            "handshape_pct": round(hs_acc, 2),
            "movement_pct": round(mv_acc, 2),
            "palm_orientation_pct": round(palm_acc, 2),
            "extended_finger_pct": round(ext_acc, 2),
            "body_location_pct": round(loc_acc, 2),
            "contact_state_pct": round(contact_acc, 2),
            "symmetry_pct": round(sym_acc, 2)
        },
        "pipeline_accuracy": {
            "token_precision_pct": round(token_prec, 2),
            "compound_hs_mv_pct": round(both_acc, 2),
            "full_compound_exact_pct": round(full_comp_acc, 2),
            "collision_free_rate_pct": round(cf_rate, 2),
            "xml_validity_rate_pct": round(xml_rate, 2)
        },
        "detailed_results": results
    }

    out_file = os.path.join(ROOT_DIR, "v8_benchmark_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_summary, f, indent=2)

    print(f"\n[SAVED] V8 Benchmark evaluation exported to: {out_file}")

if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    run_benchmark(num_videos=count)
