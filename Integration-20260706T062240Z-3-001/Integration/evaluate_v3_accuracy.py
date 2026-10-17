#!/usr/bin/env python3
"""
Phase 5: Quantitative Verification & Accuracy Benchmark (V3 Hybrid vs Ground Truth)
Evaluates pipeline_v3_hybrid.py across ISL videos against authentic human-annotated
SiGML ground truth files in SignFiles/.

Reference: HYBRID_NN_ARCHITECTURE_AND_TRAINING_SPECIFICATION.md - Phase 5
"""

import os
import sys
import json
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))

VIDEOS_DIR = os.path.join(_REPO_ROOT, "ISL_videos")
SIGML_DIR = os.path.join(_REPO_ROOT, "SignFiles")
LABELS_CSV = os.path.join(_SCRIPT_DIR, "isl_ground_truth_labels.csv")

from pipeline_v3_hybrid import get_v3_pipeline

def evaluate_pipeline(num_samples=25):
    print("=" * 65)
    print("  Phase 5: Quantitative Evaluation of V3 Hybrid Architecture")
    print(f"  Testing on {num_samples} ground-truth validated ISL video signs")
    print("=" * 65)
    
    if not os.path.exists(LABELS_CSV):
        raise FileNotFoundError(f"Missing {LABELS_CSV}")
        
    df = pd.read_csv(LABELS_CSV).dropna(subset=['handshape', 'movement']).reset_index(drop=True)
    
    # Stratified or randomized evaluation sample
    eval_df = df.sample(n=min(num_samples, len(df)), random_state=42).reset_index(drop=True)
    
    engine = get_v3_pipeline()
    
    metrics = {
        "handshape": {"correct": 0, "total": 0},
        "movement": {"correct": 0, "total": 0},
        "symmetry": {"correct": 0, "total": 0},
        "ext_finger": {"correct": 0, "total": 0},
        "palm_ori": {"correct": 0, "total": 0},
        "location": {"correct": 0, "total": 0},
    }
    
    results = []
    
    for idx, row in eval_df.iterrows():
        vid_path = os.path.join(VIDEOS_DIR, row['video_file'])
        if not os.path.exists(vid_path):
            continue
            
        sign_name = row.get('gloss', row['video_file'])
        try:
            hns_pred, sigml_pred = engine.process_video(vid_path)
            tokens = hns_pred.split()
        except Exception as e:
            print(f"  [Error] Processing {sign_name}: {e}")
            continue
            
        gt_hs = row['handshape']
        gt_mv = row['movement']
        gt_sym = row['two_handed']
        gt_ext = row.get('ext_finger', None)
        gt_palm = row.get('palm_ori', None)
        gt_loc = row.get('location', None)
        
        # Check matches
        pred_hs = next((t for t in tokens if t in engine.hs_classes), None)
        pred_mv = next((t for t in tokens if t in engine.mv_classes), "hamnomotion")
        pred_sym = next((t for t in tokens if t in ("hamsymmlr", "hamplus")), "none")
        pred_ext = next((t for t in tokens if t.startswith("hamextfinger")), None)
        pred_palm = next((t for t in tokens if t.startswith("hampalm")), None)
        pred_loc = next((t for t in tokens if t in ("hamchest", "hamshoulders", "hamshouldertop", "hamneutralspace", "hamforehead", "hamchin", "hamlips", "hamnose", "hamstomach")), None)
        
        hs_match = (pred_hs == gt_hs)
        mv_match = (pred_mv == gt_mv)
        sym_match = (pred_sym == gt_sym)
        ext_match = (pred_ext == gt_ext) if gt_ext and pd.notna(gt_ext) else None
        palm_match = (pred_palm == gt_palm) if gt_palm and pd.notna(gt_palm) else None
        loc_match = (pred_loc == gt_loc) if gt_loc and pd.notna(gt_loc) else None
        
        metrics["handshape"]["total"] += 1
        if hs_match: metrics["handshape"]["correct"] += 1
        
        metrics["movement"]["total"] += 1
        if mv_match: metrics["movement"]["correct"] += 1
        
        metrics["symmetry"]["total"] += 1
        if sym_match: metrics["symmetry"]["correct"] += 1
        
        if ext_match is not None:
            metrics["ext_finger"]["total"] += 1
            if ext_match: metrics["ext_finger"]["correct"] += 1
            
        if palm_match is not None:
            metrics["palm_ori"]["total"] += 1
            if palm_match: metrics["palm_ori"]["correct"] += 1
            
        if loc_match is not None:
            metrics["location"]["total"] += 1
            if loc_match: metrics["location"]["correct"] += 1
            
        results.append({
            "sign": sign_name,
            "pred_hns": hns_pred,
            "gt_hs": gt_hs, "pred_hs": pred_hs, "hs_ok": hs_match,
            "gt_mv": gt_mv, "pred_mv": pred_mv, "mv_ok": mv_match,
            "gt_sym": gt_sym, "pred_sym": pred_sym, "sym_ok": sym_match
        })
        
        status_sym = "[OK]" if (hs_match and mv_match) else "[PARTIAL]"
        print(f"  {status_sym} {sign_name:18s} | HS: {str(pred_hs):14s} (GT: {gt_hs:14s}) | MV: {str(pred_mv):12s} (GT: {gt_mv:12s})")
        
    print("\n" + "=" * 65)
    print("  QUANTITATIVE ACCURACY SUMMARY (V3 HYBRID)")
    print("=" * 65)
    
    total_score = 0
    weights_sum = 0
    for comp, stats in metrics.items():
        if stats["total"] > 0:
            acc = (stats["correct"] / stats["total"]) * 100.0
            print(f"  {comp.upper():16s}: {stats['correct']:3d} / {stats['total']:3d}  -->  {acc:6.2f}%")
            total_score += acc
            weights_sum += 1
            
    composite_acc = total_score / max(weights_sum, 1)
    print("-" * 65)
    print(f"  OVERALL COMPOSITE ACCURACY: {composite_acc:.2f}%")
    print("=" * 65)
    
    # Save results to json
    res_path = os.path.join(_SCRIPT_DIR, "v3_evaluation_results.json")
    with open(res_path, "w") as f:
        json.dump({"metrics": metrics, "composite_acc": composite_acc, "samples": results}, f, indent=2)
    print(f"Detailed evaluation saved -> {res_path}")

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    evaluate_pipeline(num_samples=n)
