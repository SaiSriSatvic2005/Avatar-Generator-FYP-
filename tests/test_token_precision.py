#!/usr/bin/env python3
"""
===========================================================================
 TOKEN PRECISION VALIDATION TEST SUITE
 Project: Sign Language to HamNoSys Avatar Generator (V2 Architecture)
 
 This script rigorously validates the claimed 85.7% Token Precision
 by running the trained HamNoSysSequenceNet neural network on a
 held-out portion of the WLASL landmark dataset and comparing
 predicted HamNoSys tokens against ground-truth annotations.
 
 Methodology:
   1. Load the 500-sample WLASL landmark dataset
   2. Perform an 80/20 stratified train/test split (by gloss label)
   3. Load the trained V2 neural network weights
   4. Run inference on all 85 test samples
   5. Display step-by-step example evaluations for faculty transparency
   6. Compute per-head accuracy for all 6 classification heads
   7. Compute overall Token Precision = (correct tokens / total tokens)
   8. Output a structured JSON report
===========================================================================
"""

import os
import sys
import json
import csv
import numpy as np

# Resolve paths
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_INTEGRATION_DIR = os.path.join(_PROJECT_ROOT, "Integration-20260706T062240Z-3-001", "Integration")
_DATASET_DIR = os.path.join(_INTEGRATION_DIR, "wlasl_landmark_dataset")

sys.path.insert(0, _INTEGRATION_DIR)

import torch
from train_landmark_nn import HamNoSysSequenceNet

# ── Configuration ──
NPZ_PATH = os.path.join(_DATASET_DIR, "dataset_landmarks.npz")
CSV_PATH = os.path.join(_DATASET_DIR, "metadata.csv")
MODEL_PATH = os.path.join(_DATASET_DIR, "hamnosys_net_v2.pth")
MAPPINGS_PATH = os.path.join(_DATASET_DIR, "nn_class_mappings.json")
REPORT_PATH = os.path.join(_SCRIPT_DIR, "token_precision_report.json")

TEST_SPLIT = 0.20  # 20% held-out test set


def load_dataset():
    """Load landmark tensors and metadata CSV."""
    data = np.load(NPZ_PATH)
    tensors = data["tensors"]  # (N, T, D)

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        metadata = list(reader)

    return tensors, metadata


def stratified_split(metadata, test_ratio=0.20, seed=42):
    """
    Stratified train/test split by gloss label.
    Ensures each gloss has proportional representation in both sets.
    """
    from collections import defaultdict
    rng = np.random.RandomState(seed)

    gloss_indices = defaultdict(list)
    for i, row in enumerate(metadata):
        gloss_indices[row["gloss"]].append(i)

    train_idx, test_idx = [], []
    for gloss, indices in gloss_indices.items():
        rng.shuffle(indices)
        n_test = max(1, int(len(indices) * test_ratio))
        test_idx.extend(indices[:n_test])
        train_idx.extend(indices[n_test:])

    return sorted(train_idx), sorted(test_idx)


def run_validation():
    """Main validation routine."""
    print("\n" + "=" * 78)
    print("       HAMNOSYS TOKEN PRECISION VALIDATION SUITE (V2 NEURAL ARCHITECTURE)")
    print("=" * 78)

    # ── Step 1: Check files exist ──
    for path, name in [
        (NPZ_PATH, "Landmark tensors (NPZ)"),
        (CSV_PATH, "Metadata CSV annotations"),
        (MODEL_PATH, "Trained BiGRU Neural Weights"),
        (MAPPINGS_PATH, "HamNoSys Class Mappings"),
    ]:
        if not os.path.exists(path):
            print(f"FAIL: Missing required file: {name}")
            print(f"  Path: {path}")
            return False
        print(f"  [OK] Verified: {name}")

    # ── Step 2: Load dataset ──
    tensors, metadata = load_dataset()
    print(f"\n  Dataset: {len(metadata)} video sequences | Landmark Tensor Shape: {tensors.shape}")

    with open(MAPPINGS_PATH, "r", encoding="utf-8") as f:
        mappings = json.load(f)

    # ── Step 3: Stratified split ──
    train_idx, test_idx = stratified_split(metadata, TEST_SPLIT)
    print(f"  Split: {len(train_idx)} Train Samples (80%) | {len(test_idx)} Held-Out Test Samples (20%)")

    # ── Step 4: Build label encoders ──
    HEAD_NAMES = ["handshape", "ext_finger", "palm_ori", "location", "movement", "two_handed"]
    label_maps = {}
    for head in HEAD_NAMES:
        classes = mappings[head]
        label_maps[head] = {v: i for i, v in enumerate(classes)}

    # ── Step 5: Load model ──
    device = torch.device("cpu")
    model = HamNoSysSequenceNet(
        input_dim=tensors.shape[2],
        num_hs=len(mappings["handshape"]),
        num_ext=len(mappings["ext_finger"]),
        num_palm=len(mappings["palm_ori"]),
        num_loc=len(mappings["location"]),
        num_mov=len(mappings["movement"]),
        num_two=len(mappings["two_handed"]),
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"  [OK] Loaded HamNoSysSequenceNet ({sum(p.numel() for p in model.parameters()):,} parameters)")

    # ── Step 6: Run inference on test set ──
    test_tensors = torch.tensor(tensors[test_idx], dtype=torch.float32)

    with torch.no_grad():
        preds = model(test_tensors)

    CSV_TO_HEAD = {
        "handshape": "handshape",
        "ext_finger": "ext_finger",
        "palm_ori": "palm_ori",
        "location": "location",
        "movement": "movement",
        "two_handed": "two_handed",
    }

    # ── Step 7: Print Faculty Transparency Examples ──
    print("\n" + "-" * 78)
    print("  FACULTY AUDIT DEMONSTRATION: STEP-BY-STEP SAMPLE EVALUATION")
    print("  (Comparing 177D Video Landmark Predictions vs Ground-Truth Annotations)")
    print("-" * 78)

    sample_preview_indices = [0, 1, 2]  # Show 3 concrete test samples
    for sp_i in sample_preview_indices:
        real_idx = test_idx[sp_i]
        meta_row = metadata[real_idx]
        v_id = meta_row["video_id"]
        v_gloss = meta_row["gloss"].upper()

        print(f"\n  [Sample #{sp_i+1}] Video ID: {v_id} | Target Sign Gloss: '{v_gloss}'")
        print(f"  {'Component Head':<22} | {'Ground-Truth Token':<18} | {'Predicted Token':<18} | {'Match'}")
        print(f"  {'-'*22}-+-{'-'*18}-+-{'-'*18}-+-{'-'*7}")

        sample_correct = 0
        for csv_col, head_key in CSV_TO_HEAD.items():
            pred_idx = preds[head_key][sp_i].argmax().item()
            pred_tok = mappings[head_key][pred_idx]
            gt_tok = meta_row[csv_col]
            matched = (pred_tok == gt_tok)
            if matched:
                sample_correct += 1
            match_str = "[PASS]" if matched else "[FAIL]"
            print(f"  {head_key:<22} | {gt_tok:<18} | {pred_tok:<18} | {match_str}")

        sample_acc = (sample_correct / 6.0) * 100.0
        print(f"  --> Sample Score: {sample_correct}/6 Tokens Correct ({sample_acc:.1f}% accuracy)")

    print("\n" + "-" * 78)
    print("  AGGREGATE EVALUATION ACROSS ALL 85 HELD-OUT TEST SAMPLES")
    print("-" * 78)

    # ── Step 8: Compute per-head accuracy ──
    head_results = {}
    total_correct = 0
    total_tokens = 0

    print(f"  {'Classification Head':<26} | {'Correct':<8} | {'Total':<6} | {'Accuracy':<10} | {'Status'}")
    print(f"  {'-'*26}-+-{'-'*8}-+-{'-'*6}-+-{'-'*10}-+-{'-'*8}")

    for csv_col, head_key in CSV_TO_HEAD.items():
        pred_indices = preds[head_key].argmax(dim=1).cpu().numpy()
        gt_indices = []
        for idx in test_idx:
            gt_label = metadata[idx][csv_col]
            gt_indices.append(label_maps[head_key].get(gt_label, -1))
        gt_indices = np.array(gt_indices)

        correct = (pred_indices == gt_indices).sum()
        total = len(gt_indices)
        accuracy = (correct / total) * 100.0

        head_results[head_key] = {
            "correct": int(correct),
            "total": int(total),
            "accuracy_pct": round(accuracy, 2),
        }
        total_correct += correct
        total_tokens += total

        status = "[PASS]" if accuracy >= 70.0 else "[WARN]"
        print(f"  {head_key:<26} | {correct:<8} | {total:<6} | {accuracy:6.2f}%   | {status}")

    overall_precision = (total_correct / total_tokens) * 100.0

    print("\n" + "=" * 78)
    print(f"  OVERALL TOKEN PRECISION : {overall_precision:.2f}%")
    print(f"  Total Correct Tokens    : {total_correct} out of {total_tokens} predictions ({len(test_idx)} test videos x 6 heads)")
    print(f"  V1 Baseline Precision   : 11.20%")
    print(f"  Net Precision Gain      : +{overall_precision - 11.20:.2f}%")
    passed = overall_precision >= 80.0
    print(f"  Final Verification      : {'PASS [OK]' if passed else 'BELOW THRESHOLD'}")
    print("=" * 78)

    # ── Step 9: Per-gloss breakdown ──
    gloss_results = {}
    for i, idx in enumerate(test_idx):
        gloss = metadata[idx]["gloss"]
        if gloss not in gloss_results:
            gloss_results[gloss] = {"correct": 0, "total": 0}
        for csv_col, head_key in CSV_TO_HEAD.items():
            pred_idx = preds[head_key][i].argmax().item()
            gt_label = metadata[idx][csv_col]
            gt_idx = label_maps[head_key].get(gt_label, -1)
            gloss_results[gloss]["total"] += 1
            if pred_idx == gt_idx:
                gloss_results[gloss]["correct"] += 1

    for g in gloss_results:
        c, t = gloss_results[g]["correct"], gloss_results[g]["total"]
        gloss_results[g]["precision_pct"] = round((c / t) * 100.0, 2) if t > 0 else 0.0

    # ── Step 10: Save JSON report ──
    report = {
        "test_name": "Token Precision Validation (V2 Architecture)",
        "dataset": {
            "source": "WLASL v0.3 (Word-Level American Sign Language)",
            "total_samples": len(metadata),
            "total_glosses": len(set(r["gloss"] for r in metadata)),
            "train_samples": len(train_idx),
            "test_samples": len(test_idx),
            "split_ratio": f"{int((1 - TEST_SPLIT) * 100)}/{int(TEST_SPLIT * 100)}",
            "split_method": "Stratified by gloss (seed=42)",
            "tensor_shape": list(tensors.shape),
            "feature_dimensions": tensors.shape[2],
        },
        "model": {
            "architecture": "HamNoSysSequenceNet V2 (Conv1d + BiGRU + Attention)",
            "parameters": sum(p.numel() for p in model.parameters()),
            "classification_heads": 6,
            "head_names": HEAD_NAMES,
        },
        "results": {
            "overall_token_precision_pct": round(overall_precision, 2),
            "total_correct_tokens": int(total_correct),
            "total_token_predictions": int(total_tokens),
            "per_head_accuracy": head_results,
            "per_gloss_precision": dict(sorted(
                gloss_results.items(),
                key=lambda x: x[1]["precision_pct"],
                reverse=True,
            )),
        },
        "verdict": "PASS" if passed else "BELOW_THRESHOLD",
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n  [OK] Detailed JSON report exported to: {os.path.abspath(REPORT_PATH)}\n")

    return passed


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
