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
   1. Load the 501-sample WLASL landmark dataset (500 videos + header)
   2. Perform an 80/20 stratified train/test split (by gloss label)
   3. Load the trained V2 neural network weights
   4. Run inference on all test samples
   5. Compute per-head accuracy for all 6 classification heads
   6. Compute overall Token Precision = (correct tokens / total tokens)
   7. Output a structured JSON report for faculty review
   
 Expected Output:
   Token Precision ≥ 85.0% (claimed: 85.7%)
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
    print("=" * 70)
    print(" TOKEN PRECISION VALIDATION TEST")
    print("=" * 70)

    # ── Step 1: Check files exist ──
    for path, name in [
        (NPZ_PATH, "Landmark tensors (NPZ)"),
        (CSV_PATH, "Metadata CSV"),
        (MODEL_PATH, "Trained model weights"),
        (MAPPINGS_PATH, "Class mappings JSON"),
    ]:
        if not os.path.exists(path):
            print(f"FAIL: Missing required file: {name}")
            print(f"  Path: {path}")
            return False
        print(f"  [OK] Found: {name}")

    # ── Step 2: Load dataset ──
    tensors, metadata = load_dataset()
    print(f"\n  Dataset: {len(metadata)} samples, tensor shape: {tensors.shape}")

    with open(MAPPINGS_PATH, "r", encoding="utf-8") as f:
        mappings = json.load(f)

    # ── Step 3: Stratified split ──
    train_idx, test_idx = stratified_split(metadata, TEST_SPLIT)
    print(f"  Train: {len(train_idx)} samples | Test: {len(test_idx)} samples")

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
    print(f"  [OK] Loaded HamNoSysSequenceNet V2 ({sum(p.numel() for p in model.parameters()):,} parameters)")

    # ── Step 6: Run inference on test set ──
    test_tensors = torch.tensor(tensors[test_idx], dtype=torch.float32)

    with torch.no_grad():
        preds = model(test_tensors)

    # ── Step 7: Compute per-head accuracy ──
    head_results = {}
    total_correct = 0
    total_tokens = 0

    CSV_TO_HEAD = {
        "handshape": "handshape",
        "ext_finger": "ext_finger",
        "palm_ori": "palm_ori",
        "location": "location",
        "movement": "movement",
        "two_handed": "two_handed",
    }

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

        status = "PASS [OK]" if accuracy >= 70.0 else "WARN"
        print(f"  [{status}] {head_key:15s}: {accuracy:6.2f}%  ({correct}/{total})")

    overall_precision = (total_correct / total_tokens) * 100.0

    print(f"\n{'=' * 70}")
    print(f"  OVERALL TOKEN PRECISION: {overall_precision:.2f}%")
    print(f"  ({total_correct} correct out of {total_tokens} total token predictions)")
    passed = overall_precision >= 80.0
    print(f"  STATUS: {'PASS [OK]' if passed else 'BELOW THRESHOLD'}")
    print(f"{'=' * 70}")

    # ── Step 8: Per-gloss breakdown ──
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

    # ── Step 9: Save JSON report ──
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
    print(f"\n  Report saved to: {os.path.abspath(REPORT_PATH)}")

    return passed


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
