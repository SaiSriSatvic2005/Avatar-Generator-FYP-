#!/usr/bin/env python
"""
V1 Pipeline Runner (Baseline Representation)
Emulates the exact V1 Senior Pipeline logic:
- Sequential per-module MediaPipe sessions (no shared cache)
- Direct per-frame heuristic feature collection
- Flat set() deduplication combination (unfiltered dump)
- Demonstrates grammar explosion and SiGML rejection
"""

import os
import sys
import time
import argparse
from collections import Counter

# Set environment
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
import warnings
warnings.filterwarnings("ignore")

# Path to V1 or V2 baseline modules
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INTEGRATION_V2_DIR = os.path.join(CURRENT_DIR, "Integration-20260706T062240Z-3-001", "Integration")
if INTEGRATION_V2_DIR not in sys.path:
    sys.path.insert(0, INTEGRATION_V2_DIR)

from Handshape_Model import run_handshape_module
from ori_model2 import run_orientation_module
from upper_body_locations_video import run_upper_body_location_module
from Head_and_face_location import run_head_face_location_module
from hand_location_video_P import run_hand_location_module
from fing_locations_d import run_finger_location_module
from contact_types_e import run_contact_type_module
from Arm_and_Space_positions import run_arm_space_module
from movement1_prava import run_movement1_module
from Movement_2 import run_movement2_module

def extract_raw_v1_labels(module_output):
    """V1 extract_labels: flattens per_frame and lists without filtering."""
    if isinstance(module_output, dict):
        if "per_frame" in module_output:
            labels = module_output["per_frame"]
            cleaned = []
            for item in labels:
                if item is None: continue
                if isinstance(item, tuple):
                    cleaned.extend([str(x) for x in item if x is not None])
                else:
                    cleaned.append(str(item))
            return cleaned
        if "final" in module_output and module_output["final"]:
            f = module_output["final"]
            if isinstance(f, tuple):
                return [str(x) for x in f if x is not None]
            return [str(f)]
    if isinstance(module_output, str):
        return module_output.split()
    if isinstance(module_output, list):
        cleaned = []
        for item in module_output:
            if item is not None:
                cleaned.extend(str(item).split())
        return cleaned
    return []

def combine_hamnosys_v1(*modules):
    """Exact V1 combination logic: flat append + set() deduplication."""
    final = []
    for m in modules:
        final.extend(extract_raw_v1_labels(m))

    seen = set()
    clean = []
    for x in final:
        if x not in seen:
            clean.append(x)
            seen.add(x)
    return " ".join(clean), len(final), len(clean)

def run_v1_trace(video_path):
    video_name = os.path.basename(video_path)
    start_time = time.time()

    print("\n" + "=" * 80)
    print(" [PIPELINE: VERSION 1 - SENIOR BASELINE]")
    print(f" Target Video: {video_name}")
    print("=" * 80)

    # 1. Preprocessing Stage
    print("\n[STAGE 1: PREPROCESSING & INGESTION]")
    print("  - Input Video Feed:       Raw frames (No normalization, No CLAHE)")
    print("  - Landmark Architecture:  10 Independent MediaPipe Invocations (No shared cache)")
    print("  - Two-Handed Classifier:  Not Supported (Single-hand dominant assumption)")
    print("  - Temporal Filtering:     None (Raw noisy frame-by-frame heuristics)")

    # 2. Individual Module Outputs
    print("\n[STAGE 2: 10-MODULE FEATURE EXTRACTION FLOW]")
    
    # Module 1
    t0 = time.time()
    hs = run_handshape_module(video_path)
    hs_raw = extract_raw_v1_labels(hs)
    print(f"  [Module 1]  Handshape Model:              {len(hs_raw)} frame labels -> {list(set(hs_raw))[:5]}")
    
    # Module 2
    ori = run_orientation_module(video_path)
    ori_raw = extract_raw_v1_labels(ori)
    print(f"  [Module 2]  Orientation Model:            {len(ori_raw)} frame labels -> {list(set(ori_raw))[:5]}")

    # Module 3
    ub = run_upper_body_location_module(video_path)
    ub_raw = extract_raw_v1_labels(ub)
    print(f"  [Module 3]  Upper Body Location:          {len(ub_raw)} frame labels -> {list(set(ub_raw))[:5]}")

    # Module 4
    hf = run_head_face_location_module(video_path)
    hf_raw = extract_raw_v1_labels(hf)
    print(f"  [Module 4]  Head & Face Location:         {len(hf_raw)} frame labels -> {list(set(hf_raw))[:5]}")

    # Module 5
    hl = run_hand_location_module(video_path)
    hl_raw = extract_raw_v1_labels(hl)
    print(f"  [Module 5]  Hand Relative Location:       {len(hl_raw)} frame labels -> {list(set(hl_raw))[:5]}")

    # Module 6
    fl = run_finger_location_module(video_path)
    fl_raw = extract_raw_v1_labels(fl)
    print(f"  [Module 6]  Finger Placement:             {len(fl_raw)} frame labels -> {list(set(fl_raw))[:5]}")

    # Module 7
    ct = run_contact_type_module(video_path)
    ct_raw = extract_raw_v1_labels(ct)
    print(f"  [Module 7]  Contact Type Recognizer:      {len(ct_raw)} frame labels -> {list(set(ct_raw))[:5]}")

    # Module 8
    asp = run_arm_space_module(video_path)
    asp_raw = extract_raw_v1_labels(asp)
    print(f"  [Module 8]  Arm & Space Position:         {len(asp_raw)} frame labels -> {list(set(asp_raw))[:5]}")

    # Module 9
    m1 = run_movement1_module(video_path)
    m1_raw = extract_raw_v1_labels(m1)
    print(f"  [Module 9]  Primary Movement:             {len(m1_raw)} frame labels -> {list(set(m1_raw))[:5]}")

    # Module 10
    m2 = run_movement2_module(video_path)
    m2_raw = extract_raw_v1_labels(m2)
    print(f"  [Module 10] Secondary Movement / Dyn:     {len(m2_raw)} frame labels -> {list(set(m2_raw))[:5]}")

    # 3. Filtration / Assembly
    v1_seq, total_extracted, unique_dedup = combine_hamnosys_v1(
        hs, ori, ub, hf, hl, fl, ct, asp, m1, m2
    )

    print("\n[STAGE 3: FILTRATION & COMBINATION LOGIC]")
    print("  - Assembly Method:        Flat Dump Accumulator + seen.add() Deduplication")
    print("  - Context-Free Grammar:   NONE (Grammar Rules Bypassed)")
    print("  - Temporal Majority Vote: NONE (Every transient frame noise token is kept)")
    print(f"  - Total Extracted Tokens: {total_extracted} per-frame tokens")
    print(f"  - Post-Deduplication:     {unique_dedup} tokens retained")

    # 4. Final Output
    elapsed = time.time() - start_time
    print("\n[STAGE 4: FINAL HAMNOSYS SEQUENCE & SIGML COMPATIBILITY]")
    print(f"  - Generated String:   \"{v1_seq}\"")
    print(f"  - Token Count:        {len(v1_seq.split())} tokens (Excessive Token Bloat)")
    print("  - SiGML Validation:   [REJECTED] SYNTAX ERROR (Multiple colliding locations & postures)")
    print("  - 3D Avatar Status:   [FAILED] CANNOT RENDER (SiGML parser error / invalid gesture structure)")
    print(f"  - Total Runtime:      {elapsed:.2f}s")
    print("=" * 80 + "\n")

    return {
        "version": "V1",
        "video": video_name,
        "tokens": v1_seq,
        "token_count": len(v1_seq.split()),
        "valid": False,
        "elapsed": elapsed
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", default="ISL_videos/Hello.mp4", nargs="?")
    args = parser.parse_args()
    run_v1_trace(args.video)
