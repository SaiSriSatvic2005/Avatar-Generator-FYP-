#!/usr/bin/env python3
"""
Upgraded Benchmark on 10 DGS Videos using Local HaMeR 3D Hybrid Pipeline
Evaluates Against Clean Ground Truth:
  - Dual-Hand Symmetry (hamsymmlr vs single)
  - 3D Handshape (Model A at Kinetic Stroke Apex)
  - 3D Extended Finger Direction (Metric Longitudinal Axis)
  - 3D Palm Orientation (True Cross-Product Normal on Sphere)
  - Anatomical Location (Body Voronoi Engine)
  - 3D Movement Trajectory (Model B + Path Length Amplitude Gating)
  - CWASA SiGML XML Generation
"""

import os
import sys
import pandas as pd
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from local_hamer_pipeline import LocalHaMeRPipeline

def main():
    csv_path = os.path.join(ROOT_DIR, "downloaded_dgs_dataset", "dgs_clean_ground_truth.csv")
    video_dir = os.path.join(ROOT_DIR, "downloaded_dgs_dataset", "videos")

    if not os.path.exists(csv_path):
        print(f"Error: CSV not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    sample_df = df.head(10)

    print("=================================================================")
    print("      INITIALIZING UPGRADED LOCAL HAMER 3D BENCHMARK ENGINE      ")
    print("=================================================================")
    pipeline = LocalHaMeRPipeline()
    print("[Benchmark] Upgraded HaMeR 3D Pipeline Ready. Running 10 Videos...\n")

    results = []
    symm_correct = 0
    hs_correct = 0
    ext_correct = 0
    palm_correct = 0
    loc_correct = 0
    mv_correct = 0

    for idx, row in sample_df.iterrows():
        vname = row['video_file']
        vpath = os.path.join(video_dir, vname)
        stem = row['stem']

        gt_symm = "hamsymmlr" if row['is_two_handed'] else "single"
        gt_hs = str(row['base_handshape']).strip()
        gt_ext = str(row['extended_finger']).strip()
        gt_palm = str(row['palm_orientation']).strip()
        gt_loc = str(row['location']).strip()
        gt_mv = str(row['movement']).strip()

        if not os.path.exists(vpath):
            print(f"[{idx+1}/10] Video missing: {vpath}")
            continue

        print(f"[{idx+1}/10] Processing {vname} ({stem})...")

        try:
            res = pipeline.process_video(vpath)
            pred_symm = res["symmetry"]
            pred_hs = res["handshape"]
            pred_ext = res["extended_finger"]
            pred_palm = res["palm_orientation"]
            pred_loc = res["location"]
            pred_mv = res["movement"]
            hns_str = res["hamnosys"]
            sigml_xml = res["sigml_xml"]

            symm_match = (pred_symm == gt_symm)
            hs_match = (pred_hs == gt_hs)
            ext_match = (pred_ext == gt_ext) or (gt_ext in pred_ext) or (pred_ext in gt_ext)
            palm_match = (pred_palm == gt_palm) or (gt_palm in pred_palm) or (pred_palm in gt_palm)
            loc_match = (pred_loc == gt_loc)
            mv_match = (pred_mv == gt_mv)

            if symm_match: symm_correct += 1
            if hs_match: hs_correct += 1
            if ext_match: ext_correct += 1
            if palm_match: palm_correct += 1
            if loc_match: loc_correct += 1
            if mv_match: mv_correct += 1

            sigml_valid = ("<sigml>" in sigml_xml and "</sigml>" in sigml_xml)

            print(f"   Symm: GT={gt_symm:<12} Pred={pred_symm:<12} {'[MATCH]' if symm_match else '[DIFF]'}")
            print(f"   HS:   GT={gt_hs:<12} Pred={pred_hs:<12} {'[MATCH]' if hs_match else '[DIFF]'}")
            print(f"   Ext:  GT={gt_ext:<12} Pred={pred_ext:<12} {'[MATCH]' if ext_match else '[DIFF]'}")
            print(f"   Palm: GT={gt_palm:<12} Pred={pred_palm:<12} {'[MATCH]' if palm_match else '[DIFF]'}")
            print(f"   Loc:  GT={gt_loc:<12} Pred={pred_loc:<12} {'[MATCH]' if loc_match else '[DIFF]'}")
            print(f"   Mv:   GT={gt_mv:<12} Pred={pred_mv:<12} {'[MATCH]' if mv_match else '[DIFF]'}")
            print(f"   HNS:  {hns_str}")
            print(f"   SiGML Valid: {sigml_valid}\n")

            results.append({
                "video": vname,
                "stem": stem,
                "gt_symm": gt_symm, "pred_symm": pred_symm, "symm_match": symm_match,
                "gt_hs": gt_hs, "pred_hs": pred_hs, "hs_match": hs_match,
                "gt_ext": gt_ext, "pred_ext": pred_ext, "ext_match": ext_match,
                "gt_palm": gt_palm, "pred_palm": pred_palm, "palm_match": palm_match,
                "gt_loc": gt_loc, "pred_loc": pred_loc, "loc_match": loc_match,
                "gt_mv": gt_mv, "pred_mv": pred_mv, "mv_match": mv_match,
                "sigml_valid": sigml_valid
            })

        except Exception as e:
            print(f"   Error processing {vname}: {e}")
            import traceback
            traceback.print_exc()

    n = len(results)
    if n > 0:
        print("=================================================================")
        print("            UPGRADED HAMER 3D BENCHMARK ACCURACY (N=10)          ")
        print("=================================================================")
        print(f"Dual-Hand Symmetry Acc:    {symm_correct}/{n} ({symm_correct/n*100:.1f}%)")
        print(f"Handshape (Apex) Acc:      {hs_correct}/{n} ({hs_correct/n*100:.1f}%)")
        print(f"Extended Finger Acc:       {ext_correct}/{n} ({ext_correct/n*100:.1f}%)")
        print(f"Palm Orientation Acc:      {palm_correct}/{n} ({palm_correct/n*100:.1f}%)")
        print(f"Anatomical Location Acc:   {loc_correct}/{n} ({loc_correct/n*100:.1f}%)")
        print(f"Movement Accuracy:         {mv_correct}/{n} ({mv_correct/n*100:.1f}%)")
        total_slots = n * 6
        total_correct = symm_correct + hs_correct + ext_correct + palm_correct + loc_correct + mv_correct
        print(f"Overall Composite Slot Acc:{total_correct}/{total_slots} ({total_correct/total_slots*100:.1f}%)")
        print("=================================================================")

if __name__ == "__main__":
    main()
