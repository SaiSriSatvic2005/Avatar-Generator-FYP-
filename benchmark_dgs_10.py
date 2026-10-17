#!/usr/bin/env python3
"""
Benchmark 10 DGS Videos against Ground Truth
Evaluates:
  - Handshape
  - Palm Orientation
  - Anatomical Location
  - Movement
  - Full HamNoSys / SiGML compilation
"""

import os
import sys
import pandas as pd
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
INTEGRATION_DIR = os.path.join(ROOT_DIR, "Integration-20260706T062240Z-3-001", "Integration")
if INTEGRATION_DIR not in sys.path:
    sys.path.insert(0, INTEGRATION_DIR)

from pipeline_v3_hybrid import get_v3_pipeline

def run_benchmark():
    csv_path = os.path.join(ROOT_DIR, "downloaded_dgs_dataset", "dgs_ground_truth_labels.csv")
    video_dir = os.path.join(ROOT_DIR, "downloaded_dgs_dataset", "videos")
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV not found at {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    sample_df = df.head(10)
    
    print("[Benchmark] Initializing V3 Hybrid Pipeline...")
    pipe = get_v3_pipeline()
    print("[Benchmark] Pipeline ready. Processing 10 videos...\n")
    
    results = []
    
    hs_correct = 0
    palm_correct = 0
    loc_correct = 0
    mv_correct = 0
    
    for idx, row in sample_df.iterrows():
        vname = row['video_file']
        vpath = os.path.join(video_dir, vname)
        gloss = row['gloss']
        
        gt_hs = str(row['handshape']).strip()
        gt_palm = str(row['palm_orientation']).strip()
        gt_loc = str(row['location']).strip()
        gt_mv = str(row['movement']).strip()
        
        if not os.path.exists(vpath):
            print(f"[{idx+1}/10] Video missing: {vpath}")
            continue
            
        print(f"[{idx+1}/10] Processing {vname} (Gloss: {gloss})...")
        
        try:
            hns_str, sigml_xml = pipe.process_video(vpath)
            
            kpts_list, scores_list, fps = pipe.extract_landmarks(vpath)
            
            r_wrist_pts = []
            l_wrist_pts = []
            hip_y_pts = []
            r_hand_pts = []
            chest_pts = []
            shoulder_dists = []
            
            for kpts, scores in zip(kpts_list, scores_list):
                r_hand = kpts[112:133]
                l_hand = kpts[91:112]
                r_wrist_pts.append(r_hand[0])
                l_wrist_pts.append(l_hand[0])
                r_hand_pts.append(r_hand)
                hip_y = (kpts[11, 1] + kpts[12, 1]) / 2.0 if len(kpts) > 12 else 0.75
                hip_y_pts.append(hip_y)
                if len(kpts) > 6:
                    chest = (kpts[5] + kpts[6]) / 2.0
                    s_dist = np.linalg.norm(kpts[5] - kpts[6])
                else:
                    chest = np.array([0.5, 0.4])
                    s_dist = 0.3
                chest_pts.append(chest)
                shoulder_dists.append(max(s_dist, 1e-4))
                
            r_wrist_pts = np.array(r_wrist_pts)
            l_wrist_pts = np.array(l_wrist_pts)
            hip_y_pts = np.array(hip_y_pts)
            
            start_t, end_t = pipe.compute_active_stroke_window(r_wrist_pts, hip_y_pts, fps=fps)
            if end_t <= start_t:
                start_t, end_t = 0, len(r_hand_pts)
                
            stroke_hands = r_hand_pts[start_t:end_t] if len(r_hand_pts[start_t:end_t]) > 0 else r_hand_pts
            stroke_r_wrists = r_wrist_pts[start_t:end_t] if len(r_wrist_pts[start_t:end_t]) > 0 else r_wrist_pts
            stroke_chests = chest_pts[start_t:end_t] if len(chest_pts[start_t:end_t]) > 0 else chest_pts
            stroke_sdists = shoulder_dists[start_t:end_t] if len(shoulder_dists[start_t:end_t]) > 0 else shoulder_dists
            
            pred_hs, _, _ = pipe.predict_handshape(stroke_hands)
            mean_hand = np.mean(stroke_hands, axis=0)
            _, pred_palm = pipe.compute_3d_orientation(mean_hand)
            mean_sdist = np.mean(stroke_sdists)
            hand_center = mean_hand[9]
            pred_loc, _ = pipe.compute_anatomical_location(hand_center, kpts_list[len(kpts_list)//2], mean_sdist)
            pred_mv, _ = pipe.predict_movement(stroke_r_wrists, stroke_chests, stroke_sdists)
            
            hs_match = (pred_hs == gt_hs)
            palm_match = (pred_palm == gt_palm) or (pred_palm in gt_palm) or (gt_palm in pred_palm)
            loc_match = (pred_loc == gt_loc)
            mv_match = (pred_mv == gt_mv)
            
            if hs_match: hs_correct += 1
            if palm_match: palm_correct += 1
            if loc_match: loc_correct += 1
            if mv_match: mv_correct += 1
            
            sigml_valid = ("<sigml>" in sigml_xml or "<hamgestural_sign" in sigml_xml)
            
            results.append({
                "video": vname,
                "gloss": gloss,
                "gt_hs": gt_hs,
                "pred_hs": pred_hs,
                "hs_match": hs_match,
                "gt_palm": gt_palm,
                "pred_palm": pred_palm,
                "palm_match": palm_match,
                "gt_loc": gt_loc,
                "pred_loc": pred_loc,
                "loc_match": loc_match,
                "gt_mv": gt_mv,
                "pred_mv": pred_mv,
                "mv_match": mv_match,
                "sigml_valid": sigml_valid,
                "hamnosys": hns_str
            })
            
            print(f"   HS:   GT={gt_hs:<14} Pred={pred_hs:<14} {'[OK]' if hs_match else '[DIFF]'}")
            print(f"   Palm: GT={gt_palm:<14} Pred={pred_palm:<14} {'[OK]' if palm_match else '[DIFF]'}")
            print(f"   Loc:  GT={gt_loc:<14} Pred={pred_loc:<14} {'[OK]' if loc_match else '[DIFF]'}")
            print(f"   Mv:   GT={gt_mv:<14} Pred={pred_mv:<14} {'[OK]' if mv_match else '[DIFF]'}")
            print(f"   HNS:  {hns_str}")
            print(f"   SiGML Valid: {sigml_valid}\n")
            
        except Exception as e:
            print(f"   Error processing {vname}: {e}")
            import traceback
            traceback.print_exc()
            
    n = len(results)
    if n > 0:
        print("=================================================================")
        print("                 BENCHMARK ACCURACY SUMMARY (N=10)                ")
        print("=================================================================")
        print(f"Handshape Accuracy:        {hs_correct}/{n} ({hs_correct/n*100:.1f}%)")
        print(f"Palm Orientation Accuracy: {palm_correct}/{n} ({palm_correct/n*100:.1f}%)")
        print(f"Anatomical Location Acc:   {loc_correct}/{n} ({loc_correct/n*100:.1f}%)")
        print(f"Movement Accuracy:         {mv_correct}/{n} ({mv_correct/n*100:.1f}%)")
        total_slots = n * 4
        total_correct = hs_correct + palm_correct + loc_correct + mv_correct
        print(f"Overall Slot Accuracy:     {total_correct}/{total_slots} ({total_correct/total_slots*100:.1f}%)")
        print("=================================================================")

if __name__ == "__main__":
    run_benchmark()
