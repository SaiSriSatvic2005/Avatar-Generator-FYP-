#!/usr/bin/env python3
"""
Phase 2: DWPose Feature Extraction & Stroke Segmentation
Extracts 133 Whole-Body keypoints using DWPose (RTMPose) with DirectML GPU acceleration.
Applies kinetic stroke activity windowing and hand/body-centric normalization.

Reference: HYBRID_NN_ARCHITECTURE_AND_TRAINING_SPECIFICATION.md - Step 2, 3, 4
"""

import os
import sys
import cv2
import json
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

# Setup DirectML hardware acceleration for ONNX Runtime
try:
    import rtmlib.tools.base as base
    base.RTMLIB_SETTINGS['onnxruntime']['dml'] = 'DmlExecutionProvider'
    base.RTMLIB_SETTINGS['onnxruntime']['cuda'] = 'DmlExecutionProvider'
    from rtmlib import Wholebody
    HAS_RTM = True
except Exception as e:
    print(f"[Warning] rtmlib import error: {e}")
    HAS_RTM = False

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))

VIDEOS_DIR = os.path.join(_REPO_ROOT, "ISL_videos")
LABELS_CSV = os.path.join(_SCRIPT_DIR, "isl_ground_truth_labels.csv")
OUTPUT_NPZ = os.path.join(_SCRIPT_DIR, "isl_dwpose_dataset.npz")
ENCODERS_JSON = os.path.join(_SCRIPT_DIR, "isl_label_encoders.json")

# =====================================================================
# Stage 2: Kinetic Stroke Activity Windowing
# =====================================================================

def detect_sign_stroke_window(wrist_positions, hip_y_positions, fps=30.0, energy_thresh=0.0003):
    """
    Isolates active signing frames and trims preparation/retraction motion
    using hand kinetic energy:
      v_t = ||p_wrist(t) - p_wrist(t-1)||_2 / delta_t
      E_t = v_t^2
    Smooths with a 5-tap Gaussian kernel.
    """
    n_frames = len(wrist_positions)
    if n_frames < 6:
        return 0, max(0, n_frames - 1)
        
    delta_t = 1.0 / max(fps, 1.0)
    velocities = np.linalg.norm(np.diff(wrist_positions, axis=0), axis=1) / delta_t
    energies = velocities ** 2
    
    # 5-tap Gaussian kernel smoothing (approx sigma=1.0)
    smoothed_e = gaussian_filter1d(energies, sigma=1.0, mode='nearest')
    
    # Hand elevated above hip condition (y is lower in pixel coordinates when higher up)
    elevated = [wrist_positions[i][1] < (hip_y_positions[i] + 0.05) for i in range(len(smoothed_e))]
    
    active_indices = [i for i, (e, elev) in enumerate(zip(smoothed_e, elevated)) if e > energy_thresh and elev]
    
    if len(active_indices) >= 5:
        start_idx = max(0, active_indices[0])
        end_idx = min(n_frames - 1, active_indices[-1] + 1)
        return start_idx, end_idx
    else:
        # Fallback: frames where hand is elevated, or core 70% of video
        elev_indices = [i for i, elev in enumerate(elevated) if elev]
        if len(elev_indices) >= 5:
            return max(0, elev_indices[0]), min(n_frames - 1, elev_indices[-1] + 1)
        t_start = int(0.15 * n_frames)
        t_end = int(0.85 * n_frames)
        return t_start, max(t_start + 1, t_end)

# =====================================================================
# Normalization Utilities
# =====================================================================

def normalize_hand_coords(hand_21):
    """
    Hand-Centric Normalization (Section 6, Step 4):
    Translates wrist (pt 0) to origin and scales by palm length (||pt 9 - pt 0||).
    Yields scale- and position-invariant 63D vector.
    """
    wrist = hand_21[0]
    palm_len = np.linalg.norm(hand_21[9] - wrist)
    if palm_len < 1e-5:
        palm_len = 1.0
        
    norm_hand = (hand_21 - wrist) / palm_len # (21, 2)
    # Augment to (21, 3) with relative distances from wrist
    dists = np.linalg.norm(norm_hand, axis=1, keepdims=True)
    coords_63 = np.hstack([norm_hand, dists]).flatten() # (63,)
    return coords_63

def resample_trajectory(coords_list, target_len=30):
    """
    Resample trajectory coordinates to target_len timesteps using linear interpolation.
    """
    arr = np.array(coords_list) # (T, 3)
    t_curr = np.linspace(0, 1, len(arr))
    t_new = np.linspace(0, 1, target_len)
    f = interp1d(t_curr, arr, axis=0, kind='linear', fill_value='extrapolate')
    return f(t_new).T # (3, target_len)

# =====================================================================
# Main Feature Extraction Pipeline
# =====================================================================

def extract_features(max_videos=None, frame_subsample=2):
    print("=" * 60)
    print("  Phase 2: DWPose Whole-Body Landmark Extraction")
    print(f"  DirectML Acceleration: Active on RTX 3050 GPU")
    print("=" * 60)
    
    if not os.path.exists(LABELS_CSV):
        raise FileNotFoundError(f"Missing {LABELS_CSV}. Run parse_isl_sigml.py first.")
        
    df = pd.read_csv(LABELS_CSV)
    print(f"Loaded {len(df)} ground truth video entries from {LABELS_CSV}")
    
    # Filter valid handshape and movement entries
    df = df.dropna(subset=['handshape', 'movement']).reset_index(drop=True)
    if max_videos and max_videos < len(df):
        print(f"Selecting {max_videos} balanced representative videos for training...")
        per_class = max(1, max_videos // df['handshape'].nunique())
        sampled_dfs = []
        for hs, group in df.groupby('handshape'):
            sampled_dfs.append(group.head(per_class))
        df = pd.concat(sampled_dfs).head(max_videos).reset_index(drop=True)
        
    # Unique classes
    handshape_classes = sorted(df['handshape'].unique().tolist())
    movement_classes = sorted(df['movement'].unique().tolist())
    
    hs2idx = {c: i for i, c in enumerate(handshape_classes)}
    mv2idx = {c: i for i, c in enumerate(movement_classes)}
    
    with open(ENCODERS_JSON, "w") as f:
        json.dump({
            "handshape_classes": handshape_classes,
            "movement_classes": movement_classes,
            "handshape_to_idx": hs2idx,
            "movement_to_idx": mv2idx
        }, f, indent=2)
    print(f"Saved class mappings ({len(handshape_classes)} handshapes, {len(movement_classes)} movements) to {ENCODERS_JSON}")
    
    # Initialize Wholebody with DirectML
    wholebody = Wholebody(to_openpose=False, backend='onnxruntime', device='dml')
    
    all_handshape_feats = []
    all_handshape_labels = []
    all_movement_trajs = []
    all_movement_labels = []
    video_names = []
    
    successful_vids = 0
    total_to_process = len(df)
    
    for idx, row in df.iterrows():
        vid_file = row['video_file']
        vid_path = os.path.join(VIDEOS_DIR, vid_file)
        if not os.path.exists(vid_path):
            continue
            
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            continue
            
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames_kpts = []
        frames_scores = []
        
        frame_idx = 0
        cached_bbox = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
                
            # Process with frame subsampling for efficiency
            if frame_idx % frame_subsample == 0:
                h, w = frame.shape[:2]
                
                # Bounding box caching: Detect person once every 30 frames, run pose-only on rest
                if cached_bbox is None or frame_idx % 30 == 0:
                    try:
                        cached_bbox = wholebody.det_model(frame)
                    except Exception:
                        cached_bbox = None
                        
                if cached_bbox is not None and len(cached_bbox) > 0:
                    kpts, scores = wholebody.pose_model(frame, bboxes=cached_bbox)
                else:
                    kpts, scores = wholebody(frame)
                    
                if len(kpts.shape) == 3:
                    if len(kpts) == 0:
                        continue
                    kpts, scores = kpts[0], scores[0]
                elif len(kpts) == 0:
                    continue
                    
                # Normalize keypoints to [0, 1]
                kpts[:, 0] /= float(w)
                kpts[:, 1] /= float(h)
                
                frames_kpts.append(kpts)
                frames_scores.append(scores)
                
            frame_idx += 1
            
        cap.release()
        
        if len(frames_kpts) < 5:
            continue
            
        # Parse tracking joints
        wrist_pts = []
        hip_y_pts = []
        hand_pts_list = []
        chest_pts = []
        shoulder_dist_list = []
        
        for kpts, scores in zip(frames_kpts, frames_scores):
            # Body: 0:17
            # Left Hand: 91:112, Right Hand: 112:133
            l_score = np.mean(scores[91:112]) if len(scores) >= 133 else 0.0
            r_score = np.mean(scores[112:133]) if len(scores) >= 133 else 0.0
            
            # Use right hand by default or left if right confidence low
            if r_score >= l_score and r_score > 0.2:
                hand_21 = kpts[112:133]
                wrist = hand_21[0]
            elif l_score > 0.2:
                hand_21 = kpts[91:112]
                wrist = hand_21[0]
            else:
                hand_21 = kpts[112:133]
                wrist = kpts[10] if len(kpts) > 10 else hand_21[0] # Body right wrist
                
            wrist_pts.append(wrist)
            hand_pts_list.append(hand_21)
            
            # Mid-hip y
            hip_y = (kpts[11, 1] + kpts[12, 1]) / 2.0 if len(kpts) > 12 else 0.75
            hip_y_pts.append(hip_y)
            
            # Chest center & shoulder distance
            if len(kpts) > 6:
                chest = (kpts[5] + kpts[6]) / 2.0
                s_dist = np.linalg.norm(kpts[5] - kpts[6])
            else:
                chest = np.array([0.5, 0.4])
                s_dist = 0.3
            chest_pts.append(chest)
            shoulder_dist_list.append(max(s_dist, 1e-4))
            
        wrist_pts = np.array(wrist_pts)
        hip_y_pts = np.array(hip_y_pts)
        
        # Detect active stroke window
        start_t, end_t = detect_sign_stroke_window(wrist_pts, hip_y_pts, fps=fps/frame_subsample)
        stroke_hand_pts = hand_pts_list[start_t:end_t]
        
        if len(stroke_hand_pts) == 0:
            stroke_hand_pts = hand_pts_list
            start_t, end_t = 0, len(hand_pts_list)
            
        # 1. Model A: Frame-level handshape samples (15x multiplier)
        hs_label = hs2idx[row['handshape']]
        for h_21 in stroke_hand_pts:
            feat_63 = normalize_hand_coords(h_21)
            all_handshape_feats.append(feat_63)
            all_handshape_labels.append(hs_label)
            
        # 2. Model B: Body-centric wrist velocity trajectory (3, 30)
        mv_label = mv2idx[row['movement']]
        norm_traj_pts = []
        for i in range(start_t, end_t):
            norm_wrist = (wrist_pts[i] - chest_pts[i]) / shoulder_dist_list[i]
            # Form (x, y, z) where z is relative depth from shoulder distance
            norm_traj_pts.append(np.array([norm_wrist[0], norm_wrist[1], 0.0]))
            
        if len(norm_traj_pts) >= 2:
            velocities = np.diff(norm_traj_pts, axis=0)
            if len(velocities) < 2:
                velocities = np.tile(velocities, (2, 1))
            traj_30 = resample_trajectory(velocities, target_len=30) # (3, 30)
        else:
            traj_30 = np.zeros((3, 30), dtype=np.float32)
            
        all_movement_trajs.append(traj_30)
        all_movement_labels.append(mv_label)
        video_names.append(row.get('gloss', row['video_file']))
        
        successful_vids += 1
        if successful_vids % 25 == 0 or successful_vids == total_to_process:
            print(f"  Processed [{successful_vids}/{total_to_process}] videos | Total Handshape Frames: {len(all_handshape_feats)}")
            np.savez_compressed(
                OUTPUT_NPZ,
                handshape_features=np.array(all_handshape_feats, dtype=np.float32),
                handshape_labels=np.array(all_handshape_labels, dtype=np.int64),
                movement_trajectories=np.array(all_movement_trajs, dtype=np.float32),
                movement_labels=np.array(all_movement_labels, dtype=np.int64),
                video_names=np.array(video_names)
            )
            
    # Save final dataset to .npz
    np.savez_compressed(
        OUTPUT_NPZ,
        handshape_features=np.array(all_handshape_feats, dtype=np.float32),
        handshape_labels=np.array(all_handshape_labels, dtype=np.int64),
        movement_trajectories=np.array(all_movement_trajs, dtype=np.float32),
        movement_labels=np.array(all_movement_labels, dtype=np.int64),
        video_names=np.array(video_names)
    )
    
    print("-" * 60)
    print(f"[DONE] Successfully extracted ISL DWPose Dataset:")
    print(f"  Handshape Samples: {len(all_handshape_feats)} frame vectors (dim=63)")
    print(f"  Movement Trajectories: {len(all_movement_trajs)} sequences (shape=(3, 30))")
    print(f"  Saved to: {OUTPUT_NPZ}")
    print("=" * 60)

if __name__ == "__main__":
    # If passed an argument for max videos, use it; otherwise extract full dataset
    max_v = int(sys.argv[1]) if len(sys.argv) > 1 else None
    extract_features(max_videos=max_v, frame_subsample=2)
