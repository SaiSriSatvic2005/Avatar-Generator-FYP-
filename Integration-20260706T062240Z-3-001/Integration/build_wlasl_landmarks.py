#!/usr/bin/env python3
"""
WLASL Video Dataset Landmark Extractor & Normalizer
Processes 11,980 WLASL MP4 videos into normalized keypoint tensors (T=150, 225)
paired with auto-generated HamNoSys component labels from gloss_to_hamnosys_dict.json.
"""

import os
import sys
import json
import csv
import glob
import numpy as np
import cv2

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from shared_landmarks import get_video_landmarks

def extract_normalized_features(frame_info):
    """
    Construct normalized feature vector for a single video frame.
    Features:
    - [0:63]   = Right Hand 21 3D Landmarks (x, y, z)
    - [63:126] = Left Hand 21 3D Landmarks (x, y, z)
    - [126:177] = Body Pose 17 3D Landmarks (x, y, z)  [COCO format for RTMPose]
    Total = 177 dimensions (was 225 with MediaPipe's 33-pt pose)
    Normalized by Chest Origin (0,0,0) and Shoulder Distance = 1.0.
    """
    pose = frame_info.get("pose_world_landmarks") or frame_info.get("pose_landmarks")
    r_hand = frame_info.get("right_hand")
    l_hand = frame_info.get("left_hand")

    # 1. Determine Chest Origin and Scale Normalizer
    origin = np.array([0.5, 0.5, 0.0])
    scale = 1.0
    
    if pose and hasattr(pose, "landmark"):
        n_pts = len(pose.landmark)
        # RTMPose COCO 17-point: shoulders at index 5 (left), 6 (right)
        # MediaPipe 33-point: shoulders at index 11 (left), 12 (right)
        if n_pts >= 17 and n_pts < 33:
            l_sh_idx, r_sh_idx = 5, 6  # COCO format
        elif n_pts >= 33:
            l_sh_idx, r_sh_idx = 11, 12  # MediaPipe format
        else:
            l_sh_idx, r_sh_idx = None, None
            
        if l_sh_idx is not None:
            l_sh = np.array([pose.landmark[l_sh_idx].x, pose.landmark[l_sh_idx].y, pose.landmark[l_sh_idx].z])
            r_sh = np.array([pose.landmark[r_sh_idx].x, pose.landmark[r_sh_idx].y, pose.landmark[r_sh_idx].z])
            origin = (l_sh + r_sh) / 2.0
            scale = np.linalg.norm(l_sh - r_sh)
            if scale < 1e-4:
                scale = 1.0

    # Helper function to normalize landmark list
    def norm_lm(lm_obj, count):
        vec = np.zeros(count * 3, dtype=np.float32)
        if lm_obj and hasattr(lm_obj, "landmark"):
            for i, p in enumerate(lm_obj.landmark[:count]):
                pt = np.array([p.x, p.y, p.z])
                pt_norm = (pt - origin) / scale
                vec[i*3 : i*3+3] = pt_norm
        return vec

    r_vec = norm_lm(r_hand, 21)   # 63 dims
    l_vec = norm_lm(l_hand, 21)   # 63 dims
    p_vec = norm_lm(pose, 17)     # 51 dims (COCO 17 body keypoints)

    return np.concatenate([r_vec, l_vec, p_vec])  # (177,)


def build_wlasl_dataset(
    dict_path=os.path.join(_SCRIPT_DIR, "gloss_to_hamnosys_dict.json"),
    wlasl_json_path=os.path.join(_SCRIPT_DIR, "WLASL_videos", "archive", "WLASL_v0.3.json"),
    video_dir=os.path.join(_SCRIPT_DIR, "WLASL_videos", "archive", "videos"),
    output_dir=os.path.join(_SCRIPT_DIR, "wlasl_landmark_dataset"),
    max_samples=500,
    target_frames=150
):
    os.makedirs(output_dir, exist_ok=True)

    with open(dict_path, "r", encoding="utf-8") as f:
        gloss_dict = json.load(f)

    with open(wlasl_json_path, "r", encoding="utf-8") as f:
        wlasl_data = json.load(f)

    print(f"[DatasetBuilder] Loaded {len(gloss_dict)} gloss targets from dictionary.")
    print(f"[DatasetBuilder] Scanning WLASL videos in: {video_dir}")

    tensors = []
    metadata = []

    sample_count = 0

    for entry in wlasl_data:
        gloss = entry.get("gloss")
        if gloss not in gloss_dict:
            continue

        hamnosys_target = gloss_dict[gloss]
        instances = entry.get("instances", [])

        for inst in instances:
            video_id = inst.get("video_id")
            vpath = os.path.join(video_dir, f"{video_id}.mp4")

            if not os.path.exists(vpath):
                continue

            print(f"[{sample_count+1}/{max_samples}] Processing Gloss: '{gloss}' | Video: {video_id}.mp4")
            frames = get_video_landmarks(vpath)

            if not frames:
                continue

            # Extract per-frame feature vectors
            frame_vecs = [extract_normalized_features(f) for f in frames]
            frame_matrix = np.array(frame_vecs, dtype=np.float32) # (T_curr, 225)

            # Pad or truncate to target_frames (150)
            T_curr, D = frame_matrix.shape
            if T_curr < target_frames:
                pad_width = ((0, target_frames - T_curr), (0, 0))
                padded_matrix = np.pad(frame_matrix, pad_width, mode="edge")
            else:
                padded_matrix = frame_matrix[:target_frames]

            tensors.append(padded_matrix)
            metadata.append({
                "video_id": video_id,
                "gloss": gloss,
                "handshape": hamnosys_target["handshape"],
                "ext_finger": hamnosys_target["ext_finger"],
                "palm_ori": hamnosys_target["palm_ori"],
                "location": hamnosys_target["location"],
                "movement": hamnosys_target["movement"],
                "contact": hamnosys_target["contact"],
                "two_handed": hamnosys_target["two_handed"]
            })

            sample_count += 1
            if sample_count >= max_samples:
                break

        if sample_count >= max_samples:
            break

    if not tensors:
        print("[DatasetBuilder] Error: No valid video tensors were generated.")
        return

    npz_path = os.path.join(output_dir, "dataset_landmarks.npz")
    csv_path = os.path.join(output_dir, "metadata.csv")

    np.savez_compressed(npz_path, tensors=np.array(tensors, dtype=np.float32))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "video_id", "gloss", "handshape", "ext_finger", "palm_ori",
            "location", "movement", "contact", "two_handed"
        ])
        writer.writeheader()
        writer.writerows(metadata)

    print(f"\n[DatasetBuilder Complete] Successfully processed {len(tensors)} video sequences.")
    print(f" Saved tensors to: {os.path.abspath(npz_path)}")
    print(f" Saved metadata to: {os.path.abspath(csv_path)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=300)
    args = parser.parse_args()
    build_wlasl_dataset(max_samples=args.max_samples)
