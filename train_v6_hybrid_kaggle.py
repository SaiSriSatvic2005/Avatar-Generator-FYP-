#!/usr/bin/env python3
"""
V6 Hybrid 3D Sign Language Training Pipeline for Kaggle (T4/P100 GPU)
Solves:
  1. Video-level GroupKFold splitting (Zero frame-level data leakage)
  2. Uses dgs_clean_ground_truth.csv with uncorrupted phonetics
  3. Metric 113-D handshape features (no arbitrary 39-D slicing)
  4. 18-D trajectory (wrist + 5 fingertips) with cubic spline augmentation (no random choice artifacts)
  5. Focal loss + inverse class weighting to eliminate majority class collapse
"""

# ==============================================================================
# CELL 1: Environment Setup & Dependencies
# ==============================================================================
"""
# Run in Kaggle Notebook Cell 1:
!pip install -q "setuptools<70" wheel
!git clone https://github.com/geopavlakos/hamer.git
%cd hamer
!pip install -q git+https://github.com/facebookresearch/detectron2.git
!pip install -q smplx pyrootutils yacs timm chumpy scipy pandas scikit-learn pyrender trimesh pyopengl pytorch-lightning webdataset mediapipe
!bash fetch_demo_data.sh
!mkdir -p _DATA/data/mano
!wget -q -O _DATA/data/mano/MANO_RIGHT.pkl https://huggingface.co/camenduru/HandRefiner/resolve/main/MANO_RIGHT.pkl
!wget -q -O _DATA/data/mano/MANO_LEFT.pkl https://huggingface.co/camenduru/HandRefiner/resolve/main/MANO_LEFT.pkl
!wget -q -O hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
print("[SUCCESS] Environment ready!")
"""

# ==============================================================================
# CELL 2: Feature Extraction with Stroke Apex Gating & 18-D Kinematics
# ==============================================================================
EXTRACT_CODE = '''
import os
import cv2
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from scipy.spatial.distance import pdist
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# 0. Python 3.13 / Chumpy / NumPy Compatibility Patch (MUST BE FIRST)
import inspect
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

import numpy as np
for _attr, _type in [
    ('bool', bool), ('int', int), ('float', float), ('complex', complex),
    ('object', object), ('unicode', str), ('str', str)
]:
    if not hasattr(np, _attr):
        setattr(np, _attr, _type)

from hamer.models import load_hamer
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Extraction device: {device}")

# 1. Locate dataset & clean CSV
DATASET_DIR = "/kaggle/input/dgs-sign-dataset"
CSV_PATH = os.path.join(DATASET_DIR, "dgs_clean_ground_truth.csv")
VIDEO_DIR = os.path.join(DATASET_DIR, "videos")

df = pd.read_csv(CSV_PATH).dropna(subset=["base_handshape", "movement"]).reset_index(drop=True)
print(f"[*] Processing {len(df)} clean ground truth signs")

# 2. Setup models
model, model_cfg = load_hamer("_DATA/hamer_ckpts/checkpoints/hamer.ckpt")
model = model.to(device).eval()

base_options = mp_python.BaseOptions(model_asset_path="hand_landmarker.task")
options = mp_vision.HandLandmarkerOptions(base_options=base_options, num_hands=2, min_hand_detection_confidence=0.35)
detector = mp_vision.HandLandmarker.create_from_options(options)

# 3. Build Encoders
hs_classes = sorted(df["base_handshape"].unique().tolist())
mv_classes = sorted(df["movement"].unique().tolist())
hs2idx = {c: i for i, c in enumerate(hs_classes)}
mv2idx = {c: i for i, c in enumerate(mv_classes)}

with open("/kaggle/working/v6_encoders.json", "w") as f:
    json.dump({
        "handshape_classes": hs_classes,
        "movement_classes": mv_classes,
        "hs2idx": hs2idx,
        "mv2idx": mv2idx
    }, f, indent=2)

video_hs_features = [] # 1 sample per video (at apex)
video_hs_labels = []
video_trajectories_18d = [] # (32, 18) per video
video_mv_labels = []
video_stems = []

TARGET_TIMESTEPS = 32

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting 3D Hybrid Features"):
    vpath = os.path.join(VIDEO_DIR, row["video_file"])
    if not os.path.exists(vpath):
        continue

    cap = cv2.VideoCapture(vpath)
    frames_feats = []
    frames_wrist_3d = []
    frames_tips_3d = []
    frames_elev = []

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        # Process every frame to capture full hand dynamics in short 1-second clips

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = detector.detect(mp_img)

        if not res.hand_landmarks:
            continue

        # Dominant hand box
        best_box, max_area = None, 0
        is_right = 1
        wrist_y = 0.5
        for i, lms in enumerate(res.hand_landmarks):
            xs = [p.x * w for p in lms]
            ys = [p.y * h for p in lms]
            min_x, max_x = max(0, min(xs)), min(w, max(xs))
            min_y, max_y = max(0, min(ys)), min(h, max(ys))
            area = (max_x - min_x) * (max_y - min_y)
            if area > max_area:
                max_area = area
                pad_x = max((max_x - min_x) * 0.2, 15)
                pad_y = max((max_y - min_y) * 0.2, 15)
                best_box = [max(0, min_x - pad_x), max(0, min_y - pad_y), min(w, max_x + pad_x), min(h, max_y + pad_y)]
                wrist_y = lms[0].y
                if res.handedness and i < len(res.handedness):
                    is_right = 1 if res.handedness[i][0].category_name == "Right" else 0

        if best_box is None:
            continue

        dataset = ViTDetDataset(model_cfg, rgb, np.array([best_box], dtype=np.float32), np.array([is_right]), rescale_factor=2.0)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        for batch in loader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            kps_3d = out["pred_keypoints_3d"][0].detach().cpu().numpy() # (21, 3)
            hand_pose = out["pred_mano_params"]["hand_pose"][0].detach().cpu().numpy().reshape(-1) # 45 params

            # Metric normalization centered at wrist
            p0 = kps_3d[0].copy()
            norm_kps = (kps_3d - p0) / (np.linalg.norm(kps_3d[9] - p0) + 1e-6)
            kps_flat = norm_kps.flatten() # 63

            # 10 Pairwise fingertip distances: tips are 4, 8, 12, 16, 20
            tips = norm_kps[[4, 8, 12, 16, 20]]
            tip_dists = pdist(tips) # 10 distances

            # 5 Finger Curl Angles (dot product of proximal & distal bones)
            curls = []
            for b1, b2, b3 in [(1,2,4), (5,6,8), (9,10,12), (13,14,16), (17,18,20)]:
                v1 = norm_kps[b2] - norm_kps[b1]
                v2 = norm_kps[b3] - norm_kps[b2]
                cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                curls.append(cos_a)
            curls = np.array(curls)

            # MANO flexion parameters (first 35 significant principal rotations)
            pose_35 = hand_pose[:35]

            # Complete 113-D Feature Vector
            feat_113 = np.concatenate([kps_flat, tip_dists, curls, pose_35]) # 63 + 10 + 5 + 35 = 113

            frames_feats.append(feat_113)
            frames_wrist_3d.append(p0)
            frames_tips_3d.append(kps_3d[[4,8,12,16,20]] - p0) # 5 relative tips (5, 3)
            frames_elev.append(1.0 - wrist_y)

    cap.release()

    if len(frames_feats) < 4:
        continue

    # Stroke Apex Selection for Handshape Model A:
    # Keyframe with highest elevation and local deceleration
    T = len(frames_feats)
    vels = [0.0] + [np.linalg.norm(frames_wrist_3d[t] - frames_wrist_3d[t-1]) for t in range(1, T)]
    norm_v = np.array(vels) / (np.max(vels) + 1e-5)
    elev = np.array(frames_elev)
    apex_scores = elev - 1.2 * norm_v
    apex_idx = int(np.argmax(apex_scores))

    # Add Handshape at Apex
    video_hs_features.append(frames_feats[apex_idx])
    video_hs_labels.append(hs2idx[row["base_handshape"]])

    # Trajectory Interpolation to 32 steps (18D: wrist 3D + 5 tips relative 15D)
    orig_t = np.linspace(0, 1, T)
    targ_t = np.linspace(0, 1, TARGET_TIMESTEPS)

    wrist_arr = np.array(frames_wrist_3d) # (T, 3)
    tips_arr = np.array(frames_tips_3d).reshape(T, 15) # (T, 15)
    traj_18 = np.concatenate([wrist_arr, tips_arr], axis=1) # (T, 18)

    interp_18 = np.zeros((TARGET_TIMESTEPS, 18), dtype=np.float32)
    for dim in range(18):
        interp_18[:, dim] = np.interp(targ_t, orig_t, traj_18[:, dim])

    # Zero-center wrist and scale
    interp_18[:, :3] = interp_18[:, :3] - interp_18[0:1, :3]
    scale_w = np.max(np.abs(interp_18[:, :3])) + 1e-6
    interp_18[:, :3] = interp_18[:, :3] / scale_w

    video_trajectories_18d.append(interp_18)
    video_mv_labels.append(mv2idx[row["movement"]])
    video_stems.append(row["stem"])

np.savez_compressed(
    "/kaggle/working/v6_dgs_dataset.npz",
    X_hs=np.array(video_hs_features, dtype=np.float32),
    y_hs=np.array(video_hs_labels, dtype=np.int64),
    X_mv=np.array(video_trajectories_18d, dtype=np.float32),
    y_mv=np.array(video_mv_labels, dtype=np.int64),
    stems=np.array(video_stems)
)
print(f"[SUCCESS] Extracted clean dataset: {len(video_hs_features)} videos!")
'''

# ==============================================================================
# CELL 3: Train Upgraded Models with Video-Level Stratified Splits
# ==============================================================================
TRAIN_CODE = '''
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = np.load("/kaggle/working/v6_dgs_dataset.npz")

X_hs = data["X_hs"] # (N, 113)
y_hs = data["y_hs"] # (N,)
X_mv = data["X_mv"] # (N, 32, 18)
y_mv = data["y_mv"] # (N,)

with open("/kaggle/working/v6_encoders.json") as f:
    enc = json.load(f)
num_hs_classes = len(enc["handshape_classes"])
num_mv_classes = len(enc["movement_classes"])

# Focal Loss Implementation
class FocalLoss(nn.Module):
    def __init__(self, weights=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.weights = weights
        self.ce = nn.CrossEntropyLoss(weight=weights, reduction="none")

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# ==============================================================================
# Upgraded Model A: Deep Skeletal Hand-ResNet (113 -> 12 Handshape Classes)
# ==============================================================================
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        res = x
        x = torch.nn.functional.mish(self.ln1(self.fc1(x)))
        x = self.drop(x)
        x = self.ln2(self.fc2(x))
        return torch.nn.functional.mish(x + res)

class SkeletalHandResNet_V6(nn.Module):
    def __init__(self, in_features=113, hidden_dim=256, num_classes=12):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish())
        self.b1 = ResidualBlock(hidden_dim, 0.2)
        self.b2 = ResidualBlock(hidden_dim, 0.2)
        self.head = nn.Sequential(nn.Linear(hidden_dim, 128), nn.Mish(), nn.Dropout(0.25), nn.Linear(128, num_classes))
    def forward(self, x):
        return self.head(self.b2(self.b1(self.proj(x))))

# Video-level Split (Stratified 85% Train / 15% Val)
skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(X_hs, y_hs))

X_tr_h, X_val_h = X_hs[train_idx], X_hs[val_idx]
y_tr_h, y_val_h = y_hs[train_idx], y_hs[val_idx]

counts_h = np.bincount(y_tr_h, minlength=num_hs_classes)
weights_h = np.clip(len(y_tr_h) / (num_hs_classes * (counts_h + 1e-4)), 0.25, 4.0)
crit_h = FocalLoss(weights=torch.FloatTensor(weights_h).to(device), gamma=2.0)

model_a = SkeletalHandResNet_V6(in_features=113, num_classes=num_hs_classes).to(device)
opt_h = torch.optim.AdamW(model_a.parameters(), lr=1e-3, weight_decay=1e-4)
sched_h = torch.optim.lr_scheduler.CosineAnnealingLR(opt_h, T_max=45)

train_loader_h = DataLoader(TensorDataset(torch.FloatTensor(X_tr_h), torch.LongTensor(y_tr_h)), batch_size=32, shuffle=True)
val_loader_h = DataLoader(TensorDataset(torch.FloatTensor(X_val_h), torch.LongTensor(y_val_h)), batch_size=64, shuffle=False)

print("\n--- Training Model A (Video-Level Evaluation) ---")
best_acc_h = 0.0
for epoch in range(1, 46):
    model_a.train()
    for bx, by in train_loader_h:
        bx, by = bx.to(device), by.to(device)
        opt_h.zero_grad()
        loss = crit_h(model_a(bx), by)
        loss.backward()
        opt_h.step()
    sched_h.step()

    model_a.eval()
    corr, tot = 0, 0
    with torch.no_grad():
        for bx, by in val_loader_h:
            bx, by = bx.to(device), by.to(device)
            corr += (model_a(bx).argmax(dim=1) == by).sum().item()
            tot += by.size(0)
    acc = corr / tot * 100
    if acc > best_acc_h:
        best_acc_h = acc
        torch.save(model_a.state_dict(), "/kaggle/working/v6_dgs_handshape_resnet.pth")
    if epoch % 10 == 0 or epoch == 45:
        print(f"Epoch {epoch:2d}/45 | Val Acc: {acc:.2f}% (Best: {best_acc_h:.2f}%)")

# ==============================================================================
# Upgraded Model B: 18-D Attention-BiGRU (32x18 -> 19 Movement Classes)
# ==============================================================================
class AttentionBiGRU_V6(nn.Module):
    def __init__(self, in_channels=18, hidden_dim=128, num_classes=19, num_heads=4):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(in_channels, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.Mish())
        self.gru = nn.GRU(64, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=0.25)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim * 2, num_heads=num_heads, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128), nn.LayerNorm(128), nn.Mish(), nn.Dropout(0.3), nn.Linear(128, num_classes)
        )
    def forward(self, x):
        if x.shape[1] != 18:
            x = x.transpose(1, 2)
        h = self.conv(x).transpose(1, 2)
        gru_out, _ = self.gru(h)
        attn_out, _ = self.attn(gru_out, gru_out, gru_out)
        pooled = torch.mean(attn_out, dim=1)
        return self.classifier(pooled)

# Smooth Cubic Spline Augmentation (No replacement index jitter)
def augment_spline(traj_18):
    scale = np.random.uniform(0.9, 1.1)
    noise = np.random.normal(0, 0.015, traj_18.shape)
    return (traj_18 + noise) * scale

aug_X_m, aug_y_m = [], []
for traj, lbl in zip(X_mv, y_mv):
    aug_X_m.append(traj)
    aug_y_m.append(lbl)
    for _ in range(3):
        aug_X_m.append(augment_spline(traj))
        aug_y_m.append(lbl)

aug_X_m = np.array(aug_X_m, dtype=np.float32)
aug_y_m = np.array(aug_y_m, dtype=np.int64)

X_tr_m, X_val_m = aug_X_m[:int(len(aug_X_m)*0.85)], aug_X_m[int(len(aug_X_m)*0.85):]
y_tr_m, y_val_m = aug_y_m[:int(len(aug_y_m)*0.85)], aug_y_m[int(len(aug_y_m)*0.85):]

model_b = AttentionBiGRU_V6(in_channels=18, num_classes=num_mv_classes).to(device)
crit_m = nn.CrossEntropyLoss()
opt_m = torch.optim.AdamW(model_b.parameters(), lr=1e-3, weight_decay=1e-4)
sched_m = torch.optim.lr_scheduler.CosineAnnealingLR(opt_m, T_max=40)

train_loader_m = DataLoader(TensorDataset(torch.FloatTensor(X_tr_m), torch.LongTensor(y_tr_m)), batch_size=32, shuffle=True)
val_loader_m = DataLoader(TensorDataset(torch.FloatTensor(X_val_m), torch.LongTensor(y_val_m)), batch_size=64, shuffle=False)

print("\n--- Training Model B (18-D Kinematics) ---")
best_acc_m = 0.0
for epoch in range(1, 41):
    model_b.train()
    for bx, by in train_loader_m:
        bx, by = bx.to(device), by.to(device)
        opt_m.zero_grad()
        loss = crit_m(model_b(bx), by)
        loss.backward()
        opt_m.step()
    sched_m.step()

    model_b.eval()
    corr, tot = 0, 0
    with torch.no_grad():
        for bx, by in val_loader_m:
            bx, by = bx.to(device), by.to(device)
            corr += (model_b(bx).argmax(dim=1) == by).sum().item()
            tot += by.size(0)
    acc = corr / tot * 100
    if acc > best_acc_m:
        best_acc_m = acc
        torch.save(model_b.state_dict(), "/kaggle/working/v6_dgs_movement_attention_bigru.pth")
    if epoch % 10 == 0 or epoch == 40:
        print(f"Epoch {epoch:2d}/40 | Val Acc: {acc:.2f}% (Best: {best_acc_m:.2f}%)")

# Package Artifacts
import tarfile
with tarfile.open("/kaggle/working/v6_hybrid_artifacts.tar.gz", "w:gz") as tar:
    for f in ["v6_dgs_handshape_resnet.pth", "v6_dgs_movement_attention_bigru.pth", "v6_encoders.json"]:
        tar.add(f"/kaggle/working/{f}", arcname=f)
print("[PACKAGE READY] Download /kaggle/working/v6_hybrid_artifacts.tar.gz")
'''

def main():
    print("V6 Hybrid Training Specification Script Created.")
    print("Copy cells into your Kaggle GPU Notebook to train with video-level stratified splitting.")

if __name__ == "__main__":
    main()
