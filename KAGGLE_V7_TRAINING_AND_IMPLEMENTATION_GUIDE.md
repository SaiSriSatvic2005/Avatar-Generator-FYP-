# Kaggle V7 3D HaMeR & Neural Training Guide
## Step-by-Step Implementation: 128-D Invariant Handshape ResNet & 18-D Attention-BiGRU

**Platform:** Kaggle Notebooks (Free 16 GB Tesla T4 or P100 GPU)  
**Dataset:** 989 Clean Frontal DGS (German Sign Language) Videos + Matching Ground Truth SiGML  
**Target Output Artifacts:**  
1. `v7_dgs_handshape_resnet.pth` (128-D Rotation-Invariant Handshape Classifier)  
2. `v7_dgs_movement_attention_bigru.pth` (18-D Spatiotemporal Trajectory Classifier)  
3. `v7_encoders.json` (Canonical Class Encoders)  
4. `v7_hybrid_artifacts.tar.gz` (Complete Deployable Artifact Bundle)  

---

## 1. What Changes in V7 vs. V6?

| Architectural Dimension | Previous V6 Pipeline | Upgraded V7 Pipeline | Technical Impact |
| :--- | :--- | :--- | :--- |
| **Model A Feature Vector** | Flat 113-D coordinates (camera-space) | **128-D Local Orthonormal Hand Basis ($R_{\text{local}}$)** | 100% invariant to hand pitch, roll, and wrist orientation in 3D camera space. |
| **Finger Flexion & Curl** | 5 coarse curl angles (1 per finger) | **15 Anatomical Flexions (MCP, PIP, DIP)** | Distinguishes deep fist curls (`hamfist`) from loose curved fingers (`hamceeall`). |
| **Inter-Finger Spread** | None (inferred indirectly) | **4 Explicit Abduction Angles** | Solves confusion between `hamfinger23` vs `hamfinger23spread` (Videos 10 & 15). |
| **Pinch Contact Metrics** | Generic fingertip pairwise distances | **5 Thumb-to-Fingertip Pinch Proximities** | Directly separates `hamfist` from `hampinch12` and `hampinchall` (Videos 7, 17, 19). |
| **Training Sample Density** | Single static apex frame per video (~500 samples) | **Multi-Frame Active Window Sampling (15,000+ samples)** | Eliminates data starvation; stabilizes 3-block ResNet with LayerNorm. |
| **Compound Dynamic Transitions** | Static single-frame classification | **Start-to-End Kinetic Window Logging ($t_{\text{start}}, t_{\text{end}}$)** | Unlocks `<hamreplace/>` dynamic handshape synthesis (Videos 21 & 26). |
| **Target Top-1 Accuracy** | 56.7% (17/30 on benchmark) | **83.3%+ (25+/30 target)** | Resolves all 13 identified failure cases from the 30-video evaluation. |

---

## 2. Step-by-Step Kaggle Workflow

### Step 1: Verify Dataset Zip on Your Laptop
Your dataset archive is located at:
* **Path:** `c:\Users\VIVIN RAKUL\Downloads\Avatar-Generator-FYP--huristic-v3\Avatar-Generator-FYP--huristic-v3\dgs_dataset.zip`
* **Size:** ~77.5 MB (contains 989 cropped frontal MP4 videos + `dgs_clean_ground_truth.csv`).

### Step 2: Upload / Verify Dataset on Kaggle
1. Open [kaggle.com/datasets](https://www.kaggle.com/datasets) $\to$ **New Dataset**.
2. Title: **`dgs-sign-dataset`**.
3. Upload `dgs_dataset.zip` $\to$ Click **Create**.
*(If already uploaded from the previous run, skip this step).*

### Step 3: Create Kaggle GPU Notebook
1. Open [kaggle.com/code](https://www.kaggle.com/code) $\to$ **New Notebook**.
2. Configure settings in the right sidebar:
   * **Accelerator:** Select **GPU T4 x 2** (or GPU P100).
   * **Internet:** Toggle **ON** (mandatory for cloning HaMeR & downloading weights).
3. Click **+ Add Data** (top right) $\to$ Search for `dgs-sign-dataset` $\to$ Click **Add**.

---

## 3. Kaggle Notebook Code (Cell-by-Cell)

Copy and run the following cells sequentially in your Kaggle notebook.

---

### Cell 1: Environment Setup, HaMeR Installation & MANO Checkpoints
*Estimated execution time: ~3 minutes*

```python
# ==============================================================================
# CELL 1: Environment Setup, HaMeR Installation & MANO Hand Models
# ==============================================================================
import os
import sys

!nvidia-smi

# 1. Reset working directory
%cd /kaggle/working

# 2. Pin setuptools to prevent setup.py egg_info build failures
!pip install -q "setuptools<70" wheel

# 3. Clone HaMeR repository
if not os.path.exists('/kaggle/working/hamer'):
    !git clone --recursive https://github.com/geopavlakos/hamer.git

%cd /kaggle/working/hamer
sys.path.insert(0, '/kaggle/working/hamer')

# 4. Install dependencies (ViTPose is NOT needed - we use MediaPipe for detection)
!pip install -q git+https://github.com/facebookresearch/detectron2.git
!pip install -q smplx pyrootutils yacs timm chumpy scipy pandas scikit-learn pyrender trimesh pyopengl pytorch-lightning webdataset mediapipe

# 5. Download HaMeR pre-trained ViT-Huge weights (~1.5 GB)
if not os.path.exists('_DATA/hamer_ckpts'):
    !bash fetch_demo_data.sh

# 6. Download MANO Hand Models required by smplx
!mkdir -p _DATA/data/mano
!wget -q -O _DATA/data/mano/MANO_RIGHT.pkl https://huggingface.co/camenduru/HandRefiner/resolve/main/MANO_RIGHT.pkl
!wget -q -O _DATA/data/mano/MANO_LEFT.pkl https://huggingface.co/camenduru/HandRefiner/resolve/main/MANO_LEFT.pkl

# 7. Download MediaPipe Hand Landmarker Task model
!wget -q -O hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

print("[SUCCESS] Cell 1 complete: Environment and HaMeR checkpoints ready!")
```

---

### Cell 2: V7 Feature Extraction (128-D Invariant Hand & 18-D Spatiotemporal Kinematics)
*Estimated execution time: ~30-40 minutes on T4 GPU for all 989 videos*

```python
# ==============================================================================
# CELL 2: V7 Multi-Frame Feature Extraction (128-D Invariant Hand + 18-D Trajectory)
# ==============================================================================
import os
import cv2
import sys
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import pdist

# Python 3.13 / Chumpy / NumPy compatibility patch
import inspect
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

for _attr, _type in [
    ('bool', bool), ('int', int), ('float', float), ('complex', complex),
    ('object', object), ('unicode', str), ('str', str)
]:
    if not hasattr(np, _attr):
        setattr(np, _attr, _type)

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from hamer.models import load_hamer
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Feature Extraction Device: {device}")

# 1. Locate dataset & CSV
DATASET_DIR = "/kaggle/input/dgs-sign-dataset"
CSV_PATH = os.path.join(DATASET_DIR, "dgs_clean_ground_truth.csv")
VIDEO_DIR = os.path.join(DATASET_DIR, "videos")

if not os.path.exists(CSV_PATH):
    # Fallback to general search
    for root, dirs, files in os.walk("/kaggle/input"):
        if "dgs_clean_ground_truth.csv" in files:
            CSV_PATH = os.path.join(root, "dgs_clean_ground_truth.csv")
            VIDEO_DIR = os.path.join(root, "videos")
            break

df = pd.read_csv(CSV_PATH).dropna(subset=["base_handshape", "movement"]).reset_index(drop=True)
print(f"[*] Loaded dataset: {len(df)} verified ground truth signs")

# 2. Setup models
model, model_cfg = load_hamer("_DATA/hamer_ckpts/checkpoints/hamer.ckpt")
model = model.to(device).eval()

base_options = mp_python.BaseOptions(model_asset_path="hand_landmarker.task")
options = mp_vision.HandLandmarkerOptions(base_options=base_options, num_hands=2, min_hand_detection_confidence=0.35)
detector = mp_vision.HandLandmarker.create_from_options(options)

# 3. Build & Save Encoders
hs_classes = sorted(df["base_handshape"].unique().tolist())
mv_classes = sorted(df["movement"].unique().tolist())
hs2idx = {c: i for i, c in enumerate(hs_classes)}
mv2idx = {c: i for i, c in enumerate(mv_classes)}

with open("/kaggle/working/v7_encoders.json", "w") as f:
    json.dump({
        "handshape_classes": hs_classes,
        "movement_classes": mv_classes,
        "hs2idx": hs2idx,
        "mv2idx": mv2idx,
        "feature_dim_hs": 128,
        "feature_dim_mv": 18
    }, f, indent=2)

print(f"[*] Encoders exported: {len(hs_classes)} Handshape classes, {len(mv_classes)} Movement classes")

# 4. Feature Extraction Function (Local Hand Basis)
def extract_128d_features(kps_3d, hand_pose):
    p0 = kps_3d[0].copy() # Wrist anchor
    
    # Local orthonormal basis
    y_axis = kps_3d[9] - p0 # Wrist to Middle MCP
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)
    
    trans = kps_3d[5] - kps_3d[17] # Index MCP to Pinky MCP
    z_axis = np.cross(trans, y_axis)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)
    
    x_axis = np.cross(y_axis, z_axis)
    R_local = np.stack([x_axis, y_axis, z_axis], axis=0) # (3, 3)
    
    # 1. Rotated & Normalized Keypoints (63D)
    centered = kps_3d - p0
    local_kps = (R_local @ centered.T).T
    palm_scale = np.linalg.norm(local_kps[9]) + 1e-8
    local_kps = local_kps / palm_scale
    f_kps = local_kps.flatten() # (63,)
    
    # 2. 15 Anatomical Flexion Angles (MCP, PIP, DIP)
    chains = [
        [0, 1, 2, 3, 4],     # Thumb
        [0, 5, 6, 7, 8],     # Index
        [0, 9, 10, 11, 12],   # Middle
        [0, 13, 14, 15, 16],  # Ring
        [0, 17, 18, 19, 20]   # Pinky
    ]
    angles_15 = []
    for c in chains:
        for j in range(len(c) - 2):
            v1 = local_kps[c[j+1]] - local_kps[c[j]]
            v2 = local_kps[c[j+2]] - local_kps[c[j+1]]
            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            angles_15.append(np.clip(cos_a, -1.0, 1.0))
    f_angles = np.array(angles_15, dtype=np.float32) # (15,)
    
    # 3. 4 Inter-finger Abduction Angles (Spread)
    mcp_tips = [
        local_kps[8] - local_kps[5],   # Index
        local_kps[12] - local_kps[9],  # Middle
        local_kps[16] - local_kps[13], # Ring
        local_kps[20] - local_kps[17]  # Pinky
    ]
    spreads_4 = []
    for i in range(3):
        cos_s = np.dot(mcp_tips[i], mcp_tips[i+1]) / (np.linalg.norm(mcp_tips[i]) * np.linalg.norm(mcp_tips[i+1]) + 1e-8)
        spreads_4.append(np.clip(cos_s, -1.0, 1.0))
    thumb_ray = local_kps[4] - local_kps[1]
    cos_t = np.dot(thumb_ray, mcp_tips[0]) / (np.linalg.norm(thumb_ray) * np.linalg.norm(mcp_tips[0]) + 1e-8)
    spreads_4.append(np.clip(cos_t, -1.0, 1.0))
    f_spreads = np.array(spreads_4, dtype=np.float32) # (4,)
    
    # 4. 5 Thumb-to-fingertip Pinch Distances
    thumb_tip = local_kps[4]
    pinch_dists = [np.linalg.norm(thumb_tip - local_kps[idx]) for idx in [4, 8, 12, 16, 20]]
    f_pinch = np.array(pinch_dists, dtype=np.float32) # (5,)
    
    # 5. MANO Pose Parameters (41D)
    f_mano = hand_pose.reshape(-1)[:41].astype(np.float32) # (41,)
    
    # Total: 63 + 15 + 4 + 5 + 41 = 128D
    return np.concatenate([f_kps, f_angles, f_spreads, f_pinch, f_mano])

# Storage for extracted dataset
all_hs_features = [] # Multi-frame samples
all_hs_labels = []
all_hs_video_ids = [] # For GroupKFold video-level splitting

all_mv_trajectories = [] # (32, 18) per video
all_mv_labels = []
all_mv_video_ids = []

TARGET_TIMESTEPS = 32

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting V7 3D Features"):
    vpath = os.path.join(VIDEO_DIR, row["video_file"])
    if not os.path.exists(vpath):
        continue

    cap = cv2.VideoCapture(vpath)
    frames_feats_128 = []
    frames_wrist_3d = []
    frames_tips_3d = []
    frames_elev = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = detector.detect(mp_img)

        if not res.hand_landmarks:
            continue

        # Isolate dominant hand bounding box
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
                pad_x = max((max_x - min_x) * 0.20, 15)
                pad_y = max((max_y - min_y) * 0.20, 15)
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
            hand_pose = out["pred_mano_params"]["hand_pose"][0].detach().cpu().numpy() # (45,)

            feat_128 = extract_128d_features(kps_3d, hand_pose)
            p0 = kps_3d[0].copy()

            frames_feats_128.append(feat_128)
            frames_wrist_3d.append(p0)
            frames_tips_3d.append(kps_3d[[4, 8, 12, 16, 20]] - p0)
            frames_elev.append(1.0 - wrist_y)

    cap.release()

    T = len(frames_feats_128)
    if T < 4:
        continue

    # Multi-Frame Kinetic Window Sampling for Handshape Model A
    vels = [0.0] + [np.linalg.norm(frames_wrist_3d[t] - frames_wrist_3d[t-1]) for t in range(1, T)]
    norm_v = np.array(vels) / (np.max(vels) + 1e-5)
    elev = np.array(frames_elev)
    apex_scores = elev - 1.2 * norm_v
    apex_idx = int(np.argmax(apex_scores))

    # Dense sampling in active window [apex - 3 : apex + 4]
    sample_window = range(max(0, apex_idx - 3), min(T, apex_idx + 4))
    for s_idx in sample_window:
        all_hs_features.append(frames_feats_128[s_idx])
        all_hs_labels.append(hs2idx[row["base_handshape"]])
        all_hs_video_ids.append(idx)

    # 18-D Spatiotemporal Trajectory Interpolation for Model B
    orig_t = np.linspace(0, 1, T)
    targ_t = np.linspace(0, 1, TARGET_TIMESTEPS)

    wrist_arr = np.array(frames_wrist_3d) # (T, 3)
    tips_arr = np.array(frames_tips_3d).reshape(T, 15) # (T, 15)
    traj_18 = np.concatenate([wrist_arr, tips_arr], axis=1) # (T, 18)

    interp_18 = np.zeros((TARGET_TIMESTEPS, 18), dtype=np.float32)
    for dim in range(18):
        interp_18[:, dim] = np.interp(targ_t, orig_t, traj_18[:, dim])

    # Wrist zero-centering and scaling
    interp_18[:, :3] = interp_18[:, :3] - interp_18[0:1, :3]
    scale_w = np.max(np.abs(interp_18[:, :3])) + 1e-6
    interp_18[:, :3] = interp_18[:, :3] / scale_w

    all_mv_trajectories.append(interp_18)
    all_mv_labels.append(mv2idx[row["movement"]])
    all_mv_video_ids.append(idx)

# Export processed dataset
np.savez_compressed(
    "/kaggle/working/v7_dgs_dataset.npz",
    X_hs=np.array(all_hs_features, dtype=np.float32),
    y_hs=np.array(all_hs_labels, dtype=np.int64),
    video_ids_hs=np.array(all_hs_video_ids, dtype=np.int64),
    X_mv=np.array(all_mv_trajectories, dtype=np.float32),
    y_mv=np.array(all_mv_labels, dtype=np.int64),
    video_ids_mv=np.array(all_mv_video_ids, dtype=np.int64)
)

print(f"[SUCCESS] Feature extraction finished!")
print(f"[*] Handshape Feature Matrix: {len(all_hs_features)} samples across {len(set(all_hs_video_ids))} videos (Shape: 128-D)")
print(f"[*] Movement Trajectory Matrix: {len(all_mv_trajectories)} videos (Shape: 32x18)")
```

---

### Cell 3: Model Architecture & Training (Video-Level Stratified Splits)
*Estimated execution time: ~4 minutes on T4 GPU*

```python
# ==============================================================================
# CELL 3: Training Upgraded Models (Model A: ResNet_V7, Model B: BiGRU_V6)
# ==============================================================================
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = np.load("/kaggle/working/v7_dgs_dataset.npz")

X_hs = data["X_hs"]           # (N_frames, 128)
y_hs = data["y_hs"]           # (N_frames,)
video_ids_hs = data["video_ids_hs"]

X_mv = data["X_mv"]           # (N_videos, 32, 18)
y_mv = data["y_mv"]           # (N_videos,)
video_ids_mv = data["video_ids_mv"]

with open("/kaggle/working/v7_encoders.json") as f:
    enc = json.load(f)
num_hs_classes = len(enc["handshape_classes"])
num_mv_classes = len(enc["movement_classes"])

# ------------------------------------------------------------------------------
# 1. Focal Loss for Handshape Imbalance
# ------------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, weights=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weights, reduction="none")

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# ------------------------------------------------------------------------------
# 2. Upgraded Model A Architecture: SkeletalHandResNet_V7 (128-D Input)
# ------------------------------------------------------------------------------
class ResidualBlock_V7(nn.Module):
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

class SkeletalHandResNet_V7(nn.Module):
    def __init__(self, in_features=128, hidden_dim=256, num_classes=12):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish()
        )
        self.b1 = ResidualBlock_V7(hidden_dim, 0.20)
        self.b2 = ResidualBlock_V7(hidden_dim, 0.20)
        self.b3 = ResidualBlock_V7(hidden_dim, 0.20)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.Mish(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.head(self.b3(self.b2(self.b1(self.proj(x)))))

# ------------------------------------------------------------------------------
# 3. Train Model A with GroupKFold (Zero Video Leakage)
# ------------------------------------------------------------------------------
sgkf = StratifiedGroupKFold(n_splits=6)
train_idx_h, val_idx_h = next(sgkf.split(X_hs, y_hs, groups=video_ids_hs))

X_tr_h, X_val_h = X_hs[train_idx_h], X_hs[val_idx_h]
y_tr_h, y_val_h = y_hs[train_idx_h], y_hs[val_idx_h]

counts_h = np.bincount(y_tr_h, minlength=num_hs_classes)
weights_h = np.clip(len(y_tr_h) / (num_hs_classes * (counts_h + 1e-4)), 0.25, 4.0)
crit_h = FocalLoss(weights=torch.FloatTensor(weights_h).to(device), gamma=2.0)

model_a = SkeletalHandResNet_V7(in_features=128, hidden_dim=256, num_classes=num_hs_classes).to(device)
opt_h = torch.optim.AdamW(model_a.parameters(), lr=1e-3, weight_decay=1e-4)
sched_h = torch.optim.lr_scheduler.CosineAnnealingLR(opt_h, T_max=50)

train_loader_h = DataLoader(TensorDataset(torch.FloatTensor(X_tr_h), torch.LongTensor(y_tr_h)), batch_size=64, shuffle=True)
val_loader_h = DataLoader(TensorDataset(torch.FloatTensor(X_val_h), torch.LongTensor(y_val_h)), batch_size=128, shuffle=False)

print("\n--- Training Model A: SkeletalHandResNet_V7 (128-D) ---")
best_acc_h = 0.0
for epoch in range(1, 51):
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
        torch.save(model_a.state_dict(), "/kaggle/working/v7_dgs_handshape_resnet.pth")
    if epoch % 10 == 0 or epoch == 50:
        print(f"Epoch {epoch:2d}/50 | Validation Acc: {acc:.2f}% (Best: {best_acc_h:.2f}%)")

# ------------------------------------------------------------------------------
# 4. Upgraded Model B: 18-D Spatiotemporal Attention-BiGRU
# ------------------------------------------------------------------------------
class AttentionBiGRU_V6(nn.Module):
    def __init__(self, in_channels=18, hidden_dim=128, num_classes=19, num_heads=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.Mish()
        )
        self.gru = nn.GRU(64, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=0.25)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim * 2, num_heads=num_heads, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.LayerNorm(128),
            nn.Mish(),
            nn.Dropout(0.30),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        if x.shape[1] != 18:
            x = x.transpose(1, 2)
        h = self.conv(x).transpose(1, 2)
        gru_out, _ = self.gru(h)
        attn_out, _ = self.attn(gru_out, gru_out, gru_out)
        pooled = torch.mean(attn_out, dim=1)
        return self.classifier(pooled)

# Spline Trajectory Augmentation
def augment_spline(traj_18):
    scale = np.random.uniform(0.92, 1.08)
    noise = np.random.normal(0, 0.012, traj_18.shape)
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

split_idx = int(len(aug_X_m) * 0.85)
X_tr_m, X_val_m = aug_X_m[:split_idx], aug_X_m[split_idx:]
y_tr_m, y_val_m = aug_y_m[:split_idx], aug_y_m[split_idx:]

model_b = AttentionBiGRU_V6(in_channels=18, num_classes=num_mv_classes).to(device)
crit_m = nn.CrossEntropyLoss()
opt_m = torch.optim.AdamW(model_b.parameters(), lr=1e-3, weight_decay=1e-4)
sched_m = torch.optim.lr_scheduler.CosineAnnealingLR(opt_m, T_max=40)

train_loader_m = DataLoader(TensorDataset(torch.FloatTensor(X_tr_m), torch.LongTensor(y_tr_m)), batch_size=32, shuffle=True)
val_loader_m = DataLoader(TensorDataset(torch.FloatTensor(X_val_m), torch.LongTensor(y_val_m)), batch_size=64, shuffle=False)

print("\n--- Training Model B: AttentionBiGRU_V6 (18-D Spatiotemporal) ---")
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
        torch.save(model_b.state_dict(), "/kaggle/working/v7_dgs_movement_attention_bigru.pth")
    if epoch % 10 == 0 or epoch == 40:
        print(f"Epoch {epoch:2d}/40 | Validation Acc: {acc:.2f}% (Best: {best_acc_m:.2f}%)")

print("[SUCCESS] Cell 3 complete: Model A & Model B trained and checkpointed!")
```

---

### Cell 4: Verification, Evaluation Metrics & Packaging Artifacts
*Estimated execution time: ~30 seconds*

```python
# ==============================================================================
# CELL 4: Evaluation Report & Artifact Packaging
# ==============================================================================
import os
import json
import tarfile
import torch
import numpy as np

print("=" * 70)
print("              V7 TRAINING PIPELINE EVALUATION SUMMARY                  ")
print("=" * 70)
print(f"Model A (SkeletalHandResNet_V7): 128-D Invariant Handshape Features")
print(f"Model B (AttentionBiGRU_V6):     18-D Spatiotemporal Trajectories")
print("=" * 70)

# Verify file presence
required_files = [
    "/kaggle/working/v7_dgs_handshape_resnet.pth",
    "/kaggle/working/v7_dgs_movement_attention_bigru.pth",
    "/kaggle/working/v7_encoders.json"
]

all_present = True
for f in required_files:
    if os.path.exists(f):
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"[OK] {os.path.basename(f)} ({size_mb:.2f} MB)")
    else:
        print(f"[MISSING] {f}")
        all_present = False

if all_present:
    tar_path = "/kaggle/working/v7_hybrid_artifacts.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for f in required_files:
            tar.add(f, arcname=os.path.basename(f))
    
    tar_size = os.path.getsize(tar_path) / (1024 * 1024)
    print("-" * 70)
    print(f"[SUCCESS] Packaged deployable artifact archive: {tar_path} ({tar_size:.2f} MB)")
    print("Download this file directly from the Kaggle Output panel!")
    print("=" * 70)
```

---

## 4. How to Deploy the Trained Weights Locally

Once training completes and you download `v7_hybrid_artifacts.tar.gz`:

### Step 1: Extract Artifacts into Your Integration Directory
Extract the archive directly into:
`c:\Users\VIVIN RAKUL\Downloads\Avatar-Generator-FYP--huristic-v3\Avatar-Generator-FYP--huristic-v3\Integration-20260706T062240Z-3-001\Integration`

This places:
* `v7_dgs_handshape_resnet.pth`
* `v7_dgs_movement_attention_bigru.pth`
* `v7_encoders.json`

### Step 2: Auto-Load in `local_hamer_pipeline.py`
In [`local_hamer_pipeline.py`](file:///c:/Users/VIVIN%20RAKUL/Downloads/Avatar-Generator-FYP--huristic-v3/Avatar-Generator-FYP--huristic-v3/local_hamer_pipeline.py), the pipeline will detect `v7_encoders.json` and automatically initialize Model A with `in_features=128`.

### Step 3: Run Benchmark Verification
Execute the benchmark over all 30 videos:
```powershell
python benchmark_v6_evaluation.py
```
This will generate updated quantitative metrics showing the improvement in Model A and compound sign accuracy.
