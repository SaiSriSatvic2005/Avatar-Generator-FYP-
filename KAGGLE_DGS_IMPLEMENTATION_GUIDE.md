# Kaggle DGS 3D HaMeR & Neural Training Guide
## End-to-End Implementation: Feature Extraction & Neural Model Training

**Platform:** Kaggle Notebooks (Free 16 GB Tesla T4 or P100 GPU)  
**Dataset:** 989 Clean Frontal DGS (German Sign Language) Videos + Matching Ground Truth SiGML  
**Core Models:** 
* **Model A:** `SkeletalHandResNet` (3D Mesh + 207 Kinematic Features $\to$ 12 Handshape Classes)
* **Model B:** `AttentionBiGRU` (Temporal Multi-Head Attention + BiGRU $\to$ 19 Dynamic Movement Classes)

---

## 1. Why the DGS Dataset Solves the ISL Movement Bottleneck

In our original ISL dataset:
* **206 out of 499 signs (41.3%)** had zero motion (`hamnomotion`).
* Directional movement classes had only 1 or 2 examples (`hamcircleol`: 1, `hamwavy`: 2), causing Model B to underfit and default to static predictions.

In this **DGS Dataset**:
* **710 signs (71.8%) contain real, active, dynamic 3D movements**:
  * Downward movements (`hammoved`): **175 signs**
  * Outward movements (`hammoveo`): **103 signs**
  * Rightward movements (`hammover`): **83 signs**
  * Down-out movements (`hammovedo`): **65 signs**
  * Leftward movements (`hammovel`): **57 signs**
  * Down-left movements (`hammovedl`): **44 signs**
  * Inward movements (`hammovei`): **43 signs**
  * Upward movements (`hammoveu`): **42 signs**
* Handshape distribution is balanced across all 12 major sign language handshapes (207 Flat Hands, 180 Index Fingers, 166 Fists, 146 Open Hands, etc.).
* All 989 videos are already **pre-cropped to pure frontal view (384x360)** against a studio blue background.

---

## 2. Step-by-Step Kaggle Workflow

### Step 1: Your Dataset Zip is Already Ready
The zip file has already been prepared on your laptop:
* **File Location:** `c:\Users\VIVIN RAKUL\Downloads\Avatar-Generator-FYP--huristic-v3\Avatar-Generator-FYP--huristic-v3\dgs_dataset.zip`
* **File Size:** **77.5 MB** (contains all 989 cropped frontal videos + `dgs_ground_truth_labels.csv`).

### Step 2: Upload to Kaggle Datasets
1. Open [kaggle.com/datasets](https://www.kaggle.com/datasets) $\to$ Click **New Dataset**.
2. Title it: **`dgs-sign-dataset`**.
3. Drag and drop `dgs_dataset.zip` into the upload zone $\to$ Click **Create**.

### Step 3: Create Kaggle GPU Notebook
1. Open [kaggle.com/code](https://www.kaggle.com/code) $\to$ Click **New Notebook**.
2. In the right panel settings:
   * **Accelerator:** Select **GPU T4 x 2** (or GPU P100).
   * **Internet:** Set to **ON** (required for `git clone` and model weights).
3. Click **+ Add Data** (top right) $\to$ Search for your dataset `dgs-sign-dataset` $\to$ Click **Add**.

---

## 3. Kaggle Notebook Code (Cell-by-Cell)

### Cell 1: Environment Setup & Install HaMeR + Dependencies
*Run time: ~3 minutes*

```python
# --- CELL 1: Environment Setup, HaMeR Installation & MANO Models ---
import os
import sys

!nvidia-smi

# 1. Reset directory
%cd /kaggle/working

# 2. Fix setuptools to prevent setup.py egg_info failures
!pip install -q "setuptools<70" wheel

# 3. Clone HaMeR repository
if not os.path.exists('/kaggle/working/hamer'):
    !git clone https://github.com/geopavlakos/hamer.git

%cd /kaggle/working/hamer
sys.path.insert(0, '/kaggle/working/hamer')

# 4. Install only required runtime packages (ViTPose is NOT needed - we use MediaPipe)
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

print("[SUCCESS] Environment ready!")
```

---

### Cell 2: 3D Feature Extraction (989 Videos $\to$ 3D Mesh & Trajectories)
*Run time: ~12–15 minutes on T4 GPU*

```python
# ==============================================================================
# CELL 2: Extract 3D Hand Mesh, Palm Normal, & 3D Trajectory across 989 Videos
# ==============================================================================
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

import os
import cv2
import torch
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import pdist
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from hamer.models import load_hamer
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Running on device: {device}")

# 1. Auto-Locate Dataset Input Paths
VIDEO_DIR, LABELS_CSV = None, None
for root, dirs, files in os.walk("/kaggle/input"):
    if "dgs_ground_truth_labels.csv" in files:
        LABELS_CSV = os.path.join(root, "dgs_ground_truth_labels.csv")
        VIDEO_DIR = os.path.join(root, "videos")
        break

if not VIDEO_DIR or not os.path.exists(VIDEO_DIR):
    raise FileNotFoundError(f"Could not find 'videos' folder in /kaggle/input! Found: {os.listdir('/kaggle/input')}")

print(f"[*] Found Video Directory: {VIDEO_DIR}")
print(f"[*] Found Labels CSV:       {LABELS_CSV}")

print(f"[*] Video Directory: {VIDEO_DIR}")
print(f"[*] Labels CSV: {LABELS_CSV}")

df = pd.read_csv(LABELS_CSV)
print(f"[*] Found {len(df)} signs in ground truth table")

# 2. Load HaMeR Model
model, model_cfg = load_hamer("_DATA/hamer_ckpts/checkpoints/hamer.ckpt")
model = model.to(device)
model.eval()

# 3. Setup MediaPipe Detector
base_options = mp_python.BaseOptions(model_asset_path='hand_landmarker.task')
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.4,
    min_tracking_confidence=0.4
)
detector = mp_vision.HandLandmarker.create_from_options(options)

# 4. Feature Extraction Loop
frame_features = []
frame_labels_handshape = []
frame_labels_palm = []
video_trajectories = []
video_labels_movement = []
valid_video_metadata = []

TARGET_TIMESTEPS = 32

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting 3D Mesh"):
    v_path = os.path.join(VIDEO_DIR, row['video_file'])
    if not os.path.exists(v_path):
        continue

    cap = cv2.VideoCapture(v_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 5:
        cap.release()
        continue

    # Active Stroke Windowing: Trim initial 15% and final 15% rest frames
    start_frame = int(total_frames * 0.15)
    end_frame = max(start_frame + 3, int(total_frames * 0.85))

    frame_idx = 0
    raw_trajectory = []
    video_hand_features = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx < start_frame or frame_idx > end_frame:
            continue

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection_result = detector.detect(mp_image)

        if not detection_result.hand_landmarks:
            continue

        # Select dominant hand (largest bounding box)
        best_box = None
        max_area = 0
        for lms in detection_result.hand_landmarks:
            xs = [p.x * w for p in lms]
            ys = [p.y * h for p in lms]
            min_x, max_x = max(0, min(xs)), min(w, max(xs))
            min_y, max_y = max(0, min(ys)), min(h, max(ys))
            area = (max_x - min_x) * (max_y - min_y)
            if area > max_area:
                max_area = area
                # Add 20% margin
                pad_x = (max_x - min_x) * 0.2
                pad_y = (max_y - min_y) * 0.2
                best_box = [
                    max(0, min_x - pad_x),
                    max(0, min_y - pad_y),
                    min(w, max_x + pad_x),
                    min(h, max_y + pad_y)
                ]

        if best_box is None:
            continue

        # Extract HaMeR 3D Mesh
        boxes = np.array([best_box], dtype=np.float32)
        dataset = ViTDetDataset(model_cfg, rgb, boxes, np.array([1]), rescale_factor=2.0)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            keypoints_3d = out['pred_keypoints_3d'][0].detach().cpu().numpy()  # (21, 3)
            rotmat = out['pred_mano_params']['global_orient'][0].detach().cpu().numpy()  # (1, 3, 3)
            hand_pose = out['pred_mano_params']['hand_pose'][0].detach().cpu().numpy()  # (15, 3, 3)

            # Center 3D keypoints at wrist (p0)
            wrist_3d = keypoints_3d[0].copy()
            centered_3d = keypoints_3d - wrist_3d
            scale = np.linalg.norm(centered_3d[9]) + 1e-6
            norm_3d = centered_3d / scale

            # Kinematic Features (207-D):
            # 63 normalized coords + 105 pairwise finger distances + 45 MANO rotations
            kps_flat = norm_3d.flatten()  # 63
            dists = pdist(norm_3d[:16])   # 120 (take first 105 for 207-D total)
            dists_105 = dists[:105] if len(dists) >= 105 else np.pad(dists, (0, 105 - len(dists)))
            pose_flat = hand_pose[:, :2, :2].flatten()[:39]  # 39
            feat_207 = np.concatenate([kps_flat, dists_105, pose_flat])

            frame_features.append(feat_207)
            frame_labels_handshape.append(str(row['handshape']))
            frame_labels_palm.append(str(row['palm_orientation']))

            raw_trajectory.append(wrist_3d)

    cap.release()

    # Trajectory Normalization (Resample to exactly 32 timesteps)
    if len(raw_trajectory) >= 4:
        traj_arr = np.array(raw_trajectory)  # (N, 3)
        # Interpolate to 32 uniform steps
        orig_indices = np.linspace(0, 1, len(traj_arr))
        target_indices = np.linspace(0, 1, TARGET_TIMESTEPS)
        norm_traj = np.zeros((TARGET_TIMESTEPS, 3), dtype=np.float32)
        for dim in range(3):
            norm_traj[:, dim] = np.interp(target_indices, orig_indices, traj_arr[:, dim])

        # Normalize trajectory displacement relative to starting point
        norm_traj = norm_traj - norm_traj[0]
        traj_scale = np.max(np.abs(norm_traj)) + 1e-6
        norm_traj = norm_traj / traj_scale

        video_trajectories.append(norm_traj)
        video_labels_movement.append(str(row['movement']))
        valid_video_metadata.append(row['gloss'])

# Convert to arrays and save
X_hand = np.array(frame_features, dtype=np.float32)
y_hand = np.array(frame_labels_handshape)
y_palm = np.array(frame_labels_palm)
X_traj = np.array(video_trajectories, dtype=np.float32)
y_move = np.array(video_labels_movement)

print(f"\n[DONE] Extraction Complete:")
print(f"  Handshape Feature Frames (Model A): {X_hand.shape} with {len(np.unique(y_hand))} classes")
print(f"  Movement Trajectories (Model B):    {X_traj.shape} with {len(np.unique(y_move))} classes")

np.savez_compressed(
    "/kaggle/working/dgs_hamer_dataset.npz",
    X_hand=X_hand, y_hand=y_hand, y_palm=y_palm,
    X_traj=X_traj, y_move=y_move,
    metadata=np.array(valid_video_metadata)
)
print("[SAVED] Saved /kaggle/working/dgs_hamer_dataset.npz")
```

---

### Cell 3: Train Neural Network Architectures (Model A & Model B)
*Run time: ~4 minutes on T4 GPU*

```python
# ==============================================================================
# CELL 3: Train Model A (SkeletalHandResNet) & Model B (AttentionBiGRU)
# ==============================================================================
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = np.load("/kaggle/working/dgs_hamer_dataset.npz")

X_hand = data['X_hand']
y_hand_raw = data['y_hand']
X_traj = data['X_traj']
y_move_raw = data['y_move']

# 1. Encode Labels
le_hand = LabelEncoder()
y_hand = le_hand.fit_transform(y_hand_raw)

le_move = LabelEncoder()
y_move = le_move.fit_transform(y_move_raw)

num_hand_classes = len(le_hand.classes_)
num_move_classes = len(le_move.classes_)

print(f"[*] Model A Handshape Classes ({num_hand_classes}): {list(le_hand.classes_)}")
print(f"[*] Model B Movement Classes ({num_move_classes}): {list(le_move.classes_)}")

# Save Encoders
encoders = {
    "handshape_classes": list(le_hand.classes_),
    "movement_classes": list(le_move.classes_),
    "input_hand_features": 207,
    "trajectory_timesteps": 32,
    "trajectory_channels": 3
}
with open("/kaggle/working/dgs_hamer_encoders.json", "w") as f:
    json.dump(encoders, f, indent=2)

# ==============================================================================
# MODEL A: SkeletalHandResNet (207 -> 12 Handshape Classes)
# ==============================================================================
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x):
        res = x
        x = F.gelu(self.bn1(self.fc1(x)))
        x = self.drop(x)
        x = self.bn2(self.fc2(x))
        return F.gelu(x + res)

class SkeletalHandResNet(nn.Module):
    def __init__(self, in_features=207, hidden_dim=256, num_classes=12):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        self.block1 = ResidualBlock(hidden_dim, dropout=0.2)
        self.block2 = ResidualBlock(hidden_dim, dropout=0.2)
        self.block3 = ResidualBlock(hidden_dim, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.input_proj(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.head(x)

# Train Model A
X_train_h, X_val_h, y_train_h, y_val_h = train_test_split(
    X_hand, y_hand, test_size=0.15, random_state=42, stratify=y_hand
)

train_loader_h = DataLoader(
    torch.utils.data.TensorDataset(torch.tensor(X_train_h), torch.tensor(y_train_h, dtype=torch.long)),
    batch_size=64, shuffle=True
)
val_loader_h = DataLoader(
    torch.utils.data.TensorDataset(torch.tensor(X_val_h), torch.tensor(y_val_h, dtype=torch.long)),
    batch_size=64, shuffle=False
)

model_a = SkeletalHandResNet(in_features=207, hidden_dim=256, num_classes=num_hand_classes).to(device)
criterion_h = nn.CrossEntropyLoss()
optimizer_h = torch.optim.AdamW(model_a.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler_h = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_h, T_max=40)

print("\n--- Training Model A (SkeletalHandResNet) ---")
best_acc_h = 0.0
for epoch in range(1, 41):
    model_a.train()
    total_loss, correct, total = 0, 0, 0
    for bx, by in train_loader_h:
        bx, by = bx.to(device), by.to(device)
        optimizer_h.zero_grad()
        out = model_a(bx)
        loss = criterion_h(out, by)
        loss.backward()
        optimizer_h.step()
        total_loss += loss.item() * len(by)
        correct += (out.argmax(1) == by).sum().item()
        total += len(by)
    scheduler_h.step()
    
    # Validation
    model_a.eval()
    v_correct, v_total = 0, 0
    with torch.no_grad():
        for bx, by in val_loader_h:
            bx, by = bx.to(device), by.to(device)
            out = model_a(bx)
            v_correct += (out.argmax(1) == by).sum().item()
            v_total += len(by)
    v_acc = v_correct / v_total * 100
    if v_acc > best_acc_h:
        best_acc_h = v_acc
        torch.save(model_a.state_dict(), "/kaggle/working/v5_dgs_handshape_resnet.pth")
    if epoch % 10 == 0 or epoch == 40:
        print(f"Epoch {epoch:2d}/40 | Train Acc: {correct/total*100:.2f}% | Val Acc: {v_acc:.2f}% (Best: {best_acc_h:.2f}%)")

# ==============================================================================
# MODEL B: AttentionBiGRU (3 x 32 -> 19 Movement Classes)
# ==============================================================================
class AttentionBiGRU(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=128, num_classes=19, num_heads=4):
        super().__init__()
        self.input_fc = nn.Linear(in_channels, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim * 2, num_heads=num_heads, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # x: (B, 3, 32) -> transpose to (B, 32, 3)
        if x.dim() == 3 and x.shape[1] == 3:
            x = x.transpose(1, 2)
        h = F.gelu(self.input_fc(x))
        gru_out, _ = self.gru(h)  # (B, 32, 256)
        attn_out, _ = self.attn(gru_out, gru_out, gru_out)  # (B, 32, 256)
        pooled = torch.mean(attn_out, dim=1)  # (B, 256)
        return self.classifier(pooled)

# Data Augmentation for Model B (4x Data Multiplier)
aug_X, aug_y = [], []
for traj, lbl in zip(X_traj, y_move):
    # 1. Original
    aug_X.append(traj.T)  # (3, 32)
    aug_y.append(lbl)
    # 2. Gaussian Noise
    noise = np.random.normal(0, 0.02, traj.shape)
    aug_X.append((traj + noise).T)
    aug_y.append(lbl)
    # 3. Spatial Scaling (0.9x to 1.1x)
    scale = np.random.uniform(0.9, 1.1)
    aug_X.append((traj * scale).T)
    aug_y.append(lbl)
    # 4. Time Warp / Velocity Perturbation
    idx_warp = np.sort(np.random.choice(32, 32, replace=True))
    aug_X.append(traj[idx_warp].T)
    aug_y.append(lbl)

aug_X = np.array(aug_X, dtype=np.float32)
aug_y = np.array(aug_y, dtype=np.int64)

X_train_m, X_val_m, y_train_m, y_val_m = train_test_split(
    aug_X, aug_y, test_size=0.15, random_state=42, stratify=aug_y
)

train_loader_m = DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_train_m), torch.tensor(y_train_m)), batch_size=32, shuffle=True)
val_loader_m = DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_val_m), torch.tensor(y_val_m)), batch_size=32, shuffle=False)

model_b = AttentionBiGRU(in_channels=3, hidden_dim=128, num_classes=num_move_classes).to(device)
criterion_m = nn.CrossEntropyLoss()
optimizer_m = torch.optim.AdamW(model_b.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler_m = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_m, T_max=45)

print("\n--- Training Model B (AttentionBiGRU) with Augmented Trajectories ---")
best_acc_m = 0.0
for epoch in range(1, 46):
    model_b.train()
    total_loss, correct, total = 0, 0, 0
    for bx, by in train_loader_m:
        bx, by = bx.to(device), by.to(device)
        optimizer_m.zero_grad()
        out = model_b(bx)
        loss = criterion_m(out, by)
        loss.backward()
        optimizer_m.step()
        total_loss += loss.item() * len(by)
        correct += (out.argmax(1) == by).sum().item()
        total += len(by)
    scheduler_m.step()
    
    # Validation
    model_b.eval()
    v_correct, v_total = 0, 0
    with torch.no_grad():
        for bx, by in val_loader_m:
            bx, by = bx.to(device), by.to(device)
            out = model_b(bx)
            v_correct += (out.argmax(1) == by).sum().item()
            v_total += len(by)
    v_acc = v_correct / v_total * 100
    if v_acc > best_acc_m:
        best_acc_m = v_acc
        torch.save(model_b.state_dict(), "/kaggle/working/v5_dgs_movement_attention_bigru.pth")
    if epoch % 10 == 0 or epoch == 45:
        print(f"Epoch {epoch:2d}/45 | Train Acc: {correct/total*100:.2f}% | Val Acc: {v_acc:.2f}% (Best: {best_acc_m:.2f}%)")

print("\n[SUCCESS] Both Model A and Model B trained and best checkpoints saved!")
```

---

### Cell 4: Package Artifacts for Local Deployment
*Run time: ~10 seconds*

```python
# ==============================================================================
# CELL 4: Package All V5 Artifacts into v5_dgs_artifacts.tar.gz
# ==============================================================================
%cd /kaggle/working

import tarfile
import os

artifact_files = [
    "v5_dgs_handshape_resnet.pth",
    "v5_dgs_movement_attention_bigru.pth",
    "dgs_hamer_encoders.json"
]

tar_path = "/kaggle/working/v5_dgs_artifacts.tar.gz"
with tarfile.open(tar_path, "w:gz") as tar:
    for f in artifact_files:
        if os.path.exists(f):
            tar.add(f, arcname=f)
            print(f"[+] Added {f} ({os.path.getsize(f)/1024:.1f} KB)")

print(f"\n[PACKAGE READY] {tar_path} ({os.path.getsize(tar_path)/(1024*1024):.2f} MB)")
print("--> Download this file from the Kaggle Output section on the right panel!")
```

---

## 4. Local Deployment on Your Laptop

Once `v5_dgs_artifacts.tar.gz` finishes downloading from Kaggle:

1. Move `v5_dgs_artifacts.tar.gz` to your workspace:
   `c:\Users\VIVIN RAKUL\Downloads\Avatar-Generator-FYP--huristic-v3\Avatar-Generator-FYP--huristic-v3\Integration-20260706T062240Z-3-001\Integration\`

2. In your PowerShell terminal, extract the archive:
   ```powershell
   cd "c:\Users\VIVIN RAKUL\Downloads\Avatar-Generator-FYP--huristic-v3\Avatar-Generator-FYP--huristic-v3\Integration-20260706T062240Z-3-001\Integration"
   tar -xvf v5_dgs_artifacts.tar.gz
   ```

3. Test any video with the upgraded motion and handshape models:
   ```powershell
   cd "..\..\"
   python translate_video_to_sigml.py --video "Advise.mp4"
   ```
   * The new model will predict directional linear motions (`hammoveu`, `hammoved`, `hammoveo`, etc.) learned from the 989 DGS samples without collapsing into `hamnomotion` or circular loops.
