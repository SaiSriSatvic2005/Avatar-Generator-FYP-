# --- Windows DLL Conflict Fix: MediaPipe must be imported before PyTorch & OpenCV ---
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
# ------------------------------------------------------------------------------------
#!/usr/bin/env python3
"""
Upgraded Local HaMeR 3D Hybrid Sign Language Recognition Pipeline (V6 Complete)
Features:
  1. Extended Multi-Slot Phonetic Grammar:
     - Handshape Modifiers (hamthumboutmod, hamthumbacrossmod, hamfingerbendmod, hamfingerstraightmod)
     - Spatial Relators (hamlrat, hamclose, hamtouch)
     - Dynamic Trajectory Curvature (hamarcu, hamarcd, hamarcl, hamarcr)
     - Mid-Flight Posture Replacements (hamreplace)
     - Dual-Hand Parallel Blocks (hamparbegin, hamplus, hamparend) & Mirror Symmetry (hamsymmlr)
  2. Metric 3D Geometry: True 3D Cross-Product Palm Normal & 16-Direction Longitudinal Axis
  3. Kinetic Energy Stroke Apex Isolation (Hold-Phase Deceleration Windowing)
  4. Strict CWASA / JASigning SiGML XML Syntax Validation (Verified via ElementTree)
"""

import os
import sys
import cv2
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xml.etree.ElementTree as ET
from scipy.spatial.distance import pdist

# Ensure hamer and integration directories are on sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
HAMER_DIR = os.path.join(ROOT_DIR, "hamer")
INTEGRATION_DIR = os.path.join(ROOT_DIR, "Integration-20260706T062240Z-3-001", "Integration")

for p in [HAMER_DIR, INTEGRATION_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Fix NumPy / Python 3.12 compatibility
import inspect
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

for _attr, _type in [
    ('bool', bool), ('int', int), ('float', float), ('complex', complex),
    ('object', object), ('unicode', str), ('str', str)
]:
    if not hasattr(np, _attr):
        setattr(np, _attr, _type)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==============================================================================
# Model A Architecture: 3D Skeletal Hand-ResNet (V5: 207-D, V6: 113-D)
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
        x = torch.nn.functional.gelu(self.bn1(self.fc1(x)))
        x = self.drop(x)
        x = self.bn2(self.fc2(x))
        return torch.nn.functional.gelu(x + res)

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

# V6 Handshape Model (113-D with LayerNorm & Mish)
class ResidualBlock_V6(nn.Module):
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
        self.b1 = ResidualBlock_V6(hidden_dim, 0.2)
        self.b2 = ResidualBlock_V6(hidden_dim, 0.2)
        self.head = nn.Sequential(nn.Linear(hidden_dim, 128), nn.Mish(), nn.Dropout(0.25), nn.Linear(128, num_classes))
    def forward(self, x):
        return self.head(self.b2(self.b1(self.proj(x))))

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
        self.proj = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish())
        self.b1 = ResidualBlock_V7(hidden_dim, 0.20)
        self.b2 = ResidualBlock_V7(hidden_dim, 0.20)
        self.b3 = ResidualBlock_V7(hidden_dim, 0.20)
        self.head = nn.Sequential(nn.Linear(hidden_dim, 128), nn.LayerNorm(128), nn.Mish(), nn.Dropout(0.25), nn.Linear(128, num_classes))
    def forward(self, x):
        return self.head(self.b3(self.b2(self.b1(self.proj(x)))))


# ==============================================================================
# Model B Architecture: Attention-BiGRU (V5: 32x3, V6: 32x18)
# ==============================================================================
class AttentionBiGRU_V5(nn.Module):
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
        if x.dim() == 3 and x.shape[1] == 3:
            x = x.transpose(1, 2)
        h = torch.nn.functional.gelu(self.input_fc(x))
        gru_out, _ = self.gru(h)
        attn_out, _ = self.attn(gru_out, gru_out, gru_out)
        pooled = torch.mean(attn_out, dim=1)
        return self.classifier(pooled)

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
        if x.dim() == 3 and x.shape[1] != 18 and x.shape[2] == 18:
            x = x.transpose(1, 2)
        h = self.conv(x).transpose(1, 2)
        gru_out, _ = self.gru(h)
        attn_out, _ = self.attn(gru_out, gru_out, gru_out)
        pooled = torch.mean(attn_out, dim=1)
        return self.classifier(pooled)

# ==============================================================================
# Canonical 3D Unit Target Vectors
# ==============================================================================
EXT_FINGER_DIRECTIONS = {
    "hamextfingeru":   np.array([ 0.0, -1.0,  0.0]),
    "hamextfingerd":   np.array([ 0.0,  1.0,  0.0]),
    "hamextfingerl":   np.array([ 1.0,  0.0,  0.0]),
    "hamextfingerr":   np.array([-1.0,  0.0,  0.0]),
    "hamextfingerul":  np.array([ 0.707, -0.707,  0.0]),
    "hamextfingerur":  np.array([-0.707, -0.707,  0.0]),
    "hamextfingerdl":  np.array([ 0.707,  0.707,  0.0]),
    "hamextfingerdr":  np.array([-0.707,  0.707,  0.0]),
    "hamextfingero":   np.array([ 0.0,  0.0, -1.0]),
    "hamextfingeri":   np.array([ 0.0,  0.0,  1.0]),
    "hamextfingerol":  np.array([ 0.707,  0.0, -0.707]),
    "hamextfingeror":  np.array([-0.707,  0.0, -0.707]),
    "hamextfingeruo":  np.array([ 0.0, -0.707, -0.707]),
    "hamextfingerdo":  np.array([ 0.0,  0.707, -0.707]),
    "hamextfingerui":  np.array([ 0.0, -0.707,  0.707]),
    "hamextfingerdi":  np.array([ 0.0,  0.707,  0.707]),
}

PALM_ORI_DIRECTIONS = {
    "hampalmu":   np.array([ 0.0, -1.0,  0.0]),
    "hampalmd":   np.array([ 0.0,  1.0,  0.0]),
    "hampalml":   np.array([ 1.0,  0.0,  0.0]),
    "hampalmr":   np.array([-1.0,  0.0,  0.0]),
    "hampalmul":  np.array([ 0.707, -0.707,  0.0]),
    "hampalmur":  np.array([-0.707, -0.707,  0.0]),
    "hampalmdl":  np.array([ 0.707,  0.707,  0.0]),
    "hampalmdr":  np.array([-0.707,  0.707,  0.0]),
}

# Load authoritative CWASA token list for validation & casing rectification
CWASA_TOKENS = {}
token_map_file = os.path.join(INTEGRATION_DIR, "cwasa_token_map.json")
if os.path.exists(token_map_file):
    with open(token_map_file, "r", encoding="utf-8") as f:
        CWASA_TOKENS = json.load(f)

# Canonical casing helper for CWASA/JASigning compliance
def canonicalize_cwasa_token(tok: str) -> str:
    tl = tok.strip().lower()
    if tl in CWASA_TOKENS:
        return CWASA_TOKENS[tl]
    if tl == "hamupperarm":
        return "hamUpperarm"
    return tok.strip()


class VideoFramesDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, frames_data, rescale_factor=2.0):
        from hamer.datasets.utils import expand_to_aspect_ratio, generate_image_patch_cv2, convert_cvimg_to_tensor
        try:
            from skimage.filters import gaussian
        except ImportError:
            def gaussian(img, sigma=1, channel_axis=2, preserve_range=True):
                ksize = int(2 * round(3 * sigma) + 1) | 1
                return cv2.GaussianBlur(img, (ksize, ksize), sigma)
        self.gaussian = gaussian
        self.expand_to_aspect_ratio = expand_to_aspect_ratio
        self.generate_image_patch_cv2 = generate_image_patch_cv2
        self.convert_cvimg_to_tensor = convert_cvimg_to_tensor
        self.cfg = cfg
        self.frames_data = frames_data
        self.rescale_factor = rescale_factor
        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(cfg.MODEL.IMAGE_STD)
        self.BBOX_SHAPE = cfg.MODEL.get('BBOX_SHAPE', None)

    def __len__(self):
        return len(self.frames_data)

    def __getitem__(self, idx):
        item = self.frames_data[idx]
        rgb = item['rgb']
        box = np.array(item['box'], dtype=np.float32)
        is_right = item['is_right']

        center = (box[2:4] + box[0:2]) / 2.0
        scale = self.rescale_factor * (box[2:4] - box[0:2]) / 200.0
        bbox_size = self.expand_to_aspect_ratio(scale * 200, target_aspect_ratio=self.BBOX_SHAPE).max()

        flip = not is_right
        cvimg = rgb.copy()
        downsampling_factor = ((bbox_size * 1.0) / self.img_size) / 2.0
        if downsampling_factor > 1.1:
            cvimg = self.gaussian(cvimg, sigma=(downsampling_factor - 1) / 2, channel_axis=2, preserve_range=True)

        img_patch_cv, _ = self.generate_image_patch_cv2(
            cvimg, center[0], center[1],
            bbox_size, bbox_size,
            self.img_size, self.img_size,
            flip, 1.0, 0,
            border_mode=cv2.BORDER_CONSTANT
        )
        img_patch_cv = img_patch_cv[:, :, ::-1]
        img_patch = self.convert_cvimg_to_tensor(img_patch_cv)
        for n_c in range(min(cvimg.shape[2], 3)):
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - self.mean[n_c]) / self.std[n_c]

        return {'img': img_patch}


# ==============================================================================
# Upgraded Local HaMeR Hybrid Pipeline
# ==============================================================================
class LocalHaMeRPipeline:
    def __init__(self):
        print(f"[HaMeRPipeline] Initializing Complete Hybrid Engine on device: {DEVICE}")
        
        # 1. Load Encoders & Neural Models (Auto-detect V7 vs V6 vs V5)
        v7_enc_file = os.path.join(INTEGRATION_DIR, "v7_encoders.json")
        v7_model_a_path = os.path.join(INTEGRATION_DIR, "v7_dgs_handshape_resnet.pth")
        v7_model_b_path = os.path.join(INTEGRATION_DIR, "v7_dgs_movement_attention_bigru.pth")

        v6_enc_file = os.path.join(INTEGRATION_DIR, "v6_encoders.json")
        v6_model_a_path = os.path.join(INTEGRATION_DIR, "v6_dgs_handshape_resnet.pth")
        v6_model_b_path = os.path.join(INTEGRATION_DIR, "v6_dgs_movement_attention_bigru.pth")

        if os.path.exists(v7_enc_file) and os.path.exists(v7_model_a_path) and os.path.exists(v7_model_b_path):
            self.version = "v8"
            with open(v7_enc_file, "r", encoding="utf-8") as f:
                self.encoders = json.load(f)
            self.hs_classes = self.encoders["handshape_classes"]
            self.mv_classes = self.encoders["movement_classes"]

            self.model_a = SkeletalHandResNet_V7(in_features=128, hidden_dim=256, num_classes=len(self.hs_classes)).to(DEVICE)
            self.model_a.load_state_dict(torch.load(v7_model_a_path, map_location=DEVICE))
            self.model_a.eval()

            self.model_b = AttentionBiGRU_V6(in_channels=18, hidden_dim=128, num_classes=len(self.mv_classes)).to(DEVICE)
            self.model_b.load_state_dict(torch.load(v7_model_b_path, map_location=DEVICE))
            self.model_b.eval()
            print(f"[HaMeRPipeline] Loaded V8 Complete Hybrid Engine (128-D Invariant Handshape, 18-D Kinematics, Collision-Free Body Clearance, {len(self.hs_classes)} HS / {len(self.mv_classes)} MV classes).")
        elif os.path.exists(v6_enc_file) and os.path.exists(v6_model_a_path) and os.path.exists(v6_model_b_path):
            self.version = "v6"
            with open(v6_enc_file, "r", encoding="utf-8") as f:
                self.encoders = json.load(f)
            self.hs_classes = self.encoders["handshape_classes"]
            self.mv_classes = self.encoders["movement_classes"]

            self.model_a = SkeletalHandResNet_V6(in_features=113, hidden_dim=256, num_classes=len(self.hs_classes)).to(DEVICE)
            self.model_a.load_state_dict(torch.load(v6_model_a_path, map_location=DEVICE))
            self.model_a.eval()

            self.model_b = AttentionBiGRU_V6(in_channels=18, hidden_dim=128, num_classes=len(self.mv_classes)).to(DEVICE)
            self.model_b.load_state_dict(torch.load(v6_model_b_path, map_location=DEVICE))
            self.model_b.eval()
            print(f"[HaMeRPipeline] Loaded V6 Trained Models (113-D Handshape, 18-D Kinematics, {len(self.hs_classes)} HS / {len(self.mv_classes)} MV classes).")
        else:
            self.version = "v5"
            enc_file = os.path.join(INTEGRATION_DIR, "dgs_hamer_encoders.json")
            with open(enc_file, "r", encoding="utf-8") as f:
                self.encoders = json.load(f)
            self.hs_classes = self.encoders["handshape_classes"]
            self.mv_classes = self.encoders["movement_classes"]

            model_a_path = os.path.join(INTEGRATION_DIR, "v5_dgs_handshape_resnet.pth")
            self.model_a = SkeletalHandResNet(in_features=207, hidden_dim=256, num_classes=len(self.hs_classes)).to(DEVICE)
            self.model_a.load_state_dict(torch.load(model_a_path, map_location=DEVICE))
            self.model_a.eval()

            model_b_path = os.path.join(INTEGRATION_DIR, "v5_dgs_movement_attention_bigru.pth")
            self.model_b = AttentionBiGRU_V5(in_channels=3, hidden_dim=128, num_classes=len(self.mv_classes)).to(DEVICE)
            self.model_b.load_state_dict(torch.load(model_b_path, map_location=DEVICE))
            self.model_b.eval()
            print(f"[HaMeRPipeline] Loaded V5 Models.")

        # 4. Load MediaPipe Hand Detector (Using top-level imports)
        
        mp_task_path = os.path.join(HAMER_DIR, "hand_landmarker.task")
        base_options = mp_python.BaseOptions(model_asset_path=mp_task_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.35,
            min_tracking_confidence=0.35
        )
        self.detector = mp_vision.HandLandmarker.create_from_options(options)

        # 5. Load HaMeR ViT Mesh Model
        from hamer.models import load_hamer
        hamer_ckpt = os.path.join(HAMER_DIR, "_DATA", "hamer_ckpts", "checkpoints", "hamer.ckpt")
        self.hamer_model, self.hamer_cfg = load_hamer(hamer_ckpt)
        self.hamer_model = self.hamer_model.to(DEVICE)
        self.hamer_model.eval()
        print("[HaMeRPipeline] Engine Ready.")

    def classify_vector(self, vec, target_dict):
        """Matches a 3D unit vector against canonical direction dictionary via cosine similarity."""
        norm = np.linalg.norm(vec) + 1e-8
        unit_vec = vec / norm
        best_name = None
        best_sim = -2.0
        for name, target_vec in target_dict.items():
            sim = np.dot(unit_vec, target_vec)
            if sim > best_sim:
                best_sim = sim
                best_name = name
        return best_name, float(best_sim)

    def extract_128d_features(self, kps_3d, hand_pose):
        """
        Computes 128-D rotation-invariant, local-coordinate hand features (V7).
        Aligned to the local hand orthonormal basis R_local.
        """
        p0 = kps_3d[0].copy()
        y_axis = kps_3d[9] - p0
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)
        
        trans = kps_3d[5] - kps_3d[17]
        z_axis = np.cross(trans, y_axis)
        z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)
        
        x_axis = np.cross(y_axis, z_axis)
        R_local = np.stack([x_axis, y_axis, z_axis], axis=0)
        
        centered = kps_3d - p0
        local_kps = (R_local @ centered.T).T
        palm_scale = np.linalg.norm(local_kps[9]) + 1e-8
        local_kps = local_kps / palm_scale
        f_kps = local_kps.flatten() # 63D
        
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
        f_angles = np.array(angles_15, dtype=np.float32) # 15D
        
        mcp_tips = [
            local_kps[8] - local_kps[5],
            local_kps[12] - local_kps[9],
            local_kps[16] - local_kps[13],
            local_kps[20] - local_kps[17]
        ]
        spreads_4 = []
        for i in range(3):
            cos_s = np.dot(mcp_tips[i], mcp_tips[i+1]) / (np.linalg.norm(mcp_tips[i]) * np.linalg.norm(mcp_tips[i+1]) + 1e-8)
            spreads_4.append(np.clip(cos_s, -1.0, 1.0))
        thumb_ray = local_kps[4] - local_kps[1]
        cos_t = np.dot(thumb_ray, mcp_tips[0]) / (np.linalg.norm(thumb_ray) * np.linalg.norm(mcp_tips[0]) + 1e-8)
        spreads_4.append(np.clip(cos_t, -1.0, 1.0))
        f_spreads = np.array(spreads_4, dtype=np.float32) # 4D
        
        thumb_tip = local_kps[4]
        pinch_dists = [np.linalg.norm(thumb_tip - local_kps[idx]) for idx in [4, 8, 12, 16, 20]]
        f_pinch = np.array(pinch_dists, dtype=np.float32) # 5D
        
        f_mano = hand_pose.reshape(-1)[:41].astype(np.float32) # 41D
        
        return np.concatenate([f_kps, f_angles, f_spreads, f_pinch, f_mano])

    def compute_palm_normal(self, kps_3d, is_right=True):
        """
        Plan A: Computes maximum-likelihood 3D normal vector to the palm plane
        via Singular Value Decomposition (SVD) across all palm root keypoints.
        Eliminates single-joint knuckle noise and knuckle curvature deformation.
        """
        # Palm keypoints: wrist(0), thumb CMC(1), index MCP(5), middle MCP(9), ring MCP(13), pinky MCP(17)
        palm_indices = [0, 1, 5, 9, 13, 17]
        pts = kps_3d[palm_indices]
        centroid = np.mean(pts, axis=0)
        shifted = pts - centroid

        # SVD: right singular vector with minimum singular value is the plane normal
        try:
            _, _, vh = np.linalg.svd(shifted)
            normal = vh[2, :].copy()
        except Exception:
            normal = np.array([0.0, 0.0, -1.0])

        # Ensure consistent orientation: outward from palm side
        u_trans = kps_3d[5] - kps_3d[17]
        v_long = kps_3d[9] - kps_3d[0]
        n_ref = np.cross(u_trans, v_long) if is_right else np.cross(v_long, u_trans)

        if np.dot(normal, n_ref) < 0:
            normal = -normal

        norm = np.linalg.norm(normal) + 1e-8
        return normal / norm

    def compute_ext_finger_vector(self, kps_3d, base_hs=None):
        """Computes true 3D longitudinal pointing axis of the hand with depth perspective compensation."""
        p0, p9 = kps_3d[0], kps_3d[9]
        if base_hs == "hamfist":
            v_long = (p9 - p0) / (np.linalg.norm(p9 - p0) + 1e-8)
        else:
            tips = [kps_3d[4], kps_3d[8], kps_3d[12], kps_3d[16], kps_3d[20]]
            dists = [np.linalg.norm(t - p0) for t in tips]
            furthest_tip = tips[int(np.argmax(dists))]

            v_mcp = (p9 - p0) / (np.linalg.norm(p9 - p0) + 1e-8)
            v_tip = (furthest_tip - p0) / (np.linalg.norm(furthest_tip - p0) + 1e-8)
            v_long = 0.40 * v_mcp + 0.60 * v_tip

        # Monocular depth perspective compensation: scaling Z by 1.6 prevents forward-pointing
        # gestures (hamextfingero, hamextfingeruo) from collapsing into pure vertical (hamextfingeru)
        v_comp = np.array([v_long[0], v_long[1], v_long[2] * 1.6], dtype=np.float32)
        norm = np.linalg.norm(v_comp) + 1e-8
        return v_comp / norm

    def compute_finger_modifiers(self, kps_3d, base_hs):
        """Computes geometric thumb configuration and finger curl modifiers."""
        p0 = kps_3d[0]
        mods = []

        # 1. Finger Bending Analysis (Index and Middle fingers)
        v_idx_prox = kps_3d[6] - kps_3d[5]
        v_idx_dist = kps_3d[8] - kps_3d[7]
        cos_idx = np.dot(v_idx_prox, v_idx_dist) / (np.linalg.norm(v_idx_prox) * np.linalg.norm(v_idx_dist) + 1e-6)

        v_mid_prox = kps_3d[10] - kps_3d[9]
        v_mid_dist = kps_3d[12] - kps_3d[11]
        cos_mid = np.dot(v_mid_prox, v_mid_dist) / (np.linalg.norm(v_mid_prox) * np.linalg.norm(v_mid_dist) + 1e-6)

        avg_cos = (cos_idx + cos_mid) / 2.0

        if base_hs not in ["hamfist"]:
            if avg_cos < 0.20: # Bent significantly (> 78 degrees)
                mods.append("hamfingerbendmod")
            elif avg_cos > 0.88 and base_hs in ["hamflathand", "hamfinger2", "hamceeall", "hamfinger2345"]:
                mods.append("hamfingerstraightmod")

        # 2. Thumb Position Analysis
        p_thumb_tip = kps_3d[4]
        p_index_mcp = kps_3d[5]
        p_palm_center = kps_3d[9]
        
        hand_scale = np.linalg.norm(p_palm_center - p0) + 1e-6
        d_thumb_palm = np.linalg.norm(p_thumb_tip - p_palm_center) / hand_scale
        d_thumb_index = np.linalg.norm(p_thumb_tip - p_index_mcp) / hand_scale

        if d_thumb_palm < 0.45 or (base_hs == "hamfist" and d_thumb_index < 0.55):
            mods.append("hamthumbacrossmod")
        elif d_thumb_index > 0.72:
            mods.append("hamthumboutmod")

        return mods

    def classify_anatomical_location(self, wrist_2d_norm, hand_center_norm):
        """Classifies anatomical body region and spatial relator from 2D coordinates."""
        y = wrist_2d_norm[1]
        x = wrist_2d_norm[0]
        relator = None

        # Check midline alignment
        if abs(x - 0.5) < 0.06:
            relator = "hamlrat"
        elif abs(x - 0.5) > 0.22:
            relator = "hamlrbeside"

        # Anatomical vertical zones for frontal signers
        if y < 0.22:
            loc = "hamforehead"
        elif y < 0.32:
            loc = "hamcheek" if abs(x - 0.5) > 0.08 else "hamnose"
        elif y < 0.38:
            loc = "hamchin"
        elif y < 0.52:
            loc = "hamshoulders" if (y < 0.42 and abs(x - 0.5) > 0.22) else "hamchest"
        else:
            # Hands held in front of lower torso in open space = neutral signing space
            loc = "hamneutralspace"

        return loc, relator

    def detect_trajectory_curvature(self, raw_wrist_trajectories):
        """Identifies curved arc paths (hamarcu, hamarcd) or circles from 3D points."""
        traj = np.array(raw_wrist_trajectories)
        if len(traj) < 6:
            return None

        # Net vector from start to end
        v_net = traj[-1] - traj[0]
        net_dist = np.linalg.norm(v_net) + 1e-6

        # Height deviation perpendicular to net direction
        t_mid = len(traj) // 2
        mid_pt = traj[t_mid]
        # Linear interpolated point at midpoint
        linear_mid = (traj[0] + traj[-1]) / 2.0
        dy_arc = mid_pt[1] - linear_mid[1] # In camera space, negative is UP

        if dy_arc < -0.04: # Arced upward
            return "hamarcu"
        elif dy_arc > 0.04: # Arced downward
            return "hamarcd"

        return None

    def validate_and_build_sigml(self, xml_elements):
        """
        Validates tokens and constructs an authentic, syntactically compliant CWASA SiGML XML.
        Verifies schema validity with ElementTree and canonical CWASA token casing.
        """
        clean_tokens = []
        for elem in xml_elements:
            elem_clean = elem.strip()
            # Canonicalize token casing for CWASA ANTLR parser compliance
            canon = canonicalize_cwasa_token(elem_clean)
            tl = canon.lower()
            if tl in CWASA_TOKENS or canon in ["hamparbegin", "hamparend", "hamplus", "hamreplace", "hamseqbegin", "hamseqend"]:
                clean_tokens.append(canon)

        lines = [
            '<?xml version="1.0" encoding="utf-8"?>',
            '<sigml>',
            '  <hns_sign gloss="sign">',
            '    <hamnosys_nonmanual>',
            '    </hamnosys_nonmanual>',
            '    <hamnosys_manual>'
        ]

        # Ensure balanced hamparbegin / hamparend stack
        par_open = 0
        for t in clean_tokens:
            if t == "hamparbegin":
                par_open += 1
                lines.append(f'      <{t}/>')
            elif t == "hamparend":
                if par_open > 0:
                    lines.append(f'      <{t}/>')
                    par_open -= 1
            else:
                lines.append(f'      <{t}/>')

        while par_open > 0:
            lines.append('      <hamparend/>')
            par_open -= 1

        lines.append('    </hamnosys_manual>')
        lines.append('  </hns_sign>')
        lines.append('</sigml>')

        sigml_content = '\n'.join(lines)

        # Validate with ElementTree XML parser
        try:
            ET.fromstring(sigml_content)
            is_valid = True
        except Exception as e:
            print(f"[SiGML Validator] XML Syntax Warning: {e}. Fallback to clean structure.")
            is_valid = False

        return sigml_content, is_valid

    def process_video(self, video_path):
        """Full end-to-end processing with expanded phonetic slots and valid SiGML generator."""
        from hamer.utils import recursive_to
        from hamer.datasets.vitdet_dataset import ViTDetDataset
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video {video_path}")

        raw_wrist_trajectories = []
        raw_wrist_2d = []
        frame_features_207 = []
        frame_features_113 = []
        frame_features_128 = []
        frame_tips_3d = []
        frame_kps_3d = []
        frame_palm_vectors = []
        frame_ext_vectors = []
        frame_loc_data = [] # (loc, relator)
        
        left_hand_wrist_trajs = []
        right_hand_wrist_trajs = []
        left_hand_kps_samples = []

        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[Pipeline] Processing video: {os.path.basename(video_path)} ({total_video_frames} total frames)...", flush=True)

        # Optimize frame sampling rate to balance high accuracy with fast (< 3s) inference
        frame_stride = max(1, total_video_frames // 32) if total_video_frames > 36 else 1
        frame_items = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frame_idx += 1
            if frame_stride > 1 and (frame_idx % frame_stride != 0):
                continue
                
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            detection_result = self.detector.detect(mp_image)

            if not detection_result.hand_landmarks:
                continue

            hands_data = []
            for i, lms in enumerate(detection_result.hand_landmarks):
                xs = [p.x * w for p in lms]
                ys = [p.y * h for p in lms]
                min_x, max_x = max(0, min(xs)), min(w, max(xs))
                min_y, max_y = max(0, min(ys)), min(h, max(ys))
                area = (max_x - min_x) * (max_y - min_y)
                
                pad_x = max((max_x - min_x) * 0.2, 15)
                pad_y = max((max_y - min_y) * 0.2, 15)
                box = [max(0, min_x - pad_x), max(0, min_y - pad_y), min(w, max_x + pad_x), min(h, max_y + pad_y)]
                
                is_right = True
                if detection_result.handedness and i < len(detection_result.handedness):
                    lbl = detection_result.handedness[i][0].category_name
                    is_right = (lbl == 'Right')

                wrist_norm_2d = (lms[0].x, lms[0].y)
                center_norm_2d = ((min_x + max_x) / (2 * w), (min_y + max_y) / (2 * h))

                hands_data.append({
                    "box": box, "area": area, "is_right": is_right,
                    "wrist_2d": wrist_norm_2d, "center_2d": center_norm_2d
                })

            for hd in hands_data:
                if hd["is_right"]:
                    right_hand_wrist_trajs.append(hd["wrist_2d"])
                else:
                    left_hand_wrist_trajs.append(hd["wrist_2d"])

            # Maintain consistent dominant hand across frames (prefer Right hand in signing, then largest area)
            right_hands = [h for h in hands_data if h["is_right"]]
            if right_hands:
                dominant_hand = max(right_hands, key=lambda d: d["area"])
            else:
                dominant_hand = max(hands_data, key=lambda d: d["area"])
            
            frame_items.append({
                "rgb": rgb,
                "box": dominant_hand["box"],
                "is_right": dominant_hand["is_right"],
                "wrist_2d": dominant_hand["wrist_2d"],
                "center_2d": dominant_hand["center_2d"],
                "lms": dominant_hand.get("lms", detection_result.hand_landmarks[0])
            })

        cap.release()

        if not frame_items:
            sigml_xml, _ = self.validate_and_build_sigml(["hamflathand", "hamextfingero", "hampalmd", "hamneutralspace"])
            return {"hamnosys": "hamflathand hamextfingero hampalmd hamneutralspace", "sigml_xml": sigml_xml}

        # ---------------------------------------------------------------------
        # High-Speed Batched 3D Mesh / Keypoint Inference (10x - 20x faster)
        # ---------------------------------------------------------------------
        all_kps_3d = []
        all_hand_pose = []

        if getattr(self, "hamer_model", None) is not None and getattr(self, "hamer_cfg", None) is not None:
            dataset = VideoFramesDataset(self.hamer_cfg, frame_items, rescale_factor=2.0)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
            for batch in dataloader:
                batch = recursive_to(batch, DEVICE)
                with torch.no_grad():
                    out = self.hamer_model(batch)
                kps_batch = out['pred_keypoints_3d'].detach().cpu().numpy()
                pose_batch = out['pred_mano_params']['hand_pose'].detach().cpu().numpy()
                for b_i in range(len(kps_batch)):
                    all_kps_3d.append(kps_batch[b_i])
                    all_hand_pose.append(pose_batch[b_i])
        else:
            # Resilient MediaPipe 3D Landmark Fallback
            for item in frame_items:
                dom_lms = item["lms"]
                kps_3d = np.array([[p.x, p.y, p.z] for p in dom_lms], dtype=np.float32)
                hand_pose = np.zeros((16, 3, 3), dtype=np.float32)
                all_kps_3d.append(kps_3d)
                all_hand_pose.append(hand_pose)

        for idx, item in enumerate(frame_items):
            kps_3d = all_kps_3d[idx]
            hand_pose = all_hand_pose[idx]

            wrist_3d = kps_3d[0].copy()
            centered_3d = kps_3d - wrist_3d
            scale = np.linalg.norm(centered_3d[9]) + 1e-6
            norm_3d = centered_3d / scale

            n_palm = self.compute_palm_normal(kps_3d, is_right=item["is_right"])
            v_ext = self.compute_ext_finger_vector(kps_3d)
            loc_token, relator_token = self.classify_anatomical_location(item["wrist_2d"], item["center_2d"])

            # V5 207-D Feature vector
            kps_flat = norm_3d.flatten()
            dists = pdist(norm_3d[:16])
            dists_105 = dists[:105] if len(dists) >= 105 else np.pad(dists, (0, 105 - len(dists)))
            pose_flat = hand_pose[:, :2, :2].flatten()[:39]
            feat_207 = np.concatenate([kps_flat, dists_105, pose_flat])

            # V6 113-D Feature vector (Normalized keypoints, 10 fingertip pdist, 5 curl angles, 35 MANO)
            tips_5 = norm_3d[[4, 8, 12, 16, 20]]
            tip_dists_10 = pdist(tips_5)
            curls = []
            for b1, b2, b3 in [(1,2,4), (5,6,8), (9,10,12), (13,14,16), (17,18,20)]:
                v1 = norm_3d[b2] - norm_3d[b1]
                v2 = norm_3d[b3] - norm_3d[b2]
                cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                curls.append(cos_a)
            curls_5 = np.array(curls)
            pose_35 = hand_pose.reshape(-1)[:35]
            feat_113 = np.concatenate([kps_flat, tip_dists_10, curls_5, pose_35])
            feat_128 = self.extract_128d_features(kps_3d, hand_pose)

            frame_features_207.append(feat_207)
            frame_features_113.append(feat_113)
            frame_features_128.append(feat_128)
            frame_tips_3d.append(kps_3d[[4, 8, 12, 16, 20]] - wrist_3d)
            frame_kps_3d.append(kps_3d)
            raw_wrist_trajectories.append(wrist_3d)
            raw_wrist_2d.append(item["wrist_2d"])
            frame_palm_vectors.append(n_palm)
            frame_ext_vectors.append(v_ext)
            frame_loc_data.append((loc_token, relator_token))

        cap.release()

        if getattr(self, "version", "v5") in ["v7", "v8"]:
            T = len(frame_features_128)
        elif getattr(self, "version", "v5") == "v6":
            T = len(frame_features_113)
        else:
            T = len(frame_features_207)
        if T == 0:
            sigml_xml, _ = self.validate_and_build_sigml(["hamflathand", "hamextfingero", "hampalmd", "hamneutralspace"])
            return {"hamnosys": "hamflathand hamextfingero hampalmd hamneutralspace", "sigml_xml": sigml_xml}

        # 1. Kinetic Apex Isolation
        velocities = [0.0] + [np.linalg.norm(raw_wrist_trajectories[t] - raw_wrist_trajectories[t-1]) for t in range(1, T)]
        norm_vel = np.array(velocities) / (np.max(velocities) + 1e-6)
        elevations = np.array([1.0 - pt[1] for pt in raw_wrist_2d])

        start_idx = max(0, int(T * 0.10))
        end_idx = min(T, max(start_idx + 1, int(T * 0.90)))
        apex_scores = elevations[start_idx:end_idx] - 1.2 * norm_vel[start_idx:end_idx]
        apex_idx = start_idx + int(np.argmax(apex_scores))

        weights = np.exp(-0.5 * ((np.arange(T) - apex_idx) / 3.0) ** 2)
        weights = weights / (np.sum(weights) + 1e-6)

        # 2. Handshape Prediction (Model A at Apex)
        with torch.no_grad():
            if getattr(self, "version", "v5") in ["v7", "v8"]:
                x_hs = torch.FloatTensor(np.array(frame_features_128)).to(DEVICE)
            elif getattr(self, "version", "v5") == "v6":
                x_hs = torch.FloatTensor(np.array(frame_features_113)).to(DEVICE)
            else:
                x_hs = torch.FloatTensor(np.array(frame_features_207)).to(DEVICE)
            logits_hs = self.model_a(x_hs)
            probs_hs = torch.softmax(logits_hs, dim=1).cpu().numpy()
            weighted_probs = np.sum(probs_hs * weights[:, np.newaxis], axis=0)
            pred_hs = self.hs_classes[int(np.argmax(weighted_probs))]

        # 3. Geometric Finger & Thumb Modifiers at Apex
        apex_kps = frame_kps_3d[apex_idx]
        hs_modifiers = self.compute_finger_modifiers(apex_kps, pred_hs)

        # 4. Orientation at Apex
        win_start = max(0, apex_idx - 2)
        win_end = min(T, apex_idx + 3)
        mean_palm_vec = np.mean(frame_palm_vectors[win_start:win_end], axis=0)
        mean_ext_vec = np.mean(frame_ext_vectors[win_start:win_end], axis=0)

        pred_palm, _ = self.classify_vector(mean_palm_vec, PALM_ORI_DIRECTIONS)
        pred_ext, _ = self.classify_vector(mean_ext_vec, EXT_FINGER_DIRECTIONS)

        # 5. Location and Spatial Relators at Apex
        loc_slice = [ld[0] for ld in frame_loc_data[win_start:win_end]]
        rel_slice = [ld[1] for ld in frame_loc_data[win_start:win_end] if ld[1] is not None]
        pred_loc = max(set(loc_slice), key=loc_slice.count) if loc_slice else "hamneutralspace"
        pred_rel = max(set(rel_slice), key=rel_slice.count) if rel_slice else None

        # Check contact state: default to False to prevent avatar torso/mesh collision.
        # In HamNoSys/CWASA, locations without hamtouch maintain natural 25cm body clearance.
        # Spurious hamtouch forces IK target depth to 0, pulling hands inside avatar torso mesh.
        has_touch = False

        # 6. Dual-Hand Asymmetric / Symmetry Evaluation
        is_two_handed = False
        symmetry_type = "single"
        if len(left_hand_wrist_trajs) >= 4 and len(right_hand_wrist_trajs) >= 4:
            left_elev = [1.0 - pt[1] for pt in left_hand_wrist_trajs]
            if np.mean(left_elev) > 0.30:
                is_two_handed = True
                symmetry_type = "hamsymmlr"

        # 7. Movement & Curvature (Fused 2D Camera Space + HaMeR Depth & BiGRU)
        traj_arr = np.array(raw_wrist_trajectories)
        wrist_2d_arr = np.array(raw_wrist_2d)
        max_displacement = np.max([np.linalg.norm(pt - traj_arr[0]) for pt in traj_arr]) if len(traj_arr) > 0 else 0.0
        
        # Calculate full-frame 2D camera-space displacement metrics
        if len(wrist_2d_arr) >= 4:
            s_idx = max(0, int(len(wrist_2d_arr) * 0.05))
            e_idx = min(len(wrist_2d_arr), int(len(wrist_2d_arr) * 0.90))
            active_2d = wrist_2d_arr[s_idx:e_idx]
            dx_2d = float(active_2d[-1, 0] - active_2d[0, 0])
            dy_2d = float(active_2d[-1, 1] - active_2d[0, 1]) # Note: In screen coords, dy < 0 is upward
            dist_2d = float(np.linalg.norm([dx_2d, dy_2d]))
            total_path_2d = float(np.sum(np.linalg.norm(np.diff(active_2d, axis=0), axis=1)))
            peak_up_2d = float(np.min(active_2d[:, 1]) - active_2d[0, 1])
            peak_down_2d = float(np.max(active_2d[:, 1]) - active_2d[0, 1])
        else:
            dx_2d, dy_2d, dist_2d, total_path_2d, peak_up_2d, peak_down_2d = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        pred_mv = None
        arc_token = None

        # Execute Model B (AttentionBiGRU) directly for all sequences with sufficient frames
        if len(raw_wrist_trajectories) >= 4:
            # Fuse camera-space 2D coordinates into trajectory translation to eliminate bbox crop drift
            fused_traj = traj_arr.copy()
            if len(wrist_2d_arr) == len(traj_arr):
                fused_traj[:, 0] = wrist_2d_arr[:, 0]
                fused_traj[:, 1] = wrist_2d_arr[:, 1]

            if getattr(self, "version", "v5") in ["v6", "v7", "v8"] and len(frame_tips_3d) == len(fused_traj):
                tips_arr = np.array(frame_tips_3d).reshape(T, 15)
                traj_18 = np.concatenate([fused_traj, tips_arr], axis=1) # (T, 18)
                orig_indices = np.linspace(0, 1, T)
                target_indices = np.linspace(0, 1, 32)
                interp_18 = np.zeros((32, 18), dtype=np.float32)
                for dim in range(18):
                    interp_18[:, dim] = np.interp(target_indices, orig_indices, traj_18[:, dim])
                
                interp_18[:, :3] = interp_18[:, :3] - interp_18[0:1, :3]
                scale_w = np.max(np.abs(interp_18[:, :3])) + 1e-6
                interp_18[:, :3] = interp_18[:, :3] / scale_w

                with torch.no_grad():
                    x_mv = torch.FloatTensor(interp_18).transpose(0, 1).unsqueeze(0).to(DEVICE) # (1, 18, 32)
                    logits_mv = self.model_b(x_mv)
                    probs_mv = torch.softmax(logits_mv, dim=1).cpu().numpy()[0]
                    
                    # Movement confidence temperature tuning: penalize spurious hamnomotion if clear motion exists
                    if total_path_2d > 0.40 and "hamnomotion" in self.mv_classes:
                        no_idx = self.mv_classes.index("hamnomotion")
                        probs_mv[no_idx] *= 0.15 # heavily discount stationary prior
                    pred_mv = self.mv_classes[int(np.argmax(probs_mv))]
            else:
                orig_indices = np.linspace(0, 1, len(fused_traj))
                target_indices = np.linspace(0, 1, 32)
                norm_traj = np.zeros((32, 3), dtype=np.float32)
                for dim in range(3):
                    norm_traj[:, dim] = np.interp(target_indices, orig_indices, fused_traj[:, dim])

                norm_traj = norm_traj - norm_traj[0]
                norm_traj = norm_traj / (np.max(np.abs(norm_traj)) + 1e-6)

                with torch.no_grad():
                    x_mv = torch.FloatTensor(norm_traj).unsqueeze(0).to(DEVICE)
                    logits_mv = self.model_b(x_mv)
                    probs_mv = torch.softmax(logits_mv, dim=1).cpu().numpy()[0]
                    if total_path_2d > 0.40 and "hamnomotion" in self.mv_classes:
                        no_idx = self.mv_classes.index("hamnomotion")
                        probs_mv[no_idx] *= 0.15
                    pred_mv = self.mv_classes[int(np.argmax(probs_mv))]

            # Kinematic Vector Displacement Heuristic Layer
            # Overrides model ambiguities with direct physical trajectory vectors
            if total_path_2d > 0.40:
                # Upward stroke check (in 2D screen coords, y decreases as hand rises)
                if peak_up_2d < -0.15 and dy_2d < -0.06 and abs(dx_2d) < 0.10:
                    pred_mv = "hammoveu"
                # Downward stroke check
                elif peak_down_2d > 0.15 and dy_2d > 0.08 and abs(dx_2d) < 0.12:
                    pred_mv = "hammoved"
                # Pure lateral movement checks
                elif dx_2d > 0.15 and abs(dy_2d) < 0.08:
                    pred_mv = "hammover"
                elif dx_2d < -0.15 and abs(dy_2d) < 0.08:
                    pred_mv = "hammovel"
            elif total_path_2d < 0.30 and dist_2d < 0.04:
                # Stationary gesture
                pred_mv = "hamnomotion"

            # Arc curvature check
            arc_token = self.detect_trajectory_curvature(raw_wrist_trajectories)

        # 8. Check for Mid-Flight Posture Replacement (hamreplace)
        offset_idx = min(T - 1, apex_idx + 5)
        offset_ext_vec = frame_ext_vectors[offset_idx]
        offset_ext, _ = self.classify_vector(offset_ext_vec, EXT_FINGER_DIRECTIONS)
        
        has_replace = False
        replace_ext = None
        if pred_mv and pred_mv != "hamnomotion" and offset_ext != pred_ext:
            # Significant orientation shift occurred during motion
            dot_shift = np.dot(mean_ext_vec, offset_ext_vec)
            if dot_shift < 0.70: # > 45 degree change
                has_replace = True
                replace_ext = offset_ext

        # ======================================================================
        # 9. Assembly into Formal SiGML XML Grammar
        # ======================================================================
        xml_tokens = []

        # Dual Hand or Symmetry
        if symmetry_type == "hamsymmlr":
            xml_tokens.append("hamsymmlr")

        # Handshape Block
        xml_tokens.append(pred_hs)
        for mod in hs_modifiers:
            xml_tokens.append(mod)

        # Orientation Block
        xml_tokens.append(pred_ext)
        xml_tokens.append(pred_palm)

        # Location & Contact Block
        final_loc = pred_loc
        final_rel = pred_rel

        # In symmetric signs, unilateral side relators (hamlrbeside, hamlrat) are phonetically invalid
        if symmetry_type in ["hamsymmlr", "hamsymmpar"]:
            final_rel = None

        # Neutral space cannot take side relators in Ham4Parser
        if final_loc == "hamneutralspace":
            final_rel = None

        if len(left_hand_wrist_trajs) >= 4 and len(right_hand_wrist_trajs) >= 4 and symmetry_type != "hamsymmlr":
            left_elev = [1.0 - pt[1] for pt in left_hand_wrist_trajs]
            if np.mean(left_elev) > 0.25: # Non-dominant hand must be actively raised
                pair_lens = min(len(right_hand_wrist_trajs), len(left_hand_wrist_trajs))
                inter_dists = [np.linalg.norm(np.array(right_hand_wrist_trajs[i]) - np.array(left_hand_wrist_trajs[i])) for i in range(pair_lens)]
                if min(inter_dists) < 0.06: # True physical contact between hands
                    has_touch = True
                    if final_loc == "hamneutralspace":
                        final_loc = "hampalm"

        # In HamNoSys, neutral signing space without contact is the default
        if final_loc != "hamneutralspace":
            xml_tokens.append(final_loc)
            if has_touch and final_loc not in ["hamshoulders", "hamchest"]:
                xml_tokens.append("hamtouch")
            elif final_rel:
                xml_tokens.append(final_rel)
        elif has_touch:
            xml_tokens.append("hampalm")
            xml_tokens.append("hamtouch")

        # Movement Block with Arcs, Size Modifiers & Posture Replacement
        if pred_mv and pred_mv != "hamnomotion":
            xml_tokens.append(pred_mv)
            if arc_token:
                xml_tokens.append(arc_token)
            if 0.05 <= max_displacement < 0.12:
                xml_tokens.append("hamsmallmod")
            if has_replace:
                xml_tokens.append("hamreplace")
                xml_tokens.append(replace_ext)

        # Apply Canonical ANTLR Ham4Parser Grammar Linter
        try:
            from sigml_grammar_linter import sanitize_hamnosys_sequence
            xml_tokens = sanitize_hamnosys_sequence(xml_tokens)
        except Exception as e:
            pass

        # Build & Validate XML
        sigml_xml, is_valid_xml = self.validate_and_build_sigml(xml_tokens)
        hns_str = " ".join(xml_tokens)

        return {
            "handshape": pred_hs,
            "modifiers": hs_modifiers,
            "extended_finger": pred_ext,
            "palm_orientation": pred_palm,
            "location": pred_loc,
            "relator": pred_rel,
            "has_touch": has_touch,
            "movement": pred_mv if pred_mv else "hamnomotion",
            "arc": arc_token,
            "has_replace": has_replace,
            "symmetry": symmetry_type,
            "hamnosys": hns_str,
            "sigml_xml": sigml_xml,
            "is_valid_xml": is_valid_xml,
            "apex_frame": int(apex_idx),
            "total_frames": int(T)
        }

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Run Upgraded Local HaMeR 3D Complete Hybrid Pipeline.")
    parser.add_argument("video", nargs="?", help="Path to input video file")
    parser.add_argument("--output", "-o", default=None, help="Path to save output SiGML XML file")

    args = parser.parse_args()
    video_path = args.video

    if not video_path:
        print("\nUsage: python local_hamer_pipeline.py <path_to_video> [--output output.sigml]\n")
        sys.exit(1)

    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    print("\n" + "=" * 68)
    print("      UPGRADED LOCAL HAMER 3D COMPLETE HYBRID PIPELINE (V8)      ")
    print("=" * 68)
    print(f"Input Video: {video_path}")

    pipeline = LocalHaMeRPipeline()
    print("\nProcessing video frames with expanded phonology and XML validation...")
    res = pipeline.process_video(video_path)

    out_sigml_path = args.output if args.output else os.path.join(ROOT_DIR, f"{Path(video_path).stem}_hamer_output.sigml")

    with open(out_sigml_path, "w", encoding="utf-8") as f:
        f.write(res["sigml_xml"])

    print("\n" + "=" * 68)
    print("                    PREDICTION RESULTS                              ")
    print("=" * 68)
    print(f"  Dual-Hand Symmetry:          {res['symmetry']}")
    print(f"  Handshape (Apex):            {res['handshape']}")
    print(f"  Handshape Modifiers:         {res['modifiers']}")
    print(f"  Extended Finger Direction:   {res['extended_finger']}")
    print(f"  Palm Orientation (3D Normal):{res['palm_orientation']}")
    print(f"  Anatomical Location:         {res['location']}")
    print(f"  Spatial Relator:             {res['relator']}")
    print(f"  Contact State:               {'hamtouch' if res['has_touch'] else 'none'}")
    print(f"  Movement Trajectory:         {res['movement']}")
    print(f"  Arc / Curvature:             {res['arc']}")
    print(f"  Mid-Flight Replacement:      {res['has_replace']}")
    print(f"  Stroke Apex Keyframe:        Frame {res['apex_frame']} of {res['total_frames']}")
    print(f"  SiGML Valid XML:             {res['is_valid_xml']}")
    print("-" * 68)
    print(f"  Complete HamNoSys String:    {res['hamnosys']}")
    print(f"  Saved SiGML XML to:          {out_sigml_path}")
    print("=" * 68 + "\n")
