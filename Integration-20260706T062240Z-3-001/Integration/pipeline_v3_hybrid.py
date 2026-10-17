#!/usr/bin/env python3
"""
Phase 4: Unified Hybrid V3 Pipeline (DWPose + Neural Networks + Deterministic Geometry)
Combines:
  1. DWPose Whole-Body Landmark Extraction (DirectML GPU on RTX 3050)
  2. Kinetic Stroke Activity Windowing
  3. Model A: Spatial Handshape Classifier (Frame-Level MLP)
  4. Deterministic Geometric Modules (3D Orthonormal Basis, Voronoi Location, Contact, Symmetry)
  5. Model B: Movement Trajectory Classifier (1D-CNN)
  6. ANTLR CFG Slot Compiler & CWASA SiGML XML Generator

Reference: HYBRID_NN_ARCHITECTURE_AND_TRAINING_SPECIFICATION.md
"""

import os
import sys
import cv2
import json
import numpy as np
import torch
import xml.etree.ElementTree as ET
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))

# DirectML acceleration for RTX 3050
try:
    import rtmlib.tools.base as base
    base.RTMLIB_SETTINGS['onnxruntime']['dml'] = 'DmlExecutionProvider'
    base.RTMLIB_SETTINGS['onnxruntime']['cuda'] = 'DmlExecutionProvider'
    from rtmlib import Wholebody
    HAS_RTM = True
except Exception as e:
    print(f"[Warning] rtmlib Wholebody error: {e}")
    HAS_RTM = False

from train_isl_models import HandshapeMLP, Movement1DCNN

MODEL_A_PTH = os.path.join(_SCRIPT_DIR, "isl_handshape_mlp.pth")
MODEL_B_PTH = os.path.join(_SCRIPT_DIR, "isl_movement_cnn.pth")
ENCODERS_JSON = os.path.join(_SCRIPT_DIR, "isl_label_encoders.json")
CWASA_MAP_JSON = os.path.join(_SCRIPT_DIR, "cwasa_token_map.json")

# Upgraded V5 DGS Artifacts (Highest Priority)
V5_MODEL_B_PTH = os.path.join(_SCRIPT_DIR, "v5_dgs_movement_attention_bigru.pth")
V5_MODEL_A_PTH = os.path.join(_SCRIPT_DIR, "v5_dgs_handshape_resnet.pth")
V5_ENCODERS_JSON = os.path.join(_SCRIPT_DIR, "dgs_hamer_encoders.json")

# Upgraded V4 Artifacts
V4_MODEL_B_PTH = os.path.join(_SCRIPT_DIR, "v4_hamer_movement_attention_bigru.pth")
V4_ENCODERS_JSON = os.path.join(_SCRIPT_DIR, "isl_hamer_encoders.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================================
# Upgraded V5 Model B Architecture (Attention-BiGRU with MultiheadAttention)
# =====================================================================
class AttentionBiGRU_V5(torch.nn.Module):
    def __init__(self, in_channels=3, hidden_dim=128, num_classes=19, num_heads=4):
        super().__init__()
        self.input_fc = torch.nn.Linear(in_channels, hidden_dim)
        self.gru = torch.nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.attn = torch.nn.MultiheadAttention(embed_dim=hidden_dim * 2, num_heads=num_heads, batch_first=True)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.GELU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        if x.dim() == 3 and x.shape[1] == 3:
            x = x.transpose(1, 2)
        h = torch.nn.functional.gelu(self.input_fc(x))
        gru_out, _ = self.gru(h)
        attn_out, _ = self.attn(gru_out, gru_out, gru_out)
        pooled = torch.mean(attn_out, dim=1)
        return self.classifier(pooled)

# =====================================================================
# Upgraded V4 Model B Architecture (Attention-BiGRU)
# =====================================================================
class TemporalSelfAttention(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim // 2, 1)
        )
    def forward(self, x):
        scores = self.attn(x)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * x, dim=1)
        return context

class AttentionBiGRU(torch.nn.Module):
    def __init__(self, in_channels=3, hidden_size=64, num_classes=15):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(32),
            torch.nn.Mish()
        )
        self.bigru = torch.nn.GRU(
            input_size=32,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.25
        )
        self.attention = TemporalSelfAttention(hidden_size * 2)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 2, 64),
            torch.nn.LayerNorm(64),
            torch.nn.Mish(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, num_classes)
        )

    def forward(self, x):
        if x.shape[-1] == 3:
            x_conv = x.transpose(1, 2)
        else:
            x_conv = x
        feat = self.conv(x_conv).transpose(1, 2)
        gru_out, _ = self.bigru(feat)
        context = self.attention(gru_out)
        return self.classifier(context)

# =====================================================================
# Unit Vectors for 3D Orientation Matching (Section 8.1)
# =====================================================================

EXT_FINGER_DIRECTIONS = {
    "hamextfingeru":   np.array([ 0.0, -1.0,  0.0]),
    "hamextfingerd":   np.array([ 0.0,  1.0,  0.0]),
    "hamextfingerr":   np.array([ 1.0,  0.0,  0.0]),
    "hamextfingerl":   np.array([-1.0,  0.0,  0.0]),
    "hamextfingerur":  np.array([ 0.707, -0.707,  0.0]),
    "hamextfingerul":  np.array([-0.707, -0.707,  0.0]),
    "hamextfingerdr":  np.array([ 0.707,  0.707,  0.0]),
    "hamextfingerdl":  np.array([-0.707,  0.707,  0.0]),
    "hamextfingero":   np.array([ 0.0,  0.0, -1.0]),
    "hamextfingeri":   np.array([ 0.0,  0.0,  1.0]),
    "hamextfingerol":  np.array([-0.707,  0.0, -0.707]),
    "hamextfingeror":  np.array([ 0.707,  0.0, -0.707]),
    "hamextfingeruo":  np.array([ 0.0, -0.707, -0.707]),
    "hamextfingerdo":  np.array([ 0.0,  0.707, -0.707]),
    "hamextfingerui":  np.array([ 0.0, -0.707,  0.707]),
    "hamextfingerdi":  np.array([ 0.0,  0.707,  0.707]),
}

PALM_ORI_DIRECTIONS = {
    "hampalmu":   np.array([ 0.0, -1.0,  0.0]),
    "hampalmd":   np.array([ 0.0,  1.0,  0.0]),
    "hampalmr":   np.array([ 1.0,  0.0,  0.0]),
    "hampalml":   np.array([-1.0,  0.0,  0.0]),
    "hampalmur":  np.array([ 0.707, -0.707,  0.0]),
    "hampalmul":  np.array([-0.707, -0.707,  0.0]),
    "hampalmdr":  np.array([ 0.707,  0.707,  0.0]),
    "hampalmdl":  np.array([-0.707,  0.707,  0.0]),
    "hampalmo":   np.array([ 0.0,  0.0, -1.0]),
    "hampalmi":   np.array([ 0.0,  0.0,  1.0]),
}

# =====================================================================
# V3/V4/V5 Hybrid Pipeline Engine
# =====================================================================

class V3HybridPipeline:
    def __init__(self):
        print("[V3HybridPipeline] Initializing models and hardware acceleration...")
        
        # 1. Load label encoders (prefer V5 DGS if available, then V4 hamer, then legacy)
        if os.path.exists(V5_ENCODERS_JSON) and os.path.exists(V5_MODEL_B_PTH):
            enc_file = V5_ENCODERS_JSON
            self.active_version = "v5_dgs"
        elif os.path.exists(V4_ENCODERS_JSON):
            enc_file = V4_ENCODERS_JSON
            self.active_version = "v4_hamer"
        else:
            enc_file = ENCODERS_JSON
            self.active_version = "v3_legacy"
            
        with open(enc_file, "r") as f:
            self.encoders = json.load(f)
            
        self.hs_classes = self.encoders["handshape_classes"]
        self.mv_classes = self.encoders["movement_classes"]
        print(f"[V3HybridPipeline] Encoders loaded ({self.active_version}): {len(self.hs_classes)} handshape, {len(self.mv_classes)} movement classes")
        
        # 2. Load Model A (Handshape MLP)
        self.model_a = HandshapeMLP(in_features=63, num_classes=len(self.hs_classes)).to(DEVICE)
        self.model_a_loaded = False
        if os.path.exists(MODEL_A_PTH):
            try:
                self.model_a.load_state_dict(torch.load(MODEL_A_PTH, map_location=DEVICE))
                self.model_a.eval()
                self.model_a_loaded = True
                print(f"[V3HybridPipeline] Loaded Model A Handshape MLP from {MODEL_A_PTH}")
            except Exception as e:
                print(f"[V3HybridPipeline] Note on Model A: {e} (using geometric voting)")
        else:
            print(f"[Warning] Model A weights not found at {MODEL_A_PTH}")
            
        # 3. Load Model B (Prefer Upgraded V5 DGS Attention-BiGRU)
        if os.path.exists(V5_MODEL_B_PTH) and self.active_version == "v5_dgs":
            self.model_b = AttentionBiGRU_V5(in_channels=3, hidden_dim=128, num_classes=len(self.mv_classes)).to(DEVICE)
            self.model_b.load_state_dict(torch.load(V5_MODEL_B_PTH, map_location=DEVICE))
            self.model_b.eval()
            self.model_b_mode = "v5_dgs"
            print(f"[V3HybridPipeline] Loaded Upgraded V5 DGS Model B (Attention-BiGRU 19 classes) from {V5_MODEL_B_PTH}")
        elif os.path.exists(V4_MODEL_B_PTH):
            self.model_b = AttentionBiGRU(in_channels=3, hidden_size=64, num_classes=len(self.mv_classes)).to(DEVICE)
            self.model_b.load_state_dict(torch.load(V4_MODEL_B_PTH, map_location=DEVICE))
            self.model_b.eval()
            self.model_b_mode = "v4_hamer"
            print(f"[V3HybridPipeline] Loaded Upgraded V4 Model B (Attention-BiGRU) from {V4_MODEL_B_PTH}")
        elif os.path.exists(MODEL_B_PTH):
            self.model_b = Movement1DCNN(in_channels=3, num_classes=len(self.mv_classes)).to(DEVICE)
            self.model_b.load_state_dict(torch.load(MODEL_B_PTH, map_location=DEVICE))
            self.model_b.eval()
            self.model_b_mode = "legacy"
            print(f"[V3HybridPipeline] Loaded Model B Movement 1D-CNN from {MODEL_B_PTH}")
        else:
            self.model_b = None
            self.model_b_mode = "none"
            print(f"[Warning] Model B weights not found")
            
        # 4. Load DWPose Wholebody with DirectML
        if HAS_RTM:
            self.wholebody = Wholebody(to_openpose=False, backend='onnxruntime', device='dml')
            print("[V3HybridPipeline] Initialized DWPose Wholebody with DirectML GPU acceleration.")
        else:
            self.wholebody = None

    def extract_landmarks(self, video_path, frame_subsample=2):
        """Extract Whole-body landmarks across video frames using DWPose with bbox caching."""
        if not os.path.exists(video_path):
            alt_path = os.path.join(_REPO_ROOT, video_path)
            if os.path.exists(alt_path):
                video_path = alt_path
                
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames_kpts = []
        frames_scores = []
        
        frame_idx = 0
        cached_bbox = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
                
            if frame_idx % frame_subsample == 0:
                h, w = frame.shape[:2]
                
                # Person detection cache
                if cached_bbox is None or frame_idx % 30 == 0:
                    try:
                        cached_bbox = self.wholebody.det_model(frame)
                    except Exception:
                        cached_bbox = None
                        
                if cached_bbox is not None and len(cached_bbox) > 0:
                    kpts, scores = self.wholebody.pose_model(frame, bboxes=cached_bbox)
                else:
                    kpts, scores = self.wholebody(frame)
                    
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
        return frames_kpts, frames_scores, fps / frame_subsample

    def compute_active_stroke_window(self, wrist_pts, hip_y_pts, fps=30.0):
        """Kinetic stroke activity windowing (Stage 2)."""
        n_frames = len(wrist_pts)
        if n_frames < 6:
            return 0, max(0, n_frames - 1)
            
        delta_t = 1.0 / max(fps, 1.0)
        velocities = np.linalg.norm(np.diff(wrist_pts, axis=0), axis=1) / delta_t
        energies = velocities ** 2
        smoothed_e = gaussian_filter1d(energies, sigma=1.0, mode='nearest')
        
        elevated = [wrist_pts[i][1] < (hip_y_pts[i] + 0.05) for i in range(len(smoothed_e))]
        active_indices = [i for i, (e, elev) in enumerate(zip(smoothed_e, elevated)) if e > 0.0003 and elev]
        
        if len(active_indices) >= 5:
            return max(0, active_indices[0]), min(n_frames - 1, active_indices[-1] + 1)
        elev_indices = [i for i, elev in enumerate(elevated) if elev]
        if len(elev_indices) >= 5:
            return max(0, elev_indices[0]), min(n_frames - 1, elev_indices[-1] + 1)
        return int(0.15 * n_frames), max(1, int(0.85 * n_frames))

    def compute_3d_orientation(self, hand_21):
        """
        Calibrated Orthonormal 3D Orientation Calculator (Section 8.1)
        Matches Extension Vector and Palm Normal against HamNoSys 18-direction sphere.
        """
        p0 = hand_21[0]      # Wrist
        p4 = hand_21[4]      # Thumb tip
        p5 = hand_21[5]      # Index MCP
        p8 = hand_21[8]      # Index tip
        p9 = hand_21[9]      # Middle MCP
        p12 = hand_21[12]    # Middle tip
        p16 = hand_21[16]    # Ring tip
        p17 = hand_21[17]    # Pinky MCP
        p20 = hand_21[20]    # Pinky tip
        
        # 1. Lead extended fingertip (furthest from wrist)
        tips = [p8, p12, p16, p20]
        tip_dists = [np.linalg.norm(t - p0) for t in tips]
        lead_tip = tips[int(np.argmax(tip_dists))]
        
        palm_len = max(np.linalg.norm(p9 - p0), 1e-4)
        max_dist = max(tip_dists)
        ext_ratio = max_dist / palm_len
        
        v_palm = p9 - p0
        v_ext_tip = lead_tip - p0
        
        dx_palm, dy_palm = float(v_palm[0]), float(v_palm[1])
        dx_tip, dy_tip = float(v_ext_tip[0]), float(v_ext_tip[1])
        
        # Combined longitudinal hand axis
        v_combined = 0.40 * v_palm + 0.60 * v_ext_tip
        dx = float(v_combined[0])
        dy = float(v_combined[1])
        
        # 1. Strongly UP (both palm and tip point up on screen, dy < 0)
        if dy_palm < -0.030 and dy_tip < -0.020:
            if dx > 0.055:
                best_ext = "hamextfingerul"
            elif dx < -0.035:
                best_ext = "hamextfingerur"
            else:
                best_ext = "hamextfingeru"
        # 2. Strongly DOWN (tip points down on screen, dy > 0)
        elif dy > 0.035 and dy_tip > 0.025:
            if dx > 0.045:
                best_ext = "hamextfingerdl"
            else:
                best_ext = "hamextfingerd"
        # 3. Strongly across body to signer's LEFT (flat horizontal on screen)
        elif dx > 0.075 and abs(dy) < 0.025 and ext_ratio > 2.2:
            best_ext = "hamextfingerl"
        # 4. Outward / Forward projections (very common in ISL signs held at chest)
        elif dx > 0.045:
            best_ext = "hamextfingerol"
        else:
            best_ext = "hamextfingero"
                
        # 2. Palm Orientation
        # Across-palm width vector: from Pinky MCP to Index MCP
        v_width = p5 - p17
        wx, wy = float(v_width[0]), float(v_width[1])
        wx_signer = -wx
        
        thumb_rel = p4 - p0
        tx, ty = float(thumb_rel[0]), float(thumb_rel[1])
        tx_signer = -tx
        ty_signer = -ty
        
        if "u" in best_ext:
            if ty_signer < -0.02:
                best_palm = "hampalmd"
            elif tx_signer < -0.02:
                best_palm = "hampalml"
            elif ty_signer > 0.04:
                best_palm = "hampalmu"
            else:
                best_palm = "hampalml" if wx_signer < 0 else "hampalmd"
        elif "l" in best_ext:
            if ty_signer < -0.02:
                best_palm = "hampalmd"
            elif ty_signer > 0.02:
                best_palm = "hampalmu"
            else:
                best_palm = "hampalml"
        elif "o" in best_ext:
            if ty_signer < -0.02:
                best_palm = "hampalmd"
            elif tx_signer < -0.02:
                best_palm = "hampalml"
            elif ty_signer > 0.02:
                best_palm = "hampalmu"
            else:
                best_palm = "hampalmd"
        else:
            best_palm = "hampalmd" if ty_signer < 0 else "hampalml"
                
        return best_ext, best_palm

    def compute_anatomical_location(self, hand_center, kpts, body_scale):
        """
        Calibrated Anatomical Location Voronoi Engine (Section 8.2)
        Computes Euclidean distance to canonical upper body & face keypoints.
        """
        targets = {}
        if len(kpts) > 6:
            targets["hamchest"] = (kpts[5] + kpts[6]) / 2.0
            targets["hamshoulders"] = kpts[5] if hand_center[0] < 0.5 else kpts[6]
            targets["hamshouldertop"] = targets["hamshoulders"] - np.array([0, 0.06])
        if len(kpts) > 0:
            targets["hamnose"] = kpts[0]
            targets["hamforehead"] = kpts[0] - np.array([0, 0.08])
            targets["hamchin"] = kpts[0] + np.array([0, 0.06])
            targets["hamlips"] = kpts[0] + np.array([0, 0.04])
            targets["hameyes"] = (kpts[1] + kpts[2]) / 2.0 if len(kpts) > 2 else kpts[0]
        if len(kpts) > 12:
            targets["hamstomach"] = (kpts[11] + kpts[12]) / 2.0
            targets["hambelowstomach"] = targets["hamstomach"] + np.array([0, 0.08])
            
        thresholds = {
            "hamforehead": 0.35 * body_scale,
            "hameyes":     0.28 * body_scale,
            "hamnose":     0.25 * body_scale,
            "hamlips":     0.25 * body_scale,
            "hamchin":     0.28 * body_scale,
            "hamshoulders": 0.55 * body_scale,
            "hamshouldertop": 0.45 * body_scale,
            "hamchest":    0.85 * body_scale,
            "hamstomach":  0.65 * body_scale,
            "hambelowstomach": 0.55 * body_scale,
        }
        
        best_loc = "hamneutralspace"
        min_dist = float("inf")
        
        for loc, pt in targets.items():
            dist = np.linalg.norm(hand_center - pt)
            if dist < thresholds.get(loc, 0.45 * body_scale) and dist < min_dist:
                min_dist = dist
                best_loc = loc
                
        # If no head target matches and hand is in torso signing space, default to chest
        if best_loc == "hamneutralspace" and len(kpts) > 0:
            if hand_center[1] > kpts[0, 1] + 0.10 * body_scale:
                best_loc = "hamchest"
                
        # Check contact (Section 8.3)
        contact = None
        if best_loc != "hamneutralspace":
            if min_dist < 0.12 * body_scale:
                contact = "hamtouch"
            elif min_dist < 0.22 * body_scale:
                contact = "hamclose"
                
        return best_loc, contact

    def compute_dual_hand_symmetry(self, left_wrists, right_wrists, hip_y_pts):
        """Dual-Hand Symmetry Coordinator (Section 8.4)"""
        if len(left_wrists) < 5 or len(right_wrists) < 5:
            return "none"
            
        v_l = np.linalg.norm(np.diff(left_wrists, axis=0), axis=1)
        v_r = np.linalg.norm(np.diff(right_wrists, axis=0), axis=1)
        
        std_l = np.std(v_l)
        std_r = np.std(v_r)
        
        # Both hands elevated?
        mean_ly = np.mean([p[1] for p in left_wrists])
        mean_ry = np.mean([p[1] for p in right_wrists])
        mean_hip = np.mean(hip_y_pts)
        
        both_elevated = (mean_ly < mean_hip) and (mean_ry < mean_hip)
        
        if std_l > 1e-4 and std_r > 1e-4 and both_elevated:
            corr = np.corrcoef(v_l, v_r)[0, 1]
            if corr > 0.65:
                return "hamsymmlr"
            elif corr > 0.15:
                return "hamplus"
                
        return "none"

    def predict_handshape(self, stroke_hand_pts):
        """Model A Hybrid Inference: Spatial Handshape MLP + Frame-by-Frame Geometric Voting."""
        if not stroke_hand_pts or len(stroke_hand_pts) == 0:
            return "hamflathand", None, None
            
        frame_votes = []
        mlp_feats = []
        
        for h_21 in stroke_hand_pts:
            wrist = h_21[0]
            palm_len = max(np.linalg.norm(h_21[9] - wrist), 1e-4)
            norm_h = (h_21 - wrist) / palm_len
            dists = np.linalg.norm(norm_h, axis=1, keepdims=True)
            mlp_feats.append(np.hstack([norm_h, dists]).flatten())
            
            # Normalized distances from wrist
            d_thumb = np.linalg.norm(norm_h[4])
            d_idx = np.linalg.norm(norm_h[8])
            d_mid = np.linalg.norm(norm_h[12])
            d_ring = np.linalg.norm(norm_h[16])
            d_pky = np.linalg.norm(norm_h[20])
            d_pinch = np.linalg.norm(norm_h[4] - norm_h[8])
            
            # Tip to MCP fold metrics (< 0.38 means folded against palm)
            t2m_idx = np.linalg.norm(norm_h[8] - norm_h[5])
            t2m_mid = np.linalg.norm(norm_h[12] - norm_h[9])
            t2m_ring = np.linalg.norm(norm_h[16] - norm_h[13])
            t2m_pky = np.linalg.norm(norm_h[20] - norm_h[17])
            
            # Frame-level geometric vote
            # 1. Fist (all fingers curled against palm)
            if (t2m_idx < 0.38 and t2m_mid < 0.35 and t2m_ring < 0.38) or max(d_idx, d_mid, d_ring, d_pky) < 1.15:
                frame_votes.append("hamfist")
            # 2. Finger 2 (index extended, others curled)
            elif d_idx > 1.22 and d_mid < 1.05 and d_ring < 0.95:
                frame_votes.append("hamfinger2")
            # 3. Finger 23 (index + middle extended, ring + pinky curled)
            elif d_idx > 1.18 and d_mid > 1.18 and d_ring < 1.05 and d_pky < 0.95:
                frame_votes.append("hamfinger23")
            # 4. Cee-all (all fingers curved in open cup)
            elif 0.35 <= t2m_mid <= 0.65 and 0.95 <= d_mid <= 1.35 and d_pinch > 0.25:
                frame_votes.append("hamceeall")
            # 5. Cee-12 (index and thumb curved C)
            elif d_pinch < 0.32 and d_idx > 1.15 and d_mid < 0.95:
                frame_votes.append("hamcee12")
            # 6. Pinch 12 (index tip touches thumb tip)
            elif d_pinch < 0.22 and d_mid < 1.05:
                frame_votes.append("hampinch12")
            # 7. Extended 4-fingers: flat hand vs finger 2345
            elif min(d_idx, d_mid, d_ring, d_pky) > 1.15:
                spread = np.linalg.norm(norm_h[8] - norm_h[20])
                if spread > 0.70 and "hamfinger2345" in self.hs_classes:
                    frame_votes.append("hamfinger2345")
                else:
                    frame_votes.append("hamflathand")
                
        # Neural Model A prediction (if loaded)
        if getattr(self, 'model_a_loaded', False):
            with torch.no_grad():
                tensor_x = torch.FloatTensor(np.array(mlp_feats, dtype=np.float32)).to(DEVICE)
                logits = self.model_a(tensor_x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                avg_probs = np.mean(probs, axis=0)
        else:
            avg_probs = np.ones(len(self.hs_classes), dtype=np.float32) / len(self.hs_classes)
            
        prior = np.ones(len(self.hs_classes), dtype=np.float32)
        if len(frame_votes) > 0:
            from collections import Counter
            counts = Counter(frame_votes)
            top_geo, freq = counts.most_common(1)[0]
            if top_geo in self.hs_classes:
                if getattr(self, 'model_a_loaded', False):
                    prior[self.hs_classes.index(top_geo)] *= (1.0 + 1.2 * (freq / len(frame_votes)))
                else:
                    # Pure geometric voting when Model A is not loaded
                    return top_geo, None, None
                
        fused_probs = avg_probs * prior
        predicted_hs = self.hs_classes[np.argmax(fused_probs)]
        
        # Detect Thumb and Bend Modifiers
        thumb_mod = None
        bend_mod = None
        
        mean_hand = np.mean(stroke_hand_pts, axis=0)
        wrist = mean_hand[0]
        palm_len = max(np.linalg.norm(mean_hand[9] - wrist), 1e-4)
        thumb_dist = np.linalg.norm(mean_hand[4] - mean_hand[5])
        if thumb_dist > 0.55 * palm_len:
            thumb_mod = "hamthumboutmod"
            
        mid_ext = np.linalg.norm(mean_hand[12] - wrist) / palm_len
        if mid_ext < 0.70 and predicted_hs not in ("hamflathand", "hamfist"):
            bend_mod = "hamfingerbendmod"
            
        return predicted_hs, thumb_mod, bend_mod

    def predict_movement(self, stroke_wrist_pts, chest_pts, shoulder_dists):
        """Model B Hybrid Inference: Upgraded Attention-BiGRU / 1D-CNN + Kinematic Prior."""
        norm_traj = []
        for i in range(len(stroke_wrist_pts)):
            norm_w = (stroke_wrist_pts[i] - chest_pts[i]) / shoulder_dists[i]
            z = stroke_wrist_pts[i][2] if len(stroke_wrist_pts[i]) > 2 else 0.0
            norm_traj.append(np.array([norm_w[0], norm_w[1], z], dtype=np.float32))
            
        velocities = np.diff(norm_traj, axis=0) if len(norm_traj) >= 2 else np.zeros((1, 3))
        if len(norm_traj) >= 2:
            arr = np.array(norm_traj, dtype=np.float32)
            if getattr(self, 'model_b_mode', 'legacy') in ("v5_dgs", "v4_hamer") or getattr(self, 'is_v4_model_b', False):
                # V4/V5 Attention-BiGRU: Center-normalize matching Kaggle Cell 2
                arr_norm = arr - arr[0:1, :]
                max_scale = np.max(np.abs(arr_norm)) + 1e-6
                arr_norm = arr_norm / max_scale
                
                orig_t = np.linspace(0, 1, len(arr_norm))
                target_t = np.linspace(0, 1, 32)
                traj_32 = np.zeros((32, 3), dtype=np.float32)
                for dim in range(3):
                    traj_32[:, dim] = np.interp(target_t, orig_t, arr_norm[:, dim])
            else:
                if len(velocities) < 2:
                    velocities = np.tile(velocities, (2, 1))
                arr_v = np.array(velocities)
                t_curr = np.linspace(0, 1, len(arr_v))
                t_new = np.linspace(0, 1, 30)
                f = interp1d(t_curr, arr_v, axis=0, kind='linear', fill_value='extrapolate')
                traj_30 = f(t_new).T.astype(np.float32) # (3, 30)
        else:
            traj_32 = np.zeros((32, 3), dtype=np.float32)
            traj_30 = np.zeros((3, 30), dtype=np.float32)
            
        with torch.no_grad():
            if getattr(self, 'model_b_mode', 'legacy') in ("v5_dgs", "v4_hamer") or getattr(self, 'is_v4_model_b', False):
                tensor_x = torch.FloatTensor(traj_32).unsqueeze(0).to(DEVICE) # (1, 32, 3)
                logits = self.model_b(tensor_x)
            else:
                tensor_x = torch.FloatTensor(traj_30).unsqueeze(0).to(DEVICE) # (1, 3, 30)
                logits = self.model_b(tensor_x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
        # Kinematic closed-form trajectory analysis
        total_disp = np.sum(np.linalg.norm(np.diff([p[:2] for p in norm_traj], axis=0), axis=1)) if len(norm_traj) >= 2 else 0.0
        
        # Max displacement vector from initial position
        displacements = [p[:2] - norm_traj[0][:2] for p in norm_traj]
        dists = [np.linalg.norm(d) for d in displacements]
        max_idx = int(np.argmax(dists)) if len(dists) > 0 else 0
        delta_p = displacements[max_idx] if len(displacements) > 0 else np.zeros(2)
        
        # Enclosed area via Green's Theorem with centroid subtraction
        pts_2d = np.array([p[:2] for p in norm_traj]) if len(norm_traj) >= 4 else np.zeros((0, 2))
        if len(pts_2d) >= 6:
            centroid = np.mean(pts_2d, axis=0)
            cx = pts_2d[:, 0] - centroid[0]
            cy = pts_2d[:, 1] - centroid[1]
            enclosed_area = 0.5 * abs(np.sum(cx[:-1] * cy[1:] - cx[1:] * cy[:-1]))
            span_x = np.ptp(cx)
            span_y = np.ptp(cy)
            aspect = min(span_x, span_y) / max(span_x, span_y, 1e-4)
        else:
            enclosed_area = 0.0
            aspect = 0.0
            
        prior = np.ones(len(self.mv_classes), dtype=np.float32)
        max_disp = max(dists) if len(dists) > 0 else 0.0
        loop_dist = np.linalg.norm(pts_2d[-1] - pts_2d[0]) if len(pts_2d) >= 2 else 1.0
        
        # 1. Quasistatic hold vs Active motion
        if total_disp < 0.05 and max_disp < 0.025:
            if "hamnomotion" in self.mv_classes:
                prior[self.mv_classes.index("hamnomotion")] *= 3.0
        elif max_disp >= 0.035 or total_disp >= 0.06:
            # Active movement detected: heavily suppress hamnomotion so Model B and directional prior can predict real motion
            if "hamnomotion" in self.mv_classes:
                prior[self.mv_classes.index("hamnomotion")] *= 0.01
                
        # 2. Circular motion (requires actual area, balanced aspect ratio, and closed loop)
        is_circle = (len(pts_2d) >= 10 and max_disp >= 0.08 and 
                     loop_dist < 0.40 * max_disp and aspect > 0.35 and enclosed_area > 0.012)
        if is_circle:
            for i, c in enumerate(self.mv_classes):
                if "circle" in c or "stir" in c:
                    prior[i] *= 4.0
                    
        # 3. Directional linear motion (signer-centric)
        if max_disp >= 0.035 and not is_circle:
            dx = float(delta_p[0])
            dy = float(delta_p[1])
            dx_signer = -dx
            dy_signer = -dy  # Up is positive
            
            if abs(dy_signer) > 1.1 * abs(dx_signer):
                if dy_signer > 0.035 and "hammoveu" in self.mv_classes:
                    prior[self.mv_classes.index("hammoveu")] *= 4.0
                elif dy_signer < -0.035 and "hammoved" in self.mv_classes:
                    prior[self.mv_classes.index("hammoved")] *= 4.0
            elif abs(dx_signer) > 1.1 * abs(dy_signer):
                if dx_signer < -0.035 and "hammovel" in self.mv_classes:
                    prior[self.mv_classes.index("hammovel")] *= 4.0
                elif dx_signer > 0.035 and "hammover" in self.mv_classes:
                    prior[self.mv_classes.index("hammover")] *= 4.0
                    
        fused_probs = probs * prior
        pred_idx = np.argmax(fused_probs)
        predicted_mv = self.mv_classes[pred_idx]
        
        # Repetition check (autocorrelation)
        repetition = None
        if len(norm_traj) > 15:
            vel_mag = np.linalg.norm(velocities, axis=1)
            autocorr = np.correlate(vel_mag - np.mean(vel_mag), vel_mag - np.mean(vel_mag), mode='full')
            mid = len(autocorr) // 2
            half = autocorr[mid:]
            peaks = [i for i in range(1, len(half)-1) if half[i] > half[i-1] and half[i] > half[i+1] and half[i] > 0.3 * half[0]]
            if len(peaks) >= 2:
                repetition = "hamrepeatfromstart"
                
        return predicted_mv, repetition

    def process_video(self, video_path):
        """
        Executes complete hybrid pipeline on input video:
          Returns HamNoSys token sequence and SiGML XML.
        """
        kpts_list, scores_list, fps = self.extract_landmarks(video_path)
        if len(kpts_list) < 5:
            return "hamflathand hamextfingeru hampalmd hamchest", "<sigml/>"
            
        # Parse tracking streams
        r_wrist_pts = []
        l_wrist_pts = []
        hip_y_pts = []
        r_hand_pts = []
        chest_pts = []
        shoulder_dists = []
        
        for kpts, scores in zip(kpts_list, scores_list):
            l_score = np.mean(scores[91:112]) if len(scores) >= 133 else 0.0
            r_score = np.mean(scores[112:133]) if len(scores) >= 133 else 0.0
            
            # Primary Right Hand
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
        
        # 1. Stroke activity windowing
        start_t, end_t = self.compute_active_stroke_window(r_wrist_pts, hip_y_pts, fps=fps)
        if end_t <= start_t:
            start_t, end_t = 0, len(r_hand_pts)
            
        stroke_hands = r_hand_pts[start_t:end_t] if len(r_hand_pts[start_t:end_t]) > 0 else r_hand_pts
        stroke_r_wrists = r_wrist_pts[start_t:end_t] if len(r_wrist_pts[start_t:end_t]) > 0 else r_wrist_pts
        stroke_chests = chest_pts[start_t:end_t] if len(chest_pts[start_t:end_t]) > 0 else chest_pts
        stroke_sdists = shoulder_dists[start_t:end_t] if len(shoulder_dists[start_t:end_t]) > 0 else shoulder_dists
        
        # 2. Dual-hand symmetry
        symmetry = self.compute_dual_hand_symmetry(l_wrist_pts[start_t:end_t], stroke_r_wrists, hip_y_pts[start_t:end_t])
        
        # 3. Model A: Handshape + Modifiers
        handshape, thumb_mod, bend_mod = self.predict_handshape(stroke_hands)
        
        # 4. Deterministic Orientation
        mean_hand = np.mean(stroke_hands, axis=0)
        ext_finger, palm_ori = self.compute_3d_orientation(mean_hand)
        
        # 5. Deterministic Anatomical Location + Contact
        mean_chest = np.mean(stroke_chests, axis=0)
        mean_sdist = np.mean(stroke_sdists)
        hand_center = mean_hand[9]
        location, contact = self.compute_anatomical_location(hand_center, kpts_list[len(kpts_list)//2], mean_sdist)
        
        # 6. Model B: Movement + Repetition
        movement, repetition = self.predict_movement(stroke_r_wrists, stroke_chests, stroke_sdists)
        
        # =================================================================
        # CFG Slot Assembly (Section 9)
        # =================================================================
        tokens = []
        # Slot 0: Symmetry
        if symmetry in ("hamsymmlr", "hamplus"):
            tokens.append(symmetry)
            
        # Slot 1: Handshape Base
        tokens.append(handshape)
        
        # Slot 2: Thumb / Bend Modifiers
        if thumb_mod:
            tokens.append(thumb_mod)
        if bend_mod:
            tokens.append(bend_mod)
            
        # Slot 3: Extended Finger Orientation
        tokens.append(ext_finger)
        
        # Slot 4: Palm Orientation
        tokens.append(palm_ori)
        
        # Slot 5: Anatomical Body Location
        tokens.append(location)
        
        # Slot 6: Contact / Distance
        if contact:
            tokens.append(contact)
            
        # Slot 7: Movement
        if movement and movement != "hamnomotion":
            tokens.append(movement)
            
        # Slot 8: Repetition
        if repetition:
            tokens.append(repetition)
            
        hamnosys_str = " ".join(tokens)
        
        # Generate Validated SiGML XML
        sigml_xml = self.generate_sigml_xml(tokens)
        
        return hamnosys_str, sigml_xml

    def generate_sigml_xml(self, tokens, gloss="sign"):
        """Converts CFG tokens to CWASA-compliant SiGML XML via HamNoSys2SiGML generator."""
        import subprocess
        from sigml_grammar_linter import sanitize_sigml_xml, sanitize_hamnosys_sequence
        
        canonical_tokens = sanitize_hamnosys_sequence(tokens)
        
        spreadsheet_path = os.path.join(
            _REPO_ROOT, "Senior Code", "HamNoSys2SiGML-master", "HamNoSys2SiGML-master", "Original", "conversionSpreadSheet.txt"
        )
        ham2sigml_script = os.path.join(
            _REPO_ROOT, "Senior Code", "HamNoSys2SiGML-master", "HamNoSys2SiGML-master", "Original", "HamNoSys2SiGML.py"
        )
        
        if os.path.exists(spreadsheet_path) and os.path.exists(ham2sigml_script):
            mapping = {}
            with open(spreadsheet_path, "r", encoding="utf-8") as f:
                for line in f:
                    if "," in line:
                        parts = line.strip().split(",")
                        tag = parts[0].strip().lower()
                        code = parts[1].strip().split()[0].strip()
                        mapping[tag] = code
                        
            unicode_chars = []
            for tag in canonical_tokens:
                k = tag.lower()
                if k in mapping:
                    try:
                        unicode_chars.append(chr(int(mapping[k], 16)))
                    except Exception:
                        pass
                        
            if unicode_chars:
                unicode_str = "".join(unicode_chars)
                cmd = [sys.executable, ham2sigml_script, unicode_str]
                proc = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(ham2sigml_script))
                if proc.returncode == 0 and "<sigml" in proc.stdout:
                    return sanitize_sigml_xml(proc.stdout)
                    
        # Fallback to direct XML generation
        root = ET.Element("sigml")
        hns_sign = ET.SubElement(root, "hns_sign", gloss=gloss)
        ET.SubElement(hns_sign, "hamnosys_nonmanual")
        manual = ET.SubElement(hns_sign, "hamnosys_manual")
        
        for tok in canonical_tokens:
            ET.SubElement(manual, tok)
            
        return sanitize_sigml_xml(ET.tostring(root, encoding="unicode"))

# Singleton instance
_v3_engine = None

def get_v3_pipeline():
    global _v3_engine
    if _v3_engine is None:
        _v3_engine = V3HybridPipeline()
    return _v3_engine

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline_v3_hybrid.py <video_path>")
        sys.exit(1)
        
    vid = sys.argv[1]
    engine = get_v3_pipeline()
    hns, sigml = engine.process_video(vid)
    print("\n" + "=" * 60)
    print("  V3 Hybrid Pipeline Output")
    print("=" * 60)
    print(f"Video: {vid}")
    print(f"HamNoSys: {hns}")
    print(f"SiGML XML:\n{sigml}")
    
    # Write to files
    out_hns_path = os.path.join(_SCRIPT_DIR, "hamnosys.txt")
    out_sigml_path = os.path.join(_SCRIPT_DIR, "output.sigml")
    with open(out_hns_path, "w") as f:
        f.write(hns + "\n")
    with open(out_sigml_path, "w") as f:
        f.write(sigml)
    print(f"Saved: {out_hns_path}")
    print(f"Saved: {out_sigml_path}")
