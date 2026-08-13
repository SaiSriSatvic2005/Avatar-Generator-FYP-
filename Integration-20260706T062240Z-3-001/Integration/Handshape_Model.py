#!/usr/bin/env python
# coding: utf-8
"""
HamNoSys Handshape Classification Module — v2
Uses joint-angle-based finger curl detection (not binary open/closed).
Supports dual-hand output for two-handed signs.
"""

import cv2
import mediapipe as mp

# =============================
# HAMNOSYS HANDSHAPE LABELS
# =============================

NODE_A = [
    "hamfist","hamflathand","hamfinger2","hamfinger23","hamfinger23spread",
    "hamfinger2345","hampinch12","hampinch12open","hampinchall",
    "hamcee12","hamceeall","hamceeopen"
]

NODE_C = ["hamthumboutmod","hamthumbacrossmod","hamthumbopenmod"]

NODE_D = [
    "hamdoublebent","hamdoublehooked",
    "hamfingerstraightmod","hamfingerbendmod","hamfingerhookmod"
]

NODE_E = [
    "hamthumb","hamindexfinger","hammiddlefinger","hamringfinger","hampinky",
    "hambetween","hamfingernail","hamfingerpad","hamfingerside","hamfingermidjoint"
]

import os
import sys
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

try:
    from shared_landmarks import get_video_landmarks
except ImportError:
    get_video_landmarks = None

from temporal_utils import smooth_frame_sequence, summarize_label_sequence

# =============================
# 3D GEOMETRY HELPERS
# =============================

def dist_3d(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def _to_vec(p):
    """Convert a landmark point to numpy array."""
    return np.array([p.x, p.y, p.z], dtype=np.float64)

def _hand_scale(lm):
    """Compute hand scale as wrist-to-middle-MCP distance (normalizes for camera distance)."""
    wrist = _to_vec(lm[0])
    mid_mcp = _to_vec(lm[9])
    scale = np.linalg.norm(mid_mcp - wrist)
    return max(scale, 1e-6)

def _angle_at_joint(a, b, c):
    """Compute the angle (degrees) at point b formed by vectors b→a and b→c."""
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-8 or norm_bc < 1e-8:
        return 0.0
    cos_angle = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


# =============================
# FINGER CURL ANGLE COMPUTATION
# =============================

# MediaPipe Hand Landmark indices:
# Thumb:  1(CMC), 2(MCP), 3(IP), 4(TIP)
# Index:  5(MCP), 6(PIP), 7(DIP), 8(TIP)
# Middle: 9(MCP), 10(PIP), 11(DIP), 12(TIP)
# Ring:   13(MCP), 14(PIP), 15(DIP), 16(TIP)
# Pinky:  17(MCP), 18(PIP), 19(DIP), 20(TIP)

FINGER_JOINTS = {
    "thumb":  {"mcp": 2, "pip": 3, "dip": 3, "tip": 4, "base": 1},
    "index":  {"mcp": 5, "pip": 6, "dip": 7, "tip": 8, "base": 0},
    "middle": {"mcp": 9, "pip": 10, "dip": 11, "tip": 12, "base": 0},
    "ring":   {"mcp": 13, "pip": 14, "dip": 15, "tip": 16, "base": 0},
    "pinky":  {"mcp": 17, "pip": 18, "dip": 19, "tip": 20, "base": 0},
}


def compute_finger_curl(lm, finger_name):
    """
    Compute curl state of a finger using joint angles.
    Returns one of: 'straight', 'slightly_bent', 'bent', 'hooked'
    
    Angle thresholds:
      straight:      PIP angle > 150°  (nearly flat)
      slightly_bent: PIP angle 100°–150°
      bent:          PIP angle 60°–100°
      hooked:        PIP angle < 60° AND DIP angle < 90°
    """
    joints = FINGER_JOINTS[finger_name]
    
    if finger_name == "thumb":
        # Thumb uses different geometry: CMC→MCP→IP→TIP
        cmc = _to_vec(lm[joints["base"]])
        mcp = _to_vec(lm[joints["mcp"]])
        ip  = _to_vec(lm[joints["pip"]])
        tip = _to_vec(lm[joints["tip"]])
        
        # Thumb curl: angle at IP joint
        ip_angle = _angle_at_joint(mcp, ip, tip)
        # Thumb abduction: distance from thumb tip to index MCP
        thumb_tip = _to_vec(lm[4])
        index_mcp = _to_vec(lm[5])
        pinky_mcp = _to_vec(lm[17])
        scale = _hand_scale(lm)
        abduction = np.linalg.norm(thumb_tip - index_mcp) / scale
        
        if ip_angle > 150 and abduction > 0.4:
            return "straight"  # Thumb extended outward
        elif ip_angle > 120:
            return "slightly_bent"
        elif ip_angle > 70:
            return "bent"
        else:
            return "hooked"
    
    # For non-thumb fingers
    mcp_pt = _to_vec(lm[joints["mcp"]])
    pip_pt = _to_vec(lm[joints["pip"]])
    dip_pt = _to_vec(lm[joints["dip"]])
    tip_pt = _to_vec(lm[joints["tip"]])
    base_pt = _to_vec(lm[joints["base"]])  # wrist
    
    # PIP angle: how much the finger is curled at the main knuckle
    pip_angle = _angle_at_joint(mcp_pt, pip_pt, dip_pt)
    # DIP angle: how much the fingertip curls
    dip_angle = _angle_at_joint(pip_pt, dip_pt, tip_pt)
    
    if pip_angle > 150:
        return "straight"
    elif pip_angle > 100:
        return "slightly_bent"
    elif pip_angle > 60:
        if dip_angle < 90:
            return "hooked"  # PIP bent + DIP sharply bent = hook
        return "bent"
    else:
        return "hooked"


def get_finger_curl_states(lm):
    """Get curl state for all 5 fingers."""
    return {
        "thumb":  compute_finger_curl(lm, "thumb"),
        "index":  compute_finger_curl(lm, "index"),
        "middle": compute_finger_curl(lm, "middle"),
        "ring":   compute_finger_curl(lm, "ring"),
        "pinky":  compute_finger_curl(lm, "pinky"),
    }


# =============================
# LEGACY BINARY FINGER STATE (kept for backward compat)
# =============================

def finger_open(tip, pip, lm):
    wrist = lm[0]
    d_tip = dist_3d(lm[tip], wrist)
    d_pip = dist_3d(lm[pip], wrist)
    return d_tip > 1.15 * d_pip

def get_finger_states(lm):
    """Binary open/closed per finger (legacy, used as fallback)."""
    thumb = dist_3d(lm[4], lm[17]) > 0.18 or lm[4].x < lm[3].x if len(lm)>4 else False
    index = finger_open(8, 6, lm)
    middle = finger_open(12, 10, lm)
    ring = finger_open(16, 14, lm)
    pinky = finger_open(20, 18, lm)
    return thumb, index, middle, ring, pinky


# =============================
# ADVANCED HANDSHAPE CLASSIFICATION
# =============================

def _thumb_index_aperture(lm):
    """Distance between thumb tip and index tip, normalized by hand scale."""
    thumb_tip = _to_vec(lm[4])
    index_tip = _to_vec(lm[8])
    return np.linalg.norm(thumb_tip - index_tip) / _hand_scale(lm)

def _thumb_touches_finger(lm, finger_tip_idx):
    """Check if thumb tip is close to a specific fingertip."""
    thumb_tip = _to_vec(lm[4])
    finger_tip = _to_vec(lm[finger_tip_idx])
    return np.linalg.norm(thumb_tip - finger_tip) / _hand_scale(lm) < 0.35

def _index_middle_spread(lm):
    """Measure spread angle between index and middle fingers."""
    index_tip = _to_vec(lm[8])
    middle_tip = _to_vec(lm[12])
    index_mcp = _to_vec(lm[5])
    middle_mcp = _to_vec(lm[9])
    
    v1 = index_tip - index_mcp
    v2 = middle_tip - middle_mcp
    
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0
    
    cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def classify_handshape(lm):
    """
    Classify handshape using angle-based curl detection.
    Falls back to binary detection for edge cases.
    """
    curls = get_finger_curl_states(lm)
    
    # Helper: is a finger extended (straight or slightly bent)?
    def is_extended(finger):
        return curls[finger] in ("straight", "slightly_bent")
    
    # Helper: is a finger closed (bent or hooked)?
    def is_closed(finger):
        return curls[finger] in ("bent", "hooked")
    
    # Helper: is a finger specifically bent (not hooked, not straight)?
    def is_bent(finger):
        return curls[finger] == "bent"
    
    # Helper: is a finger specifically hooked?
    def is_hooked(finger):
        return curls[finger] == "hooked"
    
    thumb_ext = is_extended("thumb")
    index_ext = is_extended("index")
    middle_ext = is_extended("middle")
    ring_ext = is_extended("ring")
    pinky_ext = is_extended("pinky")
    
    four_fingers_ext = index_ext and middle_ext and ring_ext and pinky_ext
    four_fingers_closed = is_closed("index") and is_closed("middle") and is_closed("ring") and is_closed("pinky")
    
    # === FIST: all fingers closed ===
    if four_fingers_closed and not thumb_ext:
        return "hamfist"
    
    # === FLAT HAND: all four fingers extended ===
    if four_fingers_ext:
        if thumb_ext:
            return "hamflathand"
        return "hamflathand"  # hamflathand even without thumb
    
    # === C-SHAPES: thumb and fingers form a C/O shape ===
    aperture = _thumb_index_aperture(lm)
    if thumb_ext and index_ext and is_closed("ring") and is_closed("pinky"):
        if 0.3 < aperture < 0.8:
            if middle_ext:
                return "hamceeopen"   # C with all fingers
            return "hamcee12"         # C with thumb + index
    
    # === PINCH: thumb touching index ===
    if _thumb_touches_finger(lm, 8):
        if is_closed("middle") and is_closed("ring") and is_closed("pinky"):
            return "hampinch12"       # Pinch with thumb + index only
        if _thumb_touches_finger(lm, 12):
            return "hampinchall"      # Pinch with all fingers
        if middle_ext or ring_ext:
            return "hampinch12open"   # Pinch with other fingers open
        return "hampinch12"
    
    # === FINGER MODIFIERS: bent/hooked states ===
    if is_bent("index") and is_bent("middle") and is_bent("ring") and is_bent("pinky"):
        return "hamfingerbendmod"     # All fingers bent (claw-like)
    
    if is_hooked("index") and is_hooked("middle") and is_hooked("ring") and is_hooked("pinky"):
        return "hamfingerhookmod"     # All fingers hooked
    
    # Double bent/hooked (index + middle only)
    if (is_bent("index") or is_hooked("index")) and (is_bent("middle") or is_hooked("middle")):
        if is_closed("ring") and is_closed("pinky"):
            if is_hooked("index") and is_hooked("middle"):
                return "hamdoublehooked"
            return "hamdoublebent"
    
    # === INDEX FINGER only ===
    if index_ext and not middle_ext and not ring_ext and not pinky_ext:
        return "hamfinger2"
    
    # === TWO FINGERS (index + middle) ===
    if index_ext and middle_ext and not ring_ext and not pinky_ext:
        spread = _index_middle_spread(lm)
        if spread > 15:
            return "hamfinger23spread"  # V-sign / spread
        return "hamfinger23"            # Two fingers together
    
    # === FOUR FINGERS (no thumb) ===
    if index_ext and middle_ext and ring_ext and pinky_ext and not thumb_ext:
        return "hamfinger2345"
    
    # === THUMB MODIFIERS ===
    if thumb_ext and four_fingers_closed:
        # Check thumb position relative to hand
        thumb_tip = _to_vec(lm[4])
        index_mcp = _to_vec(lm[5])
        pinky_mcp = _to_vec(lm[17])
        
        # Thumb across palm vs. thumb out
        hand_width_vec = pinky_mcp - index_mcp
        thumb_vec = thumb_tip - _to_vec(lm[2])
        
        if np.dot(thumb_vec[:2], hand_width_vec[:2]) > 0:
            return "hamthumbacrossmod"
        return "hamthumboutmod"
    
    if thumb_ext and not index_ext:
        return "hamthumboutmod"
    
    # === STRAIGHT FINGER MODIFIER (index extended + bent at MCP) ===
    if index_ext and is_bent("middle") and is_bent("ring") and is_bent("pinky"):
        # Check if index is truly straight vs. slightly bent
        if curls["index"] == "straight":
            return "hamfingerstraightmod"
        return "hamfinger2"
    
    # === FALLBACKS ===
    if index_ext:
        return "hamfinger2"
    if thumb_ext:
        return "hamthumboutmod"
    
    return "hamflathand"


# =============================
# HANDSHAPE MODULE (with dual-hand support)
# =============================

def _classify_hand_frames(frames_info, hand_key):
    """Classify handshape for a specific hand across all frames."""
    raw_labels = []
    for frame_info in frames_info:
        hand_lm = frame_info.get(hand_key)
        if hand_lm and hasattr(hand_lm, "landmark"):
            raw_labels.append(classify_handshape(hand_lm.landmark))
        else:
            raw_labels.append(None)
    return raw_labels


def run_handshape_module(video_path):
    """
    Run handshape classification.
    Returns dual-hand output when both hands are consistently detected.
    """
    if get_video_landmarks is not None:
        frames_info = get_video_landmarks(video_path)
        
        # Classify both hands
        right_labels = _classify_hand_frames(frames_info, "right_hand")
        left_labels = _classify_hand_frames(frames_info, "left_hand")
        
        # Count valid detections for each hand
        right_valid = sum(1 for x in right_labels if x is not None)
        left_valid = sum(1 for x in left_labels if x is not None)
        total_frames = max(len(frames_info), 1)
        
        right_ratio = right_valid / total_frames
        left_ratio = left_valid / total_frames
        
        # Both hands detected in >20% of frames → dual-hand output
        is_dual = right_ratio > 0.20 and left_ratio > 0.20
        
        # Primary hand (right preferred, or whatever is detected)
        primary_labels = []
        for frame_info in frames_info:
            hand_lm = frame_info.get("primary_hand")
            if hand_lm and hasattr(hand_lm, "landmark"):
                primary_labels.append(classify_handshape(hand_lm.landmark))
            else:
                primary_labels.append("hamflathand")
        
        smoothed_primary = smooth_frame_sequence(primary_labels, window=3)
        final_primary = summarize_label_sequence(smoothed_primary, default="hamflathand", window=3)
        
        result = {"per_frame": smoothed_primary, "final": final_primary}
        
        if is_dual:
            # Fill None gaps with neighbor values for smoothing
            right_filled = [x if x is not None else "hamflathand" for x in right_labels]
            left_filled = [x if x is not None else "hamflathand" for x in left_labels]
            
            smoothed_right = smooth_frame_sequence(right_filled, window=3)
            smoothed_left = smooth_frame_sequence(left_filled, window=3)
            
            result["right_hand"] = {
                "per_frame": smoothed_right,
                "final": summarize_label_sequence(smoothed_right, default="hamflathand", window=3)
            }
            result["left_hand"] = {
                "per_frame": smoothed_left,
                "final": summarize_label_sequence(smoothed_left, default="hamflathand", window=3)
            }
            result["is_dual"] = True
        else:
            result["is_dual"] = False
        
        return result

    # Fallback to direct MediaPipe if shared_landmarks unavailable
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    raw_labels = []
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            label = "hamflathand"
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0].landmark
                label = classify_handshape(lm)
            raw_labels.append(label)

    cap.release()
    smoothed_labels = smooth_frame_sequence(raw_labels, window=3)
    final_label = summarize_label_sequence(smoothed_labels, default="hamflathand", window=3)
    return {"per_frame": smoothed_labels, "final": final_label, "is_dual": False}

