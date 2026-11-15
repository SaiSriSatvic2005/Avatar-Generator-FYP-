#!/usr/bin/env python
# coding: utf-8
"""
Handshape Model V3 - Robust Kinematic & Scale-Invariant Classifier
- Multi-finger joint chain analysis + scale-invariant extension ratios
- View-invariant palm-frame coordinate projection
- Decoupled base handshape (NODE_A), thumb modifier (NODE_C), bend modifier (NODE_D), and hand part (NODE_E)
- Multi-hand active signer tracking (highest elevation / signing space)
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import Counter

# =============================
# HAMNOSYS HANDSHAPE LABELS
# =============================

NODE_A = [
    "hamfist", "hamflathand", "hamfinger2", "hamfinger23", "hamfinger23spread",
    "hamfinger2345", "hampinch12", "hampinch12open", "hampinchall",
    "hamcee12", "hamceeall", "hamindexfinger", "hammiddlefinger"
]

NODE_C = ["hamthumboutmod", "hamthumbacrossmod", "hamthumbopenmod"]

NODE_D = [
    "hamdoublebent", "hamdoublehooked",
    "hamfingerstraightmod", "hamfingerbendmod", "hamfingerhookmod"
]

NODE_E = [
    "hamthumb", "hamindexfinger", "hammiddlefinger", "hamringfinger", "hampinky",
    "hambetween", "hamfingernail", "hamfingerpad", "hamfingerside", "hamfingermidjoint",
    "hamfingertip"
]

# MediaPipe finger landmark indices (mcp, pip, dip, tip)
FINGER_INDICES = {
    "thumb":  (1, 2, 3, 4),
    "index":  (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring":   (13, 14, 15, 16),
    "pinky":  (17, 18, 19, 20)
}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def euclidean_3d(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)


def finger_extension_ratio(lm, mcp, pip, dip, tip):
    """
    Scale-invariant ratio: direct Euclidean distance MCP->TIP divided by
    the kinematic chain perimeter MCP->PIP + PIP->DIP + DIP->TIP.
    Straight finger: ratio in [0.78, 1.0].
    Curled / closed finger: ratio < 0.55.
    """
    d_mcp_tip = euclidean_3d(lm[mcp], lm[tip])
    chain = (euclidean_3d(lm[mcp], lm[pip]) +
             euclidean_3d(lm[pip], lm[dip]) +
             euclidean_3d(lm[dip], lm[tip]))
    return float(d_mcp_tip / (chain + 1e-8))


def finger_curl_angle(lm, mcp, pip, dip, tip):
    """
    3D joint angle between MCP->PIP and PIP->TIP vectors in degrees.
    0 deg = fully extended, 180 deg = fully curled.
    """
    v1 = np.array([lm[pip].x - lm[mcp].x, lm[pip].y - lm[mcp].y, lm[pip].z - lm[mcp].z])
    v2 = np.array([lm[tip].x - lm[pip].x, lm[tip].y - lm[pip].y, lm[tip].z - lm[pip].z])
    cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))


def classify_finger_state(lm, finger_name):
    """
    Classify finger into 'straight', 'bent', 'hooked', 'closed'
    using hybrid scale-invariant extension ratio and joint curl angle.
    """
    indices = FINGER_INDICES[finger_name]
    ratio = finger_extension_ratio(lm, *indices)
    angle = finger_curl_angle(lm, *indices)

    if ratio >= 0.78 and angle < 50.0:
        return "straight"
    elif ratio >= 0.55 and angle < 85.0:
        return "bent"
    elif ratio >= 0.40 and angle < 125.0:
        return "hooked"
    else:
        return "closed"


def get_palm_metrics(lm):
    """
    Extract palm dimensions and coordinate frame:
    - palm_length: wrist(0) to middle_mcp(9)
    - palm_width: index_mcp(5) to pinky_mcp(17)
    """
    p0 = np.array([lm[0].x, lm[0].y, lm[0].z])
    p5 = np.array([lm[5].x, lm[5].y, lm[5].z])
    p9 = np.array([lm[9].x, lm[9].y, lm[9].z])
    p17 = np.array([lm[17].x, lm[17].y, lm[17].z])

    palm_len = float(np.linalg.norm(p9 - p0) + 1e-8)
    palm_width = float(np.linalg.norm(p17 - p5) + 1e-8)
    return palm_len, palm_width


def classify_handshape_components(lm):
    """
    Comprehensive feature decomposition:
    Returns (base_handshape, thumb_mod, bend_mod, hand_part)
    """
    states = {f: classify_finger_state(lm, f) for f in ["index", "middle", "ring", "pinky"]}
    palm_len, palm_width = get_palm_metrics(lm)

    # Extension ratios & curl angles
    ext_ratios = {f: finger_extension_ratio(lm, *FINGER_INDICES[f]) for f in ["index", "middle", "ring", "pinky"]}

    # Thumb analysis
    thumb_tip = np.array([lm[4].x, lm[4].y, lm[4].z])
    index_tip = np.array([lm[8].x, lm[8].y, lm[8].z])
    index_mcp = np.array([lm[5].x, lm[5].y, lm[5].z])
    wrist = np.array([lm[0].x, lm[0].y, lm[0].z])

    d_thumb_index_tip = float(np.linalg.norm(thumb_tip - index_tip) / palm_len)
    d_thumb_index_mcp = float(np.linalg.norm(thumb_tip - index_mcp) / palm_len)
    thumb_curl = finger_curl_angle(lm, *FINGER_INDICES["thumb"])

    # 1. THUMB MODIFIER (NODE_C)
    thumb_mod = None
    # Check if thumb is tucked across palm towards the fingers
    if (thumb_tip[0] - index_mcp[0]) > 0.025:
        thumb_mod = "hamthumbacrossmod"
    elif d_thumb_index_mcp > 0.55 or thumb_curl < 55.0:
        thumb_mod = "hamthumboutmod"
    elif d_thumb_index_mcp > 0.35:
        thumb_mod = "hamthumbopenmod"

    # 2. BASE HANDSHAPE (NODE_A)
    num_straight = sum(1 for s in states.values() if s == "straight")
    num_closed = sum(1 for s in states.values() if s == "closed")
    num_bent = sum(1 for s in states.values() if s == "bent")
    other_fingers_curled = all(states[f] in ("bent", "closed", "hooked") for f in ["middle", "ring", "pinky"])

    # Spread between index and middle
    spread_idx_mid = float(np.linalg.norm(index_tip - np.array([lm[12].x, lm[12].y, lm[12].z])) / palm_len)

    base_handshape = "hamflathand"

    if (num_closed + num_bent == 4 and states["index"] != "straight") or (num_closed >= 3 and states["index"] != "straight"):
        # All 4 fingers closed / bent tightly into palm -> Fist
        base_handshape = "hamfist"
    elif states["index"] == "straight" and other_fingers_curled:
        # Only index straight -> hamfinger2
        if d_thumb_index_tip < 0.22:
            base_handshape = "hampinch12"
        else:
            base_handshape = "hamfinger2"
    elif states["index"] == "straight" and states["middle"] == "straight" and states["ring"] in ("bent", "closed") and states["pinky"] in ("bent", "closed"):
        # Two fingers straight -> hamfinger23
        if spread_idx_mid > 0.28:
            base_handshape = "hamfinger23spread"
        else:
            base_handshape = "hamfinger23"
    elif d_thumb_index_tip < 0.25 and states["middle"] == "straight":
        base_handshape = "hampinch12open"
    elif num_straight >= 3:
        if spread_idx_mid > 0.32:
            base_handshape = "hamfinger2345"
        else:
            base_handshape = "hamflathand"
    elif num_bent >= 2 and d_thumb_index_tip < 0.45:
        base_handshape = "hamcee12" if states["middle"] != "straight" else "hamceeall"
    elif ext_ratios["index"] > 0.70 and ext_ratios["middle"] < 0.65:
        base_handshape = "hamfinger2"
    elif all(r > 0.68 for r in ext_ratios.values()):
        base_handshape = "hamflathand"
    else:
        # Fallback to dominant shape
        if states["index"] in ("straight", "bent") and not other_fingers_curled:
            base_handshape = "hamflathand"
        elif states["index"] == "straight":
            base_handshape = "hamfinger2"
        else:
            base_handshape = "hamfist"

    # 3. FINGER BEND MODIFIER (NODE_D)
    bend_mod = None
    if num_bent >= 2:
        bend_mod = "hamdoublebent"
    elif sum(1 for s in states.values() if s == "hooked") >= 2:
        bend_mod = "hamdoublehooked"
    elif num_straight >= 3 and thumb_mod != "hamthumboutmod":
        bend_mod = "hamfingerstraightmod"
    elif num_bent >= 1:
        bend_mod = "hamfingerbendmod"

    # 4. HAND PART / FINGER PART (NODE_E)
    hand_part = "hamfingertip"
    if spread_idx_mid > 0.30 and num_straight >= 2:
        hand_part = "hambetween"
    elif base_handshape == "hamfist" and thumb_mod == "hamthumboutmod":
        hand_part = "hamthumbside"

    return base_handshape, thumb_mod, bend_mod, hand_part


def classify_handshape(lm):
    """Backwards-compatible wrapper returning primary base token."""
    base, _, _, _ = classify_handshape_components(lm)
    return base


def run_handshape_module(video_path):
    """
    Executes handshape feature extraction across video frames.
    Extracts base handshape, thumb modifier, bend modifier, and hand part.
    Tracks dominant signing hand (highest elevation in signing space).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    per_frame_tokens = []
    base_list = []
    thumb_list = []
    bend_list = []
    part_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            # Active hand: pick hand with highest elevation (lowest wrist y)
            active_hand = min(results.multi_hand_landmarks, key=lambda h: h.landmark[0].y)
            lm = active_hand.landmark

            base, thumb_m, bend_m, h_part = classify_handshape_components(lm)
            base_list.append(base)
            if thumb_m: thumb_list.append(thumb_m)
            if bend_m: bend_list.append(bend_m)
            if h_part: part_list.append(h_part)

            # Record composite tuple for per-frame tracking
            token_tuple = tuple(x for x in [base, thumb_m, bend_m, h_part] if x)
            per_frame_tokens.append(token_tuple)
        else:
            per_frame_tokens.append(("none",))

    cap.release()

    if not base_list:
        return {
            "per_frame": [("hamflathand",)],
            "final": "hamflathand",
            "base": "hamflathand",
            "thumb_mod": "hamthumboutmod",
            "bend_mod": None,
            "hand_part": "hamfingertip"
        }

    final_base = Counter(base_list).most_common(1)[0][0]
    final_thumb = Counter(thumb_list).most_common(1)[0][0] if thumb_list else None
    final_bend = Counter(bend_list).most_common(1)[0][0] if bend_list else None
    final_part = Counter(part_list).most_common(1)[0][0] if part_list else "hamfingertip"

    return {
        "per_frame": per_frame_tokens,
        "final": final_base,
        "base": final_base,
        "thumb_mod": final_thumb,
        "bend_mod": final_bend,
        "hand_part": final_part
    }
