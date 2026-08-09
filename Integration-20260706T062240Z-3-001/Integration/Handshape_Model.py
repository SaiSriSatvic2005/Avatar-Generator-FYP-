#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
# FINGER STATE (3D EUCLIDEAN DISTANCE)
# =============================

def dist_3d(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def finger_open(tip, pip, lm):
    wrist = lm[0]
    d_tip = dist_3d(lm[tip], wrist)
    d_pip = dist_3d(lm[pip], wrist)
    return d_tip > 1.15 * d_pip

def get_finger_states(lm):
    thumb = dist_3d(lm[4], lm[17]) > 0.18 or lm[4].x < lm[3].x if len(lm)>4 else False
    index = finger_open(8, 6, lm)
    middle = finger_open(12, 10, lm)
    ring = finger_open(16, 14, lm)
    pinky = finger_open(20, 18, lm)

    return thumb, index, middle, ring, pinky


# =============================
# HANDSHAPE CLASSIFICATION
# =============================

def classify_handshape(lm):
    thumb, index, middle, ring, pinky = get_finger_states(lm)

    # fist
    if not thumb and not index and not middle and not ring and not pinky:
        return "hamfist"

    # flat hand
    if index and middle and ring and pinky:
        return "hamflathand"

    # index finger
    if index and not middle and not ring and not pinky:
        return "hamfinger2"

    # two fingers / victory
    if index and middle and not ring and not pinky:
        return "hamfinger23"

    # pinch
    if thumb and index and not middle and not ring and not pinky:
        return "hampinch12"

    # thumb modifier
    if thumb and not index:
        return "hamthumboutmod"

    # straight fingers
    if index and middle and ring and pinky and not thumb:
        return "hamflathand"

    # fallbacks
    if index:
        return "hamfinger2"
    if thumb:
        return "hamthumboutmod"

    return "hamflathand"


# =============================
# HANDSHAPE MODULE
# =============================

def run_handshape_module(video_path):
    if get_video_landmarks is not None:
        frames_info = get_video_landmarks(video_path)
        raw_labels = []
        for frame_info in frames_info:
            hand_lm = frame_info.get("primary_hand")
            if hand_lm and hasattr(hand_lm, "landmark"):
                raw_labels.append(classify_handshape(hand_lm.landmark))
            else:
                raw_labels.append("hamflathand")

        smoothed_labels = smooth_frame_sequence(raw_labels, window=3)
        final_label = summarize_label_sequence(smoothed_labels, default="hamflathand", window=3)
        return {"per_frame": smoothed_labels, "final": final_label}

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
    return {"per_frame": smoothed_labels, "final": final_label}

