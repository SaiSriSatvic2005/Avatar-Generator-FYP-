#!/usr/bin/env python
# coding: utf-8
"""
Head & Face Location Model V3
- Pose-assisted face anchors (Nose, Mouth, Eyes, Ears, Chin, Forehead)
- Robust against hand-face occlusion (Pose landmarks remain stable when hand touches face)
- Active signing hand elevation tracking
- Classifies: hamchin, hamlips, hamnose, hamforehead, hamcheek, hamear, hamunderchin, hamheadtop
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import Counter

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4
)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4
)


def get_face_anchors_from_pose(pose_landmarks):
    """
    Extract facial anchor points from MediaPipe Pose landmarks:
    Nose: 0, Left Eye: 2, Right Eye: 5, Left Ear: 7, Right Ear: 8,
    Mouth Left: 9, Mouth Right: 10
    """
    plm = pose_landmarks.landmark
    nose = np.array([plm[0].x, plm[0].y, plm[0].z])
    eye_l = np.array([plm[2].x, plm[2].y, plm[2].z])
    eye_r = np.array([plm[5].x, plm[5].y, plm[5].z])
    ear_l = np.array([plm[7].x, plm[7].y, plm[7].z])
    ear_r = np.array([plm[8].x, plm[8].y, plm[8].z])
    mouth_l = np.array([plm[9].x, plm[9].y, plm[9].z])
    mouth_r = np.array([plm[10].x, plm[10].y, plm[10].z])

    mouth_center = (mouth_l + mouth_r) / 2.0
    eye_center = (eye_l + eye_r) / 2.0
    ear_center = (ear_l + ear_r) / 2.0

    # Face scale: distance from eye center to mouth center
    face_scale = float(np.linalg.norm(mouth_center[:2] - eye_center[:2]) + 1e-6)

    # Projected anchors
    chin = mouth_center.copy()
    chin[1] += 0.65 * face_scale  # Chin is below mouth

    underchin = mouth_center.copy()
    underchin[1] += 1.10 * face_scale

    forehead = eye_center.copy()
    forehead[1] -= 0.85 * face_scale  # Forehead is above eye line

    headtop = eye_center.copy()
    headtop[1] -= 1.60 * face_scale

    cheek_l = (mouth_l + ear_l) / 2.0
    cheek_r = (mouth_r + ear_r) / 2.0

    return {
        "nose": nose,
        "mouth": mouth_center,
        "chin": chin,
        "underchin": underchin,
        "forehead": forehead,
        "headtop": headtop,
        "ear_l": ear_l,
        "ear_r": ear_r,
        "cheek_l": cheek_l,
        "cheek_r": cheek_r,
        "scale": face_scale,
        "mouth_y": mouth_center[1],
        "eye_y": eye_center[1]
    }


def classify_face_region_pose(hand_landmarks, anchors):
    """
    Classify whether hand is near a facial region and return the specific HamNoSys token.
    Uses index tip (8) and index MCP (5) for dual-point proximity.
    """
    idx_tip = np.array([hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y])
    idx_mcp = np.array([hand_landmarks.landmark[5].x, hand_landmarks.landmark[5].y])
    wrist = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y])

    # Contact point: average of index tip and index MCP for stable region classification
    contact_pt = (idx_tip * 0.7 + idx_mcp * 0.3)
    scale = anchors["scale"]

    # Hand must be within face bounding sphere (within 3.0 * face_scale of nose)
    dist_nose = np.linalg.norm(contact_pt - anchors["nose"][:2])
    if dist_nose > scale * 3.2 and wrist[1] > anchors["chin"][1] + scale * 0.5:
        return None  # Not near face

    # Compute distances to anchors
    d_chin = np.linalg.norm(contact_pt - anchors["chin"][:2])
    d_underchin = np.linalg.norm(contact_pt - anchors["underchin"][:2])
    d_mouth = np.linalg.norm(contact_pt - anchors["mouth"][:2])
    d_nose = np.linalg.norm(contact_pt - anchors["nose"][:2])
    d_forehead = np.linalg.norm(contact_pt - anchors["forehead"][:2])
    d_headtop = np.linalg.norm(contact_pt - anchors["headtop"][:2])
    d_cheek = min(np.linalg.norm(contact_pt - anchors["cheek_l"][:2]),
                  np.linalg.norm(contact_pt - anchors["cheek_r"][:2]))
    d_ear = min(np.linalg.norm(contact_pt - anchors["ear_l"][:2]),
                np.linalg.norm(contact_pt - anchors["ear_r"][:2]))

    # Prioritized region classification
    if d_chin < scale * 1.5:
        return "hamchin"
    elif d_underchin < scale * 1.3 and contact_pt[1] > anchors["chin"][1]:
        return "hamunderchin"
    elif d_mouth < scale * 1.2:
        return "hamlips"
    elif d_nose < scale * 1.1:
        return "hamnose"
    elif d_forehead < scale * 1.4:
        return "hamforehead"
    elif d_headtop < scale * 1.5 and contact_pt[1] < anchors["forehead"][1]:
        return "hamheadtop"
    elif d_cheek < scale * 1.3:
        return "hamcheek"
    elif d_ear < scale * 1.4:
        return "hamear"

    # Default vertical zone fallback if close to face midline
    rel_y = contact_pt[1]
    if rel_y > anchors["chin"][1] - scale * 0.3:
        return "hamchin"
    elif rel_y > anchors["mouth_y"] - scale * 0.2:
        return "hamlips"
    elif rel_y > anchors["eye_y"]:
        return "hamnose"
    elif rel_y < anchors["forehead"][1]:
        return "hamforehead"

    return "hamcheek"


def run_head_face_location_module(video_path):
    """
    Executes head and face region tracking across video.
    Returns detected face regions and overall majority token.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"per_frame": [], "final": None}

    predictions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        p_res = pose.process(rgb)
        h_res = hands.process(rgb)

        if p_res.pose_landmarks and h_res.multi_hand_landmarks:
            anchors = get_face_anchors_from_pose(p_res.pose_landmarks)
            # Pick active signing hand (highest in frame, lowest y)
            active_hand = min(h_res.multi_hand_landmarks, key=lambda h: h.landmark[0].y)

            loc = classify_face_region_pose(active_hand, anchors)
            if loc:
                predictions.append(loc)

    cap.release()

    if not predictions:
        return {"per_frame": [], "final": None}

    final_location = Counter(predictions).most_common(1)[0][0]
    return {
        "per_frame": predictions,
        "final": final_location
    }
