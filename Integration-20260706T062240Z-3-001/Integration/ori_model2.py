#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''from google.colab import drive

drive.mount('/content/drive')
'''


# In[ ]:


#!pip install mediapipe


# In[ ]:


import numpy as np
from math import atan2, degrees
from collections import Counter

# HAMNOSYS CLASS SETS

SIGNER_CLASSES = [
    "hamextfingeru","hamextfingerur","hamextfingerr","hamextfingerdr",
    "hamextfingerd","hamextfingerul","hamextfingerl","hamextfingerdl"
]

BIRD_CLASSES = [
    "hamextfingero","hamextfingeror","hamextfingerr","hamextfingerir",
    "hamextfingeri","hamextfingerol","hamextfingerl","hamextfingeril"
]

RIGHT_CLASSES = [
    "hamextfingeru","hamextfingeruo","hamextfingero","hamextfingerdo",
    "hamextfingerd","hamextfingerui","hamextfingeri","hamextfingerdi"
]

PALM_CLASSES = [
    "hampalmu","hampalmur","hampalmr","hampalmdr",
    "hampalmd","hampalmul","hampalml","hampalmdl"
]


class ID3HandTree:

    # ---------------------- VIEW CLASSIFIER ----------------------
    def classify_view(self, wrist, eye_avg, right_shoulder):
        # wrist, eye_avg, right_shoulder are 2D (x,y)
        if wrist[1] < eye_avg[1] - 0.15:
            return "bird"
        elif wrist[0] < right_shoulder[0] - 0.05:
            return "right"
        else:
            return "signer"

    # --------------------- ANGLE CLASSIFIER ----------------------
    def angle_to_dir(self, angle):
        # Convert Viewer Screen direction to Signer Anatomical direction:
        # Viewer Right (angle 0) = Signer Left ("l")
        # Viewer Up (angle 90) = Signer Up ("u")
        # Viewer Left (angle 180) = Signer Right ("r")
        # Viewer Down (angle 270) = Signer Down ("d")
        if 0 <= angle < 22 or 337 <= angle <= 360: return "l"
        if 22 <= angle < 67: return "ul"
        if 67 <= angle < 112: return "u"
        if 112 <= angle < 157: return "ur"
        if 157 <= angle < 202: return "r"
        if 202 <= angle < 247: return "dr"
        if 247 <= angle < 292: return "d"
        return "dl"

    # ---------------------- FINGER CLASSIFIER ----------------------
    def classify_finger(self, view, wrist, tip):

        vx = tip[0] - wrist[0]
        vy = tip[1] - wrist[1]
        vz = tip[2] - wrist[2]

        # FORCE correct outward for right-view left pointing
        if view == "right" and vx < -0.05:
            return "hamextfingero"

        # fallback original angle logic
        angle = degrees(atan2(-vy, vx)) % 360
        base_dir = self.angle_to_dir(angle)

        if vz > 0.08:
            tilt = "i"
        elif vz < -0.08:
            tilt = "o"
        else:
            tilt = ""

        candidate = f"hamextfinger{base_dir}{tilt}"

        if view == "signer":
            allowed = SIGNER_CLASSES
        elif view == "bird":
            allowed = BIRD_CLASSES
        else:
            allowed = RIGHT_CLASSES

        if candidate in allowed:
            return candidate

        fallback = f"hamextfinger{base_dir}"
        if fallback in allowed:
            return fallback

        return allowed[0]

    # ---------------------- PALM CLASSIFIER (8-DIRECTIONS) ----------------------
    def classify_palm(self, wrist, index_mcp, pinky_mcp):
        # Palm plane vectors
        v1 = index_mcp - wrist
        v2 = pinky_mcp - wrist
        normal = np.cross(v1, v2)

        # In Mediapipe:
        # x → right, y → down, z → toward camera
        # We convert to a consistent 2D system:
        nx = normal[0]
        ny = -normal[1]   # invert because screen Y is reversed

        # If vector too small → fallback
        if abs(nx) < 1e-6 and abs(ny) < 1e-6:
            return "hampalmd"

        # Compute angle in degrees
        angle = (degrees(atan2(ny, nx)) + 360) % 360

        # Map angle to 8 palm orientations (Viewer to Signer frame)
        if   337 <= angle or angle < 22:    return "hampalml"    # Viewer Right -> Signer Left
        elif 22 <= angle < 67:              return "hampalmul"   # Viewer Up-Right -> Signer Up-Left
        elif 67 <= angle < 112:             return "hampalmu"    # Viewer Up -> Signer Up
        elif 112 <= angle < 157:            return "hampalmur"   # Viewer Up-Left -> Signer Up-Right
        elif 157 <= angle < 202:            return "hampalmr"    # Viewer Left -> Signer Right
        elif 202 <= angle < 247:            return "hampalmdr"   # Viewer Down-Left -> Signer Down-Right
        elif 247 <= angle < 292:            return "hampalmd"    # Viewer Down -> Signer Down
        else:                               return "hampalmdl"   # Viewer Down-Right -> Signer Down-Left


# In[ ]:


import random

tree = ID3HandTree()

def generate_synthetic():

    target_view = random.choice(["bird", "right", "signer"])

    # realistic eye position
    eye_avg = np.array([
        np.random.uniform(0.42, 0.58),
        np.random.uniform(0.18, 0.28)
    ])

    # realistic shoulder position
    right_shoulder = np.array([
        eye_avg[0] + np.random.uniform(0.08, 0.15),
        eye_avg[1] + np.random.uniform(0.20, 0.30)
    ])

    # realistic wrist for each view
    if target_view == "bird":
        # wrist clearly ABOVE eyes
        wrist = np.array([
            np.random.uniform(0.35, 0.65),
            eye_avg[1] - np.random.uniform(0.12, 0.22),
            -0.1
        ])

    elif target_view == "right":
        # wrist to the LEFT of right shoulder, around shoulder height
        wrist = np.array([
            right_shoulder[0] - np.random.uniform(0.08, 0.20),
            right_shoulder[1] - np.random.uniform(-0.05, 0.12),  # sometimes above, sometimes below
            -0.1
        ])

    else:  # signer
        # wrist to the RIGHT of right shoulder (near signer POV)
        wrist = np.array([
            right_shoulder[0] + np.random.uniform(0.05, 0.12),
            right_shoulder[1] - np.random.uniform(-0.05, 0.12),
            -0.1
        ])

    # MCPs
    index_mcp = wrist + np.array([
        np.random.uniform(0.06, 0.15),
        np.random.uniform(-0.10, 0.05),
        np.random.uniform(-0.08, 0.05)
    ])

    pinky_mcp = wrist + np.array([
        np.random.uniform(-0.15, -0.06),
        np.random.uniform(-0.10, 0.05),
        np.random.uniform(-0.08, 0.05)
    ])

    # fingertip direction
    theta = np.random.uniform(0, 2*np.pi)
    phi = np.random.uniform(-0.6, 0.6)
    r = np.random.uniform(0.15, 0.45)

    tip = wrist + np.array([
        np.cos(theta)*np.cos(phi),
        np.sin(phi),
        np.sin(theta)*np.cos(phi)
    ]) * r

    # TEACHER labels
    view_label = tree.classify_view(wrist[:2], eye_avg, right_shoulder)
    finger_label = tree.classify_finger(view_label, wrist, tip)
    palm_label = tree.classify_palm(wrist, index_mcp, pinky_mcp)

    return wrist, tip, index_mcp, pinky_mcp, eye_avg, right_shoulder, view_label, finger_label, palm_label



# In[ ]:


import joblib
import os

# =====================================================
# LOAD PRE-TRAINED MODELS (or retrain if missing)
# =====================================================

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

_PKL_FILES = [
    "clf_view", "enc_view",
    "clf_finger_signer", "enc_finger_signer",
    "clf_finger_bird", "enc_finger_bird",
    "clf_finger_right", "enc_finger_right",
    "clf_palm", "enc_palm",
]

def _all_pkls_exist():
    return all(
        os.path.exists(os.path.join(_MODEL_DIR, f"{name}.pkl"))
        for name in _PKL_FILES
    )

def _load_pkl(name):
    return joblib.load(os.path.join(_MODEL_DIR, f"{name}.pkl"))

if _all_pkls_exist():
    # Fast path: load pre-trained models (< 1 second)
    clf_view = _load_pkl("clf_view")
    enc_view = _load_pkl("enc_view")
    clf_finger_signer = _load_pkl("clf_finger_signer")
    enc_finger_signer = _load_pkl("enc_finger_signer")
    clf_finger_bird = _load_pkl("clf_finger_bird")
    enc_finger_bird = _load_pkl("enc_finger_bird")
    clf_finger_right = _load_pkl("clf_finger_right")
    enc_finger_right = _load_pkl("enc_finger_right")
    clf_palm = _load_pkl("clf_palm")
    enc_palm = _load_pkl("enc_palm")
else:
    # Slow path: generate synthetic data and train (takes minutes)
    print("[ori_model2] Pre-trained .pkl files not found — training from scratch...")

    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import RandomForestClassifier

    X_view, y_view = [], []
    X_finger_signer, y_finger_signer = [], []
    X_finger_bird, y_finger_bird = [], []
    X_finger_right, y_finger_right = [], []
    X_palm, y_palm = [], []

    N = 60000

    for _ in range(N):
        wrist, tip, i_mcp, p_mcp, eye, shoulder, v_lbl, f_lbl, p_lbl = generate_synthetic()

        fv_view = np.concatenate([wrist[:2] - eye, wrist[:2] - shoulder])
        X_view.append(fv_view)
        y_view.append(v_lbl)

        fv_hand = np.concatenate([tip - wrist, i_mcp - wrist, p_mcp - wrist])

        if v_lbl == "signer":
            X_finger_signer.append(fv_hand)
            y_finger_signer.append(f_lbl)
        elif v_lbl == "bird":
            X_finger_bird.append(fv_hand)
            y_finger_bird.append(f_lbl)
        else:
            X_finger_right.append(fv_hand)
            y_finger_right.append(f_lbl)

        X_palm.append(fv_hand)
        y_palm.append(p_lbl)

    enc_view = LabelEncoder()
    y_view_enc = enc_view.fit_transform(y_view)
    clf_view = RandomForestClassifier(n_estimators=350, n_jobs=-1)
    clf_view.fit(X_view, y_view_enc)

    enc_finger_signer = LabelEncoder()
    clf_finger_signer = RandomForestClassifier(n_estimators=350, n_jobs=-1)
    clf_finger_signer.fit(X_finger_signer, enc_finger_signer.fit_transform(y_finger_signer))

    enc_finger_bird = LabelEncoder()
    clf_finger_bird = RandomForestClassifier(n_estimators=350, n_jobs=-1)
    clf_finger_bird.fit(X_finger_bird, enc_finger_bird.fit_transform(y_finger_bird))

    enc_finger_right = LabelEncoder()
    clf_finger_right = RandomForestClassifier(n_estimators=350, n_jobs=-1)
    clf_finger_right.fit(X_finger_right, enc_finger_right.fit_transform(y_finger_right))

    enc_palm = LabelEncoder()
    clf_palm = RandomForestClassifier(n_estimators=350, n_jobs=-1)
    clf_palm.fit(X_palm, enc_palm.fit_transform(y_palm))

    # Save for next time
    for name, obj in [
        ("clf_view", clf_view), ("enc_view", enc_view),
        ("clf_finger_signer", clf_finger_signer), ("enc_finger_signer", enc_finger_signer),
        ("clf_finger_bird", clf_finger_bird), ("enc_finger_bird", enc_finger_bird),
        ("clf_finger_right", clf_finger_right), ("enc_finger_right", enc_finger_right),
        ("clf_palm", clf_palm), ("enc_palm", enc_palm),
    ]:
        joblib.dump(obj, os.path.join(_MODEL_DIR, f"{name}.pkl"))

    print("[ori_model2] Models trained and saved.")


# In[ ]:


import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

def extract_landmarks(path):
    img = cv2.imread(path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(static_image_mode=True) as pose:
        p_res = pose.process(rgb)

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        h_res = hands.process(rgb)

    if not p_res.pose_landmarks or not h_res.multi_hand_landmarks:
        return None

    p = p_res.pose_landmarks.landmark
    h = h_res.multi_hand_landmarks[0].landmark

    left_eye = np.array([p[2].x,p[2].y])
    right_eye = np.array([p[5].x,p[5].y])
    eye_avg = (left_eye + right_eye)/2

    shoulder = np.array([p[12].x, p[12].y])

    wrist = np.array([h[0].x,h[0].y,h[0].z])
    tip = np.array([h[8].x,h[8].y,h[8].z])
    i_mcp = np.array([h[5].x,h[5].y,h[5].z])
    p_mcp = np.array([h[17].x,h[17].y,h[17].z])

    return wrist, tip, i_mcp, p_mcp, eye_avg, shoulder


# In[ ]:


def ml_predict(path):

    L = extract_landmarks(path)
    if L is None:
        print("No landmarks detected")
        return None

    wrist, tip, i_mcp, p_mcp, eye, shoulder = L

    # ---- View features ----
    fv_view = np.concatenate([
        wrist[:2] - eye,
        wrist[:2] - shoulder
    ]).reshape(1,-1)

    view_idx = clf_view.predict(fv_view)[0]
    view = enc_view.inverse_transform([view_idx])[0]

    # ---- Hand feature ----
    fv_hand = np.concatenate([
        tip - wrist, i_mcp - wrist, p_mcp - wrist
    ]).reshape(1,-1)

    # ---- Finger based on view ----
    if view == "signer":
        f_idx = clf_finger_signer.predict(fv_hand)[0]
        finger = enc_finger_signer.inverse_transform([f_idx])[0]

    elif view == "bird":
        f_idx = clf_finger_bird.predict(fv_hand)[0]
        finger = enc_finger_bird.inverse_transform([f_idx])[0]

    else: # right
        f_idx = clf_finger_right.predict(fv_hand)[0]
        finger = enc_finger_right.inverse_transform([f_idx])[0]

    # ---- Palm ----
    p_idx = clf_palm.predict(fv_hand)[0]
    palm = enc_palm.inverse_transform([p_idx])[0]

    return view, finger, palm


# In[ ]:


import tempfile
import os
import cv2

def calculate_orientation_from_landmarks(h_landmarks, handedness="Right", pose_landmarks=None):
    wrist = np.array([h_landmarks[0].x, h_landmarks[0].y, h_landmarks[0].z])
    index_mcp = np.array([h_landmarks[5].x, h_landmarks[5].y, h_landmarks[5].z])
    middle_tip = np.array([h_landmarks[12].x, h_landmarks[12].y, h_landmarks[12].z])
    pinky_mcp = np.array([h_landmarks[17].x, h_landmarks[17].y, h_landmarks[17].z])

    view = "signer"
    if pose_landmarks is not None:
        if isinstance(pose_landmarks, dict):
            left_eye = np.array(pose_landmarks.get("left_eye", (0.46, 0.22)), dtype=float)
            right_eye = np.array(pose_landmarks.get("right_eye", (0.54, 0.22)), dtype=float)
            right_shoulder = np.array(pose_landmarks.get("right_shoulder", (0.62, 0.40)), dtype=float)
            eye_avg = (left_eye + right_eye) / 2.0
            if wrist[1] < eye_avg[1] - 0.12:
                view = "bird"
            elif wrist[0] < right_shoulder[0] - 0.03:
                view = "right"
            else:
                view = "signer"

    # 1. Extended Finger Direction (Use index tip landmark 8 if extended, or max-extended tip)
    tips = [8, 12, 16, 20]
    tip_dists = [np.linalg.norm(np.array([h_landmarks[t].x - wrist[0], h_landmarks[t].y - wrist[1], h_landmarks[t].z - wrist[2]])) for t in tips]
    best_tip_idx = tips[np.argmax(tip_dists)]
    best_tip = np.array([h_landmarks[best_tip_idx].x, h_landmarks[best_tip_idx].y, h_landmarks[best_tip_idx].z])

    finger_vec = best_tip - wrist
    fx, fy, fz = finger_vec[0], -finger_vec[1], finger_vec[2]

    if view == "right":
        fx = -fx
    elif view == "bird":
        fy = -fy

    if abs(fy) >= abs(fx):
        finger_tag = "hamextfingeru" if fy > 0 else "hamextfingerd"
    else:
        finger_tag = "hamextfingerr" if fx > 0 else "hamextfingerl"



    # 2. Robust Palm Normal using properly scaled Z coordinates
    # MediaPipe Z is roughly scaled to the hand's bounding box size, which is smaller than screen coords.
    # We should normalize the 3D vectors first before cross product to ensure axes have equal weight.
    v1 = index_mcp - wrist
    v2 = pinky_mcp - wrist

    # Invert for Left hand so logic is consistent
    if handedness == "Left":
        v1, v2 = v2, v1

    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)

    if v1_norm > 0 and v2_norm > 0:
        v1 = v1 / v1_norm
        v2 = v2 / v2_norm

    nz = v1[0] * v2[1] - v1[1] * v2[0]
    nx = v1[1] * v2[2] - v1[2] * v2[1]
    ny = v1[2] * v2[0] - v1[0] * v2[2]

    # Normalize the resulting normal
    norm_val = np.linalg.norm([nx, ny, nz])
    if norm_val > 0:
        nx, ny, nz = nx / norm_val, ny / norm_val, nz / norm_val

    # ny is inverted for screen coordinates
    ny = -ny

    # Strict mappings based on Finger Direction
    # In HamNoSys & JASigning avatar notation:
    # hampalmd = Palm facing OUTWARD towards camera / viewer
    # hampalmu = Palm facing INWARD towards signer
    # hampalmr = Palm facing RIGHT
    # hampalml = Palm facing LEFT
    if finger_tag in ["hamextfingeru", "hamextfingerd"]:
        projs = {
            'hampalmd': -nz,
            'hampalmu': nz,
            'hampalmr': nx,
            'hampalml': -nx
        }
    else:
        projs = {
            'hampalmd': -nz if abs(nz) > abs(ny) else -ny,
            'hampalmu': nz if abs(nz) > abs(ny) else ny,
            'hampalmr': nx,
            'hampalml': -nx
        }


    palm_tag = max(projs, key=projs.get)

    return (view, finger_tag, palm_tag)


def run_orientation_module(video_path):
    try:
        from shared_landmarks import get_video_landmarks
        frames = get_video_landmarks(video_path)
        views, fingers, palms = [], [], []
        for f in frames:
            hand = f.get("right_hand") or f.get("left_hand")
            pose = f.get("pose_landmarks")
            if hand:
                h_landmarks = hand.landmark
                handedness = "Right" if f.get("right_hand") else "Left"
                view, finger, palm = calculate_orientation_from_landmarks(h_landmarks, handedness, pose_landmarks=pose)
                views.append(view)
                fingers.append(finger)
                palms.append(palm)
        if views:
            final_view   = Counter(views).most_common(1)[0][0]
            final_finger = Counter(fingers).most_common(1)[0][0]
            final_palm   = Counter(palms).most_common(1)[0][0]
            return {
                "per_frame": list(zip(views, fingers, palms)),
                "final": (final_view, final_finger, final_palm)
            }
    except Exception as e:
        pass

    views, fingers, palms = [], [], []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"per_frame": [], "final": None}

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            if not res.multi_hand_landmarks:
                continue
                
            handedness = res.multi_handedness[0].classification[0].label
            h_landmarks = res.multi_hand_landmarks[0].landmark
            view, finger, palm = calculate_orientation_from_landmarks(h_landmarks, handedness)

            views.append(view)
            fingers.append(finger)
            palms.append(palm)

    cap.release()

    if not views:
        return {"per_frame": [], "final": None}

    final_view   = Counter(views).most_common(1)[0][0]
    final_finger = Counter(fingers).most_common(1)[0][0]
    final_palm   = Counter(palms).most_common(1)[0][0]

    return {
        "per_frame": list(zip(views, fingers, palms)),
        "final": (final_view, final_finger, final_palm)
    }


# In[ ]:


