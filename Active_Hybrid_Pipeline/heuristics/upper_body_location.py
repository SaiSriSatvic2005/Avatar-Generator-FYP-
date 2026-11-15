#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
from google.colab import drive
drive.mount('/content/drive')
'''


# In[ ]:


#!pip install mediapipe==0.10.21


# In[ ]:


#!pip install mediapipe opencv-python numpy


# In[ ]:


import cv2
import numpy as np
import mediapipe as mp
from collections import Counter

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# In[ ]:


def detect_upper_body_location(video_path):

    cap = cv2.VideoCapture(video_path)
    preds = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        pose_pts = get_pose_points(frame_rgb, w, h)
        if pose_pts is None:
            continue

        lhc, rhc = get_hand_center_from_pose(pose_pts)

        label_l = classify_upper_body_location(pose_pts, lhc, "left")
        label_r = classify_upper_body_location(pose_pts, rhc, "right")

        if label_l != "unknown":
            preds.append(label_l)
        elif label_r != "unknown":
            preds.append(label_r)

    cap.release()

    if not preds:
        return None

    from collections import Counter
    return Counter(preds).most_common(1)[0][0]


# In[ ]:


import math

def norm_dist(p1, p2, scale):
    """
    Normalized Euclidean distance
    p1, p2 → (x, y)
    scale → normalization factor (example: shoulder width)
    """
    if scale == 0:
        return 999  # prevents division errors

    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) / scale


# In[ ]:


def dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


# In[ ]:


def get_pose_points(frame_rgb, w, h):
    res = pose.process(frame_rgb)
    if not res.pose_landmarks:
        return None

    lm = res.pose_landmarks.landmark

    def P(i):
        return int(lm[i].x * w), int(lm[i].y * h)

    return {
        "l_shoulder": P(mp_pose.PoseLandmark.LEFT_SHOULDER),
        "r_shoulder": P(mp_pose.PoseLandmark.RIGHT_SHOULDER),
        "l_elbow":    P(mp_pose.PoseLandmark.LEFT_ELBOW),
        "r_elbow":    P(mp_pose.PoseLandmark.RIGHT_ELBOW),
        "l_wrist":    P(mp_pose.PoseLandmark.LEFT_WRIST),
        "r_wrist":    P(mp_pose.PoseLandmark.RIGHT_WRIST),
        "l_hip":      P(mp_pose.PoseLandmark.LEFT_HIP),
        "r_hip":      P(mp_pose.PoseLandmark.RIGHT_HIP),
    }


# In[ ]:


def get_hand_center_from_pose(pose_pts):
    lw = pose_pts["l_wrist"]
    rw = pose_pts["r_wrist"]
    le = pose_pts["l_elbow"]
    re = pose_pts["r_elbow"]

    left_hand_center  = (lw[0], lw[1] + (lw[1] - le[1]) // 3)
    right_hand_center = (rw[0], rw[1] + (rw[1] - re[1]) // 3)

    return left_hand_center, right_hand_center


# In[ ]:


def classify_upper_body_location(pose_pts, contact, side):
    cx, cy = contact

    ls = pose_pts["l_shoulder"]
    rs = pose_pts["r_shoulder"]
    le = pose_pts["l_elbow"]
    re = pose_pts["r_elbow"]
    lw = pose_pts["l_wrist"]
    rw = pose_pts["r_wrist"]
    lh = pose_pts["l_hip"]
    rh = pose_pts["r_hip"]

    shoulder_y = (ls[1] + rs[1]) // 2
    hip_y = (lh[1] + rh[1]) // 2
    torso_center_x = (ls[0] + rs[0]) // 2
    shoulder_width = abs(ls[0] - rs[0])

    # =========================
    # 1️⃣ SHOULDER (PRIORITY)
    # =========================
    if dist((cx, cy), ls) < 50 or dist((cx, cy), rs) < 50:
        return "hamshouldertop" if cy < shoulder_y else "hamshoulders"

    # =========================
    # 2️⃣ ELBOW
    # =========================
    if side == "left" and dist((cx, cy), le) < 40:
        return "hamelbowinside"
    if side == "right" and dist((cx, cy), re) < 40:
        return "hamelbowinside"

    # =========================
    # 3️⃣ ARM (USE LATERAL DISTANCE)
    # =========================
    lateral_far = abs(cx - torso_center_x) > 0.25 * shoulder_width

    if side == "left":
        if ls[1] < cy < le[1] and lateral_far:
            return "hamupperarm"
        if le[1] < cy < lw[1] and lateral_far:
            return "hamlowerarm"
    else:
        if rs[1] < cy < re[1] and lateral_far:
            return "hamupperarm"
        if re[1] < cy < rw[1] and lateral_far:
            return "hamlowerarm"

    # =========================
    # 4️⃣ WRIST BACK (STRICT)
    # =========================
    if side == "left":
        if dist((cx, cy), lw) < 25 and dist((cx, cy), le) > 50:
            return "hamwristback"
    else:
        if dist((cx, cy), rw) < 25 and dist((cx, cy), re) > 50:
            return "hamwristback"

    # =========================
    # 5️⃣ TORSO (LAST)
    # =========================
    if shoulder_y < cy < hip_y:
        rel = (cy - shoulder_y) / (hip_y - shoulder_y)

        if rel < 0.35:
            return "hamchest"
        elif rel < 0.70:
            return "hamstomach"
        else:
            return "hambelowstomach"

    return "unknown"


# In[ ]:


def classify_upper_body_location(pose_pts, contact, side):
    cx, cy = contact

    ls = pose_pts["l_shoulder"]
    rs = pose_pts["r_shoulder"]
    le = pose_pts["l_elbow"]
    re = pose_pts["r_elbow"]
    lw = pose_pts["l_wrist"]
    rw = pose_pts["r_wrist"]
    lh = pose_pts["l_hip"]
    rh = pose_pts["r_hip"]

    # =========================
    # BODY SCALES (REFERENCE)
    # =========================
    shoulder_width = dist(ls, rs)
    torso_height = (lh[1] + rh[1]) / 2 - (ls[1] + rs[1]) / 2
    torso_center_x = (ls[0] + rs[0]) / 2
    shoulder_y = (ls[1] + rs[1]) / 2
    hip_y = (lh[1] + rh[1]) / 2

    # =========================
    # 1️⃣ SHOULDER (PRIORITY)
    # 20–25% of shoulder width
    # =========================
    if (
        norm_dist((cx, cy), ls, shoulder_width) < 0.25 or
        norm_dist((cx, cy), rs, shoulder_width) < 0.25
    ):
        return "hamshouldertop" if cy < shoulder_y else "hamshoulders"

    # =========================
    # 2️⃣ ELBOW
    # ~20% of upper arm length
    # =========================
    if side == "left":
        upper_arm_len = dist(ls, le)
        if norm_dist((cx, cy), le, upper_arm_len) < 0.20:
            return "hamelbowinside"
    else:
        upper_arm_len = dist(rs, re)
        if norm_dist((cx, cy), re, upper_arm_len) < 0.20:
            return "hamelbowinside"

    # =========================
    # 3️⃣ ARM (LATERAL SEPARATION)
    # Outside torso silhouette
    # =========================
    lateral_far = abs(cx - torso_center_x) > 0.25 * shoulder_width

    if side == "left":
        if ls[1] < cy < le[1] and lateral_far:
            return "hamupperarm"
        if le[1] < cy < lw[1] and lateral_far:
            return "hamlowerarm"
    else:
        if rs[1] < cy < re[1] and lateral_far:
            return "hamupperarm"
        if re[1] < cy < rw[1] and lateral_far:
            return "hamlowerarm"

    # =========================
    # 4️⃣ WRIST BACK
    # ~12–15% of forearm length
    # =========================
    if side == "left":
        forearm_len = dist(le, lw)
        if (
            norm_dist((cx, cy), lw, forearm_len) < 0.15 and
            norm_dist((cx, cy), le, forearm_len) > 0.40
        ):
            return "hamwristback"
    else:
        forearm_len = dist(re, rw)
        if (
            norm_dist((cx, cy), rw, forearm_len) < 0.15 and
            norm_dist((cx, cy), re, forearm_len) > 0.40
        ):
            return "hamwristback"

    # =========================
    # 5️⃣ TORSO (NORMALIZED)
    # =========================
    if shoulder_y < cy < hip_y:
        rel = (cy - shoulder_y) / torso_height

        if rel < 0.35:
            return "hamchest"
        elif rel < 0.70:
            return "hamstomach"
        else:
            return "hambelowstomach"

    return "unknown"


# In[ ]:


def predict_upper_body_location(video_path):
    cap = cv2.VideoCapture(video_path)
    preds = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        pose_pts = get_pose_points(frame_rgb, w, h)
        if pose_pts is None:
            continue

        lhc, rhc = get_hand_center_from_pose(pose_pts)

        label_l = classify_upper_body_location(pose_pts, lhc, "left")
        label_r = classify_upper_body_location(pose_pts, rhc, "right")

        if label_l != "unknown":
            preds.append(label_l)
        elif label_r != "unknown":
            preds.append(label_r)

    cap.release()

    if not preds:
        return "No contact detected", 0.0

    final, count = Counter(preds).most_common(1)[0]
    return final, round(count / len(preds), 2)


# In[ ]:


# Top-level demo code guarded for import safety
if __name__ == "__main__":
    label, conf = predict_upper_body_location("Prompt_1.mp4")
    print("Predicted:", label)
    print("Confidence:", conf)


# In[ ]:


def detect_upper_body_location(frame):

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape

    pose_pts = get_pose_points(frame_rgb, w, h)
    if pose_pts is None:
        return None

    lhc, rhc = get_hand_center_from_pose(pose_pts)

    label_l = classify_upper_body_location(pose_pts, lhc, "left")
    label_r = classify_upper_body_location(pose_pts, rhc, "right")

    if label_l != "unknown":
        return label_l

    if label_r != "unknown":
        return label_r

    return None


# In[ ]:


def run_upper_body_location_module(video_path):

    predictions = []

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {"per_frame": [], "final": None}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        loc = detect_upper_body_location(frame)   # ✅ FIXED

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


# In[ ]:




