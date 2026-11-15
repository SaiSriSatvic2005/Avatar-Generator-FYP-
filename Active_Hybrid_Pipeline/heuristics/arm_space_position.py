#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


#pip install mediapipe opencv-python numpy


# In[ ]:


import cv2
import mediapipe as mp
import numpy as np
import math


# In[ ]:


from collections import Counter


# In[ ]:


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


# In[ ]:


import mediapipe as mp
#print(mp)


# In[ ]:


#!pip install mediapipe==0.10.9


# In[ ]:


#!pip uninstall mediapipe -y
#!pip install mediapipe==0.10.21


# In[ ]:


import mediapipe as mp
#print(mp.__version__)
#print(hasattr(mp, "solutions"))

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
#print("POSE OK")


# In[ ]:


import math
import numpy as np
def dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
'''
def dist(a, b):
    return math.sqrt(
        (a[0] - b[0])**2 +
        (a[1] - b[1])**2 +
        (a[2] - b[2])**2
    )

def angle(a, b, c):
    # angle ABC
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)

    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
    '''


# In[ ]:


import mediapipe as mp

mp_pose = mp.solutions.pose


# In[ ]:


'''
def get_pose_points(results):
    lm = results.pose_landmarks.landmark

    pts = {
        "ls": (lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
               lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
               lm[mp_pose.PoseLandmark.LEFT_SHOULDER].z),

        "le": (lm[mp_pose.PoseLandmark.LEFT_ELBOW].x,
               lm[mp_pose.PoseLandmark.LEFT_ELBOW].y,
               lm[mp_pose.PoseLandmark.LEFT_ELBOW].z),

        "lw": (lm[mp_pose.PoseLandmark.LEFT_WRIST].x,
               lm[mp_pose.PoseLandmark.LEFT_WRIST].y,
               lm[mp_pose.PoseLandmark.LEFT_WRIST].z),

        "rs": (lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
               lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
               lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].z),

        "re": (lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
               lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y,
               lm[mp_pose.PoseLandmark.RIGHT_ELBOW].z),

        "rw": (lm[mp_pose.PoseLandmark.RIGHT_WRIST].x,
               lm[mp_pose.PoseLandmark.RIGHT_WRIST].y,
               lm[mp_pose.PoseLandmark.RIGHT_WRIST].z),
    }

    pts["torso"] = (
        (pts["ls"][0] + pts["rs"][0]) / 2,
        (pts["ls"][1] + pts["rs"][1]) / 2,
        (pts["ls"][2] + pts["rs"][2]) / 2,
    )

    return pts
'''


# In[ ]:


def get_pose_points(results):

    if not results.pose_landmarks:
        return None

    lm = results.pose_landmarks.landmark

    pts = {
        "ls": (lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
               lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
               lm[mp_pose.PoseLandmark.LEFT_SHOULDER].z),

        "le": (lm[mp_pose.PoseLandmark.LEFT_ELBOW].x,
               lm[mp_pose.PoseLandmark.LEFT_ELBOW].y,
               lm[mp_pose.PoseLandmark.LEFT_ELBOW].z),

        "lw": (lm[mp_pose.PoseLandmark.LEFT_WRIST].x,
               lm[mp_pose.PoseLandmark.LEFT_WRIST].y,
               lm[mp_pose.PoseLandmark.LEFT_WRIST].z),

        "rs": (lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
               lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
               lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].z),

        "re": (lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
               lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y,
               lm[mp_pose.PoseLandmark.RIGHT_ELBOW].z),

        "rw": (lm[mp_pose.PoseLandmark.RIGHT_WRIST].x,
               lm[mp_pose.PoseLandmark.RIGHT_WRIST].y,
               lm[mp_pose.PoseLandmark.RIGHT_WRIST].z),
    }

    pts["torso"] = (
        (pts["ls"][0] + pts["rs"][0]) / 2,
        (pts["ls"][1] + pts["rs"][1]) / 2,
        (pts["ls"][2] + pts["rs"][2]) / 2,
    )

    return pts


# In[ ]:


def compute_features(pts, side="L"):
    if side == "L":
        s, e, w = pts["ls"], pts["le"], pts["lw"]
    else:
        s, e, w = pts["rs"], pts["re"], pts["rw"]

    return {
        "elbow_angle": angle(s, e, w),
        "extension_ratio": dist(s, w) / (dist(s, e) + 1e-6),
        "x_offset": abs(w[0] - pts["torso"][0]),
        "z_offset": abs(w[2] - pts["torso"][2]),
        "wrist_y": w[1],
        "shoulder_y": s[1]
    }


# In[ ]:


MIDLINE_EPS = 0.04
LATERAL_THRESH = 0.15
Z_FRONT_THRESH = 0.05


# In[ ]:


def is_arm_extended(f):
    return f["elbow_angle"] > 135 and f["extension_ratio"] > 1.8

def is_midline(f):
    return f["x_offset"] < MIDLINE_EPS

def is_beside(f):
    return f["x_offset"] > LATERAL_THRESH and \
           f["shoulder_y"] < f["wrist_y"] < f["shoulder_y"] + 0.25

def is_double_bent(fL, fR):
    return fL["elbow_angle"] < 120 and fR["elbow_angle"] < 120

def is_neutral_space(f, contact=False):
    return (not contact) and \
           f["z_offset"] > Z_FRONT_THRESH and \
           f["shoulder_y"] < f["wrist_y"] < f["shoulder_y"] + 0.3

def is_double_hooked(handshape_L, handshape_R):
    return handshape_L == "hooked" and handshape_R == "hooked"


# In[ ]:


def classify_F_class(pts, contact=False):
    fL = compute_features(pts, "L")
    fR = compute_features(pts, "R")

    # 1️⃣ Double-arm first
    if is_double_bent(fL, fR):
        return "hamdoublebent"

    # 2️⃣ Choose dominant arm (farther from midline)
    f = fL if fL["x_offset"] > fR["x_offset"] else fR

    if is_arm_extended(f):
        return "hamarmextended"

    if is_midline(f):
        return "hamlrat"

    # ✅ hamlrbeside (relaxed realistic version)
    if (
        f["x_offset"] > 0.10 and
        f["z_offset"] < 0.08 and
        f["wrist_y"] < 0.85
    ):
        return "hamlrbeside"

    if is_neutral_space(f, contact):
        return "hamneutralspace"

    return "none"


# In[ ]:


'''
from google.colab import drive
drive.mount('/content/drive')
'''


# In[ ]:


import cv2

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Top-level demo code guarded for import safety
if __name__ == "__main__":
    import cv2

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    video_path = "Prompt_1.mp4"
    output_path = "arm_space_output.mp4"

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    ret, frame = cap.read()
    if not ret or frame is None:
        raise RuntimeError("Cannot read first frame")

    h, w, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 25, (w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(img)

        label = "none"

        if res.pose_landmarks:
            pts = get_pose_points(res)
            fL = compute_features(pts, "L")
            fR = compute_features(pts, "R")
            label = classify_F_class(pts, contact=False)

        cv2.putText(frame, label, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        out.write(frame)

    cap.release()
    out.release()

    print("Output saved to:", output_path)


# In[ ]:


def get_hand_center_from_pose(pose_pts):

    lw = pose_pts["l_wrist"]
    rw = pose_pts["r_wrist"]
    le = pose_pts["l_elbow"]
    re = pose_pts["r_elbow"]

    # Same logic used in your notebooks
    left_hand_center  = (lw[0], lw[1] + (lw[1] - le[1]) // 3)
    right_hand_center = (rw[0], rw[1] + (rw[1] - re[1]) // 3)

    return left_hand_center, right_hand_center


# In[ ]:


def detect_arm_space_position(frame):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    if not res.pose_landmarks:
        return None

    pts = get_pose_points(res)

    if pts is None:
        return None

    return classify_F_class(pts, contact=False)


# In[ ]:


def run_arm_space_module(video_path):

    predictions = []

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {"per_frame": [], "final": None}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        loc = detect_arm_space_position(frame)

        if loc:
            predictions.append(loc)

    cap.release()

    if not predictions:
        return {"per_frame": [], "final": None}

    valid = [p for p in predictions if p != "none"]
    if valid:
      final_location = Counter(valid).most_common(1)[0][0]
    else:
      final_location = "none"

    return {
        "per_frame": predictions,
        "final": final_location
    }


# In[ ]:




