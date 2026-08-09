#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip install mediapipe==0.10.21 opencv-python numpy matplotlib


# In[ ]:


import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, Counter

try:
    from shared_landmarks import get_video_landmarks
except ImportError:
    get_video_landmarks = None


# In[ ]:


def get_secondary_motion(frame):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        return None

    hand = results.multi_hand_landmarks[0]

    h, w = frame.shape[:2]

    # ✅ Wrist is most stable tracking point for movement
    wrist = hand.landmark[0]

    x = int(wrist.x * w)
    y = int(wrist.y * h)

    return (x, y)


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

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


# In[ ]:


def norm_dist(p1, p2, scale):
    if scale == 0:
        return 999
    return dist(p1, p2) / scale


# In[ ]:


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


# In[ ]:


def path_length(traj):
    traj = np.array(traj)
    return np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))


# In[ ]:


def mean_turning_angle(traj):
    traj = np.array(traj)
    v = np.diff(traj, axis=0)

    angles = []
    for i in range(len(v)-1):
        a, b = v[i], v[i+1]
        if np.linalg.norm(a) < 1e-6 or np.linalg.norm(b) < 1e-6:
            continue
        cosang = np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b))
        angles.append(np.arccos(np.clip(cosang, -1, 1)))

    return np.mean(angles) if angles else 0.0


# In[ ]:


def is_loop(traj, disp_eps=25, length_ratio=3.0):
    traj = np.array(traj)
    disp = np.linalg.norm(traj[-1] - traj[0])
    plen = path_length(traj)
    return disp < disp_eps and plen > length_ratio * disp


# In[ ]:


def dominant_plane(traj):
    var = np.var(traj, axis=0)
    idx = np.argmin(var)
    return ["YZ", "XZ", "XY"][idx]


# In[14]:


'''
def rotation_dir(traj, plane):
    if plane == "XY":
        x, y = traj[:,0], traj[:,1]
    elif plane == "XZ":
        x, y = traj[:,0], traj[:,2]
    else:
        x, y = traj[:,1], traj[:,2]

    ang = np.unwrap(np.arctan2(y - y.mean(), x - x.mean()))
    return "ccw" if np.mean(np.diff(ang)) > 0 else "cw"
'''


# In[16]:


def rotation_dir(traj, plane=None):

    traj = np.array(traj)

    if len(traj) < 3:
        return None   # Not enough points for rotation

    # ✅ Handle 2D trajectories (MOST COMMON CASE)
    if traj.shape[1] == 2:

        x = traj[:, 0]
        y = traj[:, 1]

        # Shoelace-style signed area
        area = 0.0
        for i in range(len(traj) - 1):
            area += (x[i+1] - x[i]) * (y[i+1] + y[i])

        return "cw" if area > 0 else "ccw"

    # ✅ Handle 3D trajectories (kept compatible)
    if plane == "XY":
        x, y = traj[:, 0], traj[:, 1]

    elif plane == "XZ":
        x, y = traj[:, 0], traj[:, 2]

    else:  # YZ fallback
        x, y = traj[:, 1], traj[:, 2]

    area = 0.0
    for i in range(len(traj) - 1):
        area += (x[i+1] - x[i]) * (y[i+1] + y[i])

    return "cw" if area > 0 else "ccw"


# In[11]:


'''
def classify_movement2(traj):
    traj = np.array(traj)
    traj = traj - traj[0]

    normalized_coords = np.max(np.abs(traj)) <= 2.0
    span = np.max(np.linalg.norm(traj, axis=1))

    if span < (0.03 if normalized_coords else 25):
        return "hamnomotion"

    disp = np.linalg.norm(traj[-1] - traj[0])
    plen = path_length(traj)
    turn = mean_turning_angle(traj)

    # 1️⃣ Circular / looped paths FIRST
    if is_loop(traj, disp_eps=25) and turn > 0.05:
        plane = dominant_plane(traj)
        rot = rotation_dir(traj, plane)

        # Stirring
        if disp < 15:
            return "hamstircw" if rot == "cw" else "hamstirccw"

        # Full clock
        if plen > 6 * disp:
            return "hamclockfull"

        # Clock vs circle
        if rot == "cw":
            return f"hamclock{plane.lower()}"
        else:
            return f"hamcircle{plane.lower()}"

    # 2️⃣ Oscillatory (after loop)
    vel = np.diff(traj, axis=0)

    if np.sum(np.sign(vel[:,1][:-1]) != np.sign(vel[:,1][1:])) > 6:
        return "hamnodding"

    if np.sum(np.sign(vel[:,0][:-1]) != np.sign(vel[:,0][1:])) > 6:
        return "hamswinging"

    return "unknown_movement2"
'''


# In[13]:


def classify_movement2(traj):

    if traj is None or len(traj) < 2:
        return None

    traj = np.array(traj)
    
    try:
        from video_preprocessing import smooth_trajectory
        traj = smooth_trajectory(traj, window_size=5, polyorder=2)
    except ImportError:
        pass

    traj = traj - traj[0]

    normalized_coords = np.max(np.abs(traj)) <= 2.0
    span = np.max(np.linalg.norm(traj, axis=1))

    disp = np.linalg.norm(traj[-1] - traj[0])
    plen = path_length(traj)
    turn = mean_turning_angle(traj)

    if traj.shape[1] >= 3:
        dx = traj[-1, 0] - traj[0, 0]
        dy = traj[-1, 1] - traj[0, 1]
        dz = traj[-1, 2] - traj[0, 2]
        if abs(dz) > max(abs(dx), abs(dy)) * 1.2 and abs(dz) > 0.01:
            return "hammovei" if dz < 0 else "hammoveo"

    # ✅ ROBUST LOOP DETECTION
    if plen > disp * 4 and turn > 0.05:

        plane = dominant_plane(traj)
        rot = rotation_dir(traj, plane)

        tight_loop_disp = 0.05 if normalized_coords else 15
        full_loop_path = 0.60 if normalized_coords else 6 * disp

        if disp < tight_loop_disp:
            return "hamstircw" if rot == "cw" else "hamstirccw"

        if plen > full_loop_path:
            return "hamclockfull"

        if rot == "cw":
            return f"hamclock{plane.lower()}"
        else:
            return f"hamcircle{plane.lower()}"

    vel = np.diff(traj, axis=0)

    if len(vel) < 2:
        return None

    if span > (0.06 if normalized_coords else 50) and np.sum(np.sign(vel[:,1][:-1]) != np.sign(vel[:,1][1:])) > 6:
        return "hamnodding"

    if span > (0.06 if normalized_coords else 50) and np.sum(np.sign(vel[:,0][:-1]) != np.sign(vel[:,0][1:])) > 6:
        return "hamswinging"

    return "unknown_movement2"


# In[ ]:


trajectory = deque(maxlen=80)

def hand_motion_point(lm, frame_shape):
    h, w, _ = frame_shape
    p = lm.landmark[8]   # index fingertip
    return np.array([
        p.x * w,
        p.y * h,
        p.z
    ])


def _landmark_point(hand_landmarks, point_index=8):
    p = hand_landmarks.landmark[point_index]
    return np.array([p.x, p.y, p.z], dtype=float)


def _extract_motion_from_shared_landmarks(video_path):
    if get_video_landmarks is None:
        return []

    frames = get_video_landmarks(video_path)
    trajectory = []

    for frame_info in frames:
        hand = (
            frame_info.get("primary_hand")
            or frame_info.get("secondary_hand")
            or frame_info.get("right_hand")
            or frame_info.get("left_hand")
        )
        if hand is None:
            continue
        trajectory.append(_landmark_point(hand, 8))

    return trajectory


def _extract_motion_fallback(video_path):
    trajectory = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return trajectory

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if not results.multi_hand_landmarks:
            continue

        if len(results.multi_hand_landmarks) > 1:
            best_idx = 0
            best_area = -1.0
            for idx, lm in enumerate(results.multi_hand_landmarks):
                xs = [pt.x for pt in lm.landmark]
                ys = [pt.y for pt in lm.landmark]
                area = (max(xs) - min(xs)) * (max(ys) - min(ys))
                if area > best_area:
                    best_area = area
                    best_idx = idx
            hand = results.multi_hand_landmarks[best_idx]
        else:
            hand = results.multi_hand_landmarks[0]

        trajectory.append(_landmark_point(hand, 8))

    cap.release()
    return trajectory


# In[ ]:


'''from google.colab import drive
drive.mount('/content/drive')
'''


# In[ ]:


# Top-level demo code guarded for import safety
if __name__ == "__main__":
    video_path = "Prompt_1.mp4"

    trajectory.clear()
    predictions = []
    hand_frames = 0

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            hand_frames += 1
            lm = res.multi_hand_landmarks[0]
            pt = hand_motion_point(lm, frame.shape)
            trajectory.append(pt)

            if len(trajectory) > 8:
                label = classify_movement2(list(trajectory))
                predictions.append(label)

    cap.release()

    print("Frames with hand detected:", hand_frames)

    counts = Counter(predictions)
    print("Prediction counts:", counts)

    if counts:
        print("FINAL LABEL:", counts.most_common(1)[0][0])
    else:
        print("FINAL LABEL: no prediction")


# In[ ]:


'''
traj = np.array(trajectory)

if __name__ == "__main__":
    plt.figure(figsize=(5,5))
    plt.plot(traj[:,0], traj[:,1], '-o')
    plt.gca().invert_yaxis()
    plt.grid()
    plt.show()
'''


# In[ ]:


def run_movement2_module(video_path):

    trajectory = _extract_motion_from_shared_landmarks(video_path)
    if not trajectory:
        trajectory = _extract_motion_fallback(video_path)

    movement_label = classify_movement2(trajectory)
    if movement_label in [None, "unknown_movement2"]:
        movement_label = "hamnomotion"

    return {
        "trajectory": trajectory,
        "final": movement_label
    }


# In[ ]:




