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


import cv2
import mediapipe as mp
import numpy as np
from collections import deque

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

mp_draw = mp.solutions.drawing_utils


# In[ ]:


def get_hand_center(frame):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        return None

    hand = results.multi_hand_landmarks[0]

    h, w = frame.shape[:2]

    wrist = hand.landmark[0]   # most stable for motion tracking

    x = int(wrist.x * w)
    y = int(wrist.y * h)

    return (x, y)


# In[ ]:


def hand_center(landmarks):
    pts = np.array([(lm.x, lm.y, lm.z) for lm in landmarks.landmark])
    return pts.mean(axis=0)  # (x, y, z)


# In[ ]:


trajectory = deque(maxlen=60)  # last ~2 seconds


# In[ ]:


def motion_features(traj):
    if len(traj) < 5:
        return None

    traj = np.array(traj)
    diffs = np.diff(traj, axis=0)

    dx, dy, dz = diffs.mean(axis=0)
    speed = np.linalg.norm(diffs, axis=1).mean()

    return {
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "speed": speed,
        "traj": traj
    }


# In[ ]:


def classify_direction_local(traj, eps=0.002):
    """
    HamNoSys-correct direction:
    dominant instantaneous direction, not net displacement
    """

    diffs = np.diff(traj, axis=0)
    dx = diffs[:, 0]
    dy = diffs[:, 1]
    dz = diffs[:, 2]

    counts = {
        "hammovel": np.sum(dx < -eps),
        "hammover": np.sum(dx > eps),
        "hammoveu": np.sum(dy < -eps),
        "hammoved": np.sum(dy > eps),
        "hammovei": np.sum(dz < -eps),
        "hammoveo": np.sum(dz > eps),
    }

    # pick dominant
    best = max(counts, key=counts.get)

    # if motion is very balanced
    if counts[best] < 5:
        return "hamnomotion"

    return best


# In[ ]:


def classify_speed(speed):
    if speed > 0.03:
        return "hamfast"
    if speed < 0.01:
        return "hamslow"
    return None


# In[ ]:


def classify_force(traj):
    jerk = np.mean(np.abs(np.diff(np.diff(traj, axis=0), axis=0)))
    return "hamtense" if jerk > 0.015 else "hamrest"


# In[ ]:


def detect_halt(traj):
    speed = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    return "hamhalt" if speed[-1] < 0.002 else None


# In[ ]:


def classify_size(traj):
    amp = np.max(np.linalg.norm(traj - traj[0], axis=1))

    if amp < 0.02:
        return "hamsmallmod"
    if amp > 0.08:
        return "hamlargemod"
    return None


# In[ ]:


def classify_growth(traj):
    dists = np.linalg.norm(traj - traj[0], axis=1)

    if dists[-1] > dists[0] * 1.5:
        return "hamincreasing"
    if dists[-1] < dists[0] * 0.7:
        return "hamdecreasing"
    return None


# In[ ]:


def detect_repetition(traj):
    disp = np.diff(traj, axis=0)
    signs = np.sign(disp[:, 0])

    if np.any(signs[:-1] * signs[1:] < 0):
        return "hamrepeatreverse"

    return None


# In[ ]:


def detect_repeat_count(traj):
    zero_crossings = np.sum(np.diff(np.sign(np.diff(traj[:, 0])) != 0))

    if zero_crossings > 4:
        return "hamrepeatcontinueseveral"
    if zero_crossings > 2:
        return "hamrepeatcontinue"
    return None


# In[ ]:


def classify_path(traj):
    diffs = np.diff(traj[:, :2], axis=0)
    angles = np.arctan2(diffs[:, 1], diffs[:, 0])
    angle_change = np.abs(np.diff(angles))

    if np.mean(angle_change) > 1.0:
        return "hamzigzag"

    if np.std(angles) > 0.7:
        return "hamwavy"

    cov = np.cov(traj[:, 0], traj[:, 1])
    eigvals, _ = np.linalg.eig(cov)

    if eigvals[0] / eigvals[1] > 2:
        return "hamellipseh"
    if eigvals[1] / eigvals[0] > 2:
        return "hamellipsev"

    return None


# In[ ]:


'''
cap = cv2.VideoCapture("/content/drive/MyDrive/movement1_test02.mp4")
from collections import Counter
label_log = []


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0]
        center = hand_center(lm)
        trajectory.append(center)

        feats = motion_features(trajectory)
        if feats:

            direction = classify_direction_local(feats["traj"])
            speed = classify_speed(feats["speed"])
            force = classify_force(feats["traj"])
            halt = detect_halt(feats["traj"])
            size = classify_size(feats["traj"])
            growth = classify_growth(feats["traj"])
            path = classify_path(feats["traj"])

            label_log.append((direction, speed, force, halt, size, growth, path))


    from google.colab.patches import cv2_imshow
    cv2_imshow(frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("\n=== FINAL SUMMARY ===")

for i, name in enumerate(
    ["Direction", "Speed", "Force", "Halt", "Size", "Growth", "Path"]
):
    values = [x[i] for x in label_log if x[i] is not None]
    if values:
        print(f"{name}: {Counter(values).most_common(1)[0][0]}")

'''


# In[ ]:


import numpy as np

# =====================================================
# 1. DIRECTION (HamNoSys-correct: local dominance)
# =====================================================

def classify_direction(traj, eps=0.002):
    diffs = np.diff(traj, axis=0)
    dx, dy, dz = diffs[:,0], diffs[:,1], diffs[:,2]

    counts = {
        "u":  np.sum(dy < -eps),
        "d":  np.sum(dy >  eps),
        "l":  np.sum(dx < -eps),
        "r":  np.sum(dx >  eps),
        "i":  np.sum(dz < -eps),
        "o":  np.sum(dz >  eps),
    }

    # dominant axes
    vy = max(counts["u"], counts["d"])
    vx = max(counts["l"], counts["r"])
    vz = max(counts["i"], counts["o"])

    if max(vx, vy, vz) < 5:
        return "hamnomotion"

    # vertical + horizontal
    if counts["u"] > counts["d"]:
        vert = "u"
    else:
        vert = "d"

    if counts["l"] > counts["r"]:
        hor = "l"
    else:
        hor = "r"

    if counts["i"] > counts["o"]:
        dep = "i"
    else:
        dep = "o"

    # diagonals
    if vy > vx and vy > vz:
        return f"hammove{vert}"
    if vx > vy and vx > vz:
        return f"hammove{hor}"
    if vz > vy and vz > vx:
        return f"hammove{dep}"

    # combined
    if vy and vx:
        return f"hammove{vert}{hor}"
    if vy and vz:
        return f"hammove{vert}{dep}"
    if vx and vz:
        return f"hammove{hor}{dep}"

    return "hamnomotion"


# =====================================================
# 2. SPEED
# =====================================================

def classify_speed(traj):
    diffs = np.diff(traj, axis=0)
    speed = np.linalg.norm(diffs, axis=1).mean()
    if speed > 0.03:
        return "hamfast"
    if speed < 0.01:
        return "hamslow"
    return None


# =====================================================
# 3. FORCE (tense vs rest)
# =====================================================

def classify_force(traj):
    accel = np.diff(np.diff(traj, axis=0), axis=0)
    jerk = np.linalg.norm(accel, axis=1).mean()
    return "hamtense" if jerk > 0.015 else "hamrest"


# =====================================================
# 4. HALT
# =====================================================

def detect_halt(traj):
    diffs = np.diff(traj, axis=0)
    speed = np.linalg.norm(diffs, axis=1)
    if speed[-1] < 0.002:
        return "hamhalt"
    return None


# =====================================================
# 5. SIZE
# =====================================================

def classify_size(traj):
    amp = np.max(np.linalg.norm(traj - traj[0], axis=1))
    if amp < 0.02:
        return "hamsmallmod"
    if amp > 0.08:
        return "hamlargemod"
    return None


# =====================================================
# 6. GROWTH
# =====================================================

def classify_growth(traj):
    d = np.linalg.norm(traj - traj[0], axis=1)
    if d[-1] > d[0] * 1.5:
        return "hamincreasing"
    if d[-1] < d[0] * 0.7:
        return "hamdecreasing"
    return None


# =====================================================
# 7. REPETITION
# =====================================================

def classify_repetition(traj):
    dx = np.diff(traj[:,0])
    zero_cross = np.sum(np.diff(np.sign(dx)) != 0)

    if zero_cross > 6:
        return "hamrepeatcontinueseveral"
    if zero_cross > 3:
        return "hamrepeatcontinue"
    return None


def classify_reverse(traj):
    dx = np.diff(traj[:,0])
    if np.any(dx[:-1] * dx[1:] < 0):
        return "hamrepeatreverse"
    return None


# =====================================================
# 8. PATH SHAPE
# =====================================================

def classify_path(traj):
    diffs = np.diff(traj[:,:2], axis=0)
    angles = np.arctan2(diffs[:,1], diffs[:,0])
    angle_var = np.std(np.diff(angles))

    # zigzag
    if angle_var > 1.2:
        return "hamzigzag"

    # wavy
    if 0.4 < angle_var <= 1.2:
        return "hamwavy"

    # ellipse / arc
    cov = np.cov(traj[:,0], traj[:,1])
    eigvals, _ = np.linalg.eig(cov)

    if eigvals[0] / eigvals[1] > 2:
        return "hamellipseh"
    if eigvals[1] / eigvals[0] > 2:
        return "hamellipsev"

    return None


# In[ ]:


def classify_repeat_from_start(traj, eps=0.015):
    start = traj[0]
    dists = np.linalg.norm(traj - start, axis=1)

    # count how many times we return near start
    returns = np.sum(dists < eps)

    if returns >= 3:
        return "hamrepeatfromstartseveral"
    if returns == 2:
        return "hamrepeatfromstart"
    return None


# In[ ]:


# Top-level demo code guarded for import safety
if __name__ == "__main__":
    from collections import deque, Counter

    video_path = "Prompt_1.mp4"
    cap = cv2.VideoCapture(video_path)

    trajectory = deque(maxlen=60)
    label_log = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]
            center = hand_center(lm)
            trajectory.append(center)

            if len(trajectory) > 10:
                traj = np.array(trajectory)

                direction = classify_direction(traj)
                speed     = classify_speed(traj)
                force     = classify_force(traj)
                halt      = detect_halt(traj)
                size      = classify_size(traj)
                growth    = classify_growth(traj)
                path      = classify_path(traj)

                repeat_continue   = classify_repetition(traj)
                repeat_from_start = classify_repeat_from_start(traj)
                repeat = repeat_from_start if repeat_from_start else repeat_continue

                reverse = classify_reverse(traj)

                label_log.append({
                    "Direction": direction,
                    "Speed": speed,
                    "Force": force,
                    "Halt": halt,
                    "Size": size,
                    "Growth": growth,
                    "Repeat": repeat,
                    "Reverse": reverse,
                    "Path": path
                })

    cap.release()

    print("\n=== FINAL SUMMARY ===")
    if len(label_log) == 0:
        print("No labels detected.")
    else:
        keys = label_log[0].keys()
        for key in keys:
            vals = [x[key] for x in label_log if x[key] is not None]
            if vals:
                print(f"{key}: {Counter(vals).most_common(1)[0][0]}")


# In[ ]:


from collections import Counter

def classify_movement1(trajectory_log):

    if trajectory_log is None or len(trajectory_log) == 0:
        return None

    keys = trajectory_log[0].keys()
    final_symbols = []

    for key in keys:
        vals = [x[key] for x in trajectory_log if x[key] is not None]

        if vals:
            final_symbols.append(Counter(vals).most_common(1)[0][0])

    return " ".join(final_symbols)


# In[ ]:


def run_movement1_module(video_path):

    trajectory = deque(maxlen=60)
    label_log = []

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks:

            lm = res.multi_hand_landmarks[0]
            center = hand_center(lm)
            trajectory.append(center)

            if len(trajectory) > 10:

                traj = np.array(trajectory)

                direction = classify_direction(traj)
                speed     = classify_speed(traj)
                force     = classify_force(traj)
                halt      = detect_halt(traj)
                size      = classify_size(traj)
                growth    = classify_growth(traj)
                path      = classify_path(traj)

                repeat_continue   = classify_repetition(traj)
                repeat_from_start = classify_repeat_from_start(traj)
                repeat = repeat_from_start if repeat_from_start else repeat_continue

                reverse = classify_reverse(traj)

                label_log.append({
                    "Direction": direction,
                    "Speed": speed,
                    "Force": force,
                    "Halt": halt,
                    "Size": size,
                    "Growth": growth,
                    "Repeat": repeat,
                    "Reverse": reverse,
                    "Path": path
                })

    cap.release()

    return classify_movement1(label_log)


# In[ ]:




