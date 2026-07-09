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


# cv2_imshow removed (Colab-only)


# In[ ]:


import cv2
import mediapipe as mp
import numpy as np
from collections import Counter


# In[ ]:


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# In[ ]:


import cv2
import mediapipe as mp
import numpy as np
from collections import Counter

# ==============================
# Utility
# ==============================

def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


# ==============================
# Frame-level classifier
# ==============================

def classify_hand_location(hand_pts, contact_point, side):
    wrist = hand_pts[0]
    thumb_mcp = hand_pts[2]
    index_mcp = hand_pts[5]
    middle_mcp = hand_pts[9]
    ring_mcp = hand_pts[13]
    pinky_mcp = hand_pts[17]

    palm_center = np.mean(
        [wrist, index_mcp, middle_mcp, ring_mcp, pinky_mcp],
        axis=0
    )

    palm_size = dist(wrist, middle_mcp)

    thumb_ball_center = (
        0.6 * np.array(thumb_mcp) +
        0.4 * np.array(wrist)
    )

    palm_axis = np.array(middle_mcp) - np.array(wrist)
    palm_axis /= np.linalg.norm(palm_axis)

    perp = np.array([-palm_axis[1], palm_axis[0]])
    rel = np.array(contact_point) - palm_center
    side_score = np.dot(rel, perp)

    if side == "L":
        side_score *= -1

    if dist(contact_point, thumb_ball_center) < 0.35 * palm_size:
        return "hamthumbball"

    if dist(contact_point, palm_center) < 0.55 * palm_size:
        return "hampalm"

    if side_score > 0.30 * palm_size:
        return "hamthumbside"

    if side_score < -0.30 * palm_size:
        return "hampinkyside"

    return "hamhandback"


# ==============================
# Fingertip detector
# ==============================

def detect_touching_fingertip(hand_pts):
    palm_center = np.mean(
        [hand_pts[i] for i in [0, 5, 9, 13, 17]],
        axis=0
    )
    tips = [4, 8, 12, 16, 20]
    return min(tips, key=lambda i: dist(hand_pts[i], palm_center))


# ==============================
# Final video-level decision
# ==============================

def final_hand_location(labels):
    counts = Counter(labels)

    if counts["hamthumbball"] > 0:
        return "hamthumbball"

    if counts["hampalm"] >= max(
        counts["hamthumbside"],
        counts["hampinkyside"],
        counts["hamhandback"]
    ):
        return "hampalm"

    return counts.most_common(1)[0][0]


# ==============================
# Video processing (SILENT)
# ==============================

# Top-level demo code guarded for import safety
if __name__ == "__main__":
    VIDEO_PATH = "Prompt_1.mp4"

    cap = cv2.VideoCapture(VIDEO_PATH)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    labels_buffer = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0]
            handedness = result.multi_handedness[0].classification[0].label
            side = "R" if handedness == "Right" else "L"

            hand_pts = {i: (lm.landmark[i].x, lm.landmark[i].y) for i in range(21)}

            tip_id = detect_touching_fingertip(hand_pts)
            contact_point = hand_pts[tip_id]

            label = classify_hand_location(hand_pts, contact_point, side)

            wrist = hand_pts[0]
            middle_mcp = hand_pts[9]
            palm_size = dist(wrist, middle_mcp)
            palm_center = np.mean(
                [hand_pts[i] for i in [0, 5, 9, 13, 17]],
                axis=0
            )

            # keep only real contact frames
            if dist(contact_point, palm_center) < 0.65 * palm_size:
                labels_buffer.append(label)

    cap.release()

    final_label = final_hand_location(labels_buffer)
    print("Frame counts:", Counter(labels_buffer))
    print("FINAL HAND LOCATION:", final_label)


# In[ ]:


def detect_hand_location(frame):

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if not result.multi_hand_landmarks:
        return None

    lm = result.multi_hand_landmarks[0]
    handedness = result.multi_handedness[0].classification[0].label
    side = "R" if handedness == "Right" else "L"

    hand_pts = {i: (lm.landmark[i].x, lm.landmark[i].y) for i in range(21)}

    tip_id = detect_touching_fingertip(hand_pts)
    contact_point = hand_pts[tip_id]

    label = classify_hand_location(hand_pts, contact_point, side)

    wrist = hand_pts[0]
    middle_mcp = hand_pts[9]

    palm_size = dist(wrist, middle_mcp)

    palm_center = np.mean(
        [hand_pts[i] for i in [0, 5, 9, 13, 17]],
        axis=0
    )

    # keep only valid contact frames (same logic as notebook)
    if dist(contact_point, palm_center) < 0.65 * palm_size:
        return label

    return None


# In[ ]:


def run_hand_location_module(video_path):
    """
    Hand Location Inference Wrapper
    """

    predictions = []

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 👇 IMPORTANT: CALL YOUR EXISTING LOGIC HERE
        loc = detect_hand_location(frame)   # ← CHANGE THIS if needed

        predictions.append(loc)

    cap.release()

    valid = [p for p in predictions if p]

    if valid:
      final_location = Counter(valid).most_common(1)[0][0]
    else:
      final_location = None



    return {
        "per_frame": predictions,
        "final": final_location
    }


# In[ ]:




