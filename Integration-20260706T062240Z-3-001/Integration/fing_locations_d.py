#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
from google.colab import drive
drive.mount('/content/drive')
'''


# In[ ]:


#pip uninstall -y numpy mediapipe opencv-python opencv-python-headless


# In[ ]:


#!pip install numpy==1.26.4
#!pip install mediapipe==0.10.21
#!pip install opencv-python==4.8.1.78


# In[ ]:


import cv2
import mediapipe as mp
import numpy as np
from collections import Counter, deque

hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


# In[ ]:


FINGER_LANDMARKS = {
    "hamthumb":        [1, 2, 3, 4],
    "hamindexfinger":  [5, 6, 7, 8],
    "hammiddlefinger": [9, 10, 11, 12],
    "hamringfinger":   [13, 14, 15, 16],
    "hampinky":        [17, 18, 19, 20],
}

TIP_LM  = [4, 8, 12, 16, 20]
PIP_LM  = [6, 10, 14, 18]
MCP_LM  = [5, 9, 13, 17]
PALM_LM = [0, 5, 9, 13, 17]
#To ignore natural touching between neighboring fingers and avoid false contact detection.
ADJACENT_FINGERS = {
    "hamthumb":        ["hamindexfinger"],
    "hamindexfinger":  ["hamthumb", "hammiddlefinger"],
    "hammiddlefinger": ["hamindexfinger", "hamringfinger"],
    "hamringfinger":   ["hammiddlefinger", "hampinky"],
    "hampinky":        ["hamringfinger"],
}



def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def mp_landmarks_to_xyz(hand_landmarks):
    return [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]


def finger_of_landmark(idx):
    for finger, ids in FINGER_LANDMARKS.items():
        if idx in ids:
            return finger
    return None

#It finds which fingertip is closest to the palm.
def estimate_contact_index(landmarks):
    palm_center = np.mean([landmarks[i] for i in PALM_LM], axis=0)

    best_idx, best_d = None, float("inf")
    for i in TIP_LM:
        d = dist(palm_center, landmarks[i])
        if d < best_d:
            best_d = d
            best_idx = i

    return best_idx


def has_contact_candidate(contact_idx, landmarks, thresh=0.03):
    """
    Spatial contact candidate:
    - NOT same finger
    - NOT adjacent fingers
    """
    contact = landmarks[contact_idx]
    finger = finger_of_landmark(contact_idx)

    for i, lm in enumerate(landmarks):
        other_finger = finger_of_landmark(i)

        if other_finger == finger:
            continue

        if other_finger in ADJACENT_FINGERS.get(finger, []):
            continue

        if dist(contact, lm) < thresh:
            return True

    return False


# Finger identity logic (NO CONTACT)
def finger_straightness(landmarks, finger):
    ids = FINGER_LANDMARKS[finger]
    mcp, pip, dip, tip = [np.array(landmarks[i]) for i in ids]

    v1 = pip - mcp
    v2 = dip - pip
    v3 = tip - dip

    def cos_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)

    return (cos_sim(v1, v2) + cos_sim(v2, v3)) / 2

#It finds which finger is active when there is no contact.The active finger is usually straight and extended, while others are bent or close to the palm.
def classify_finger_identity(landmarks):
    palm_center = np.mean([landmarks[i] for i in PALM_LM], axis=0)

    best_finger = None
    best_score = -1

    for finger, ids in FINGER_LANDMARKS.items():
        tip_idx = ids[-1]
        extension = dist(landmarks[tip_idx], palm_center)
        straightness = finger_straightness(landmarks, finger)

        score = extension * straightness
        if score > best_score:
            best_score = score
            best_finger = finger

    return best_finger


# Finger-part classifiers (CONTACT ONLY)
def classify_finger_part(contact, landmarks):
    if min(dist(contact, landmarks[i]) for i in TIP_LM) < 0.025:
        return "hamfingertip"
    if min(dist(contact, landmarks[i]) for i in PIP_LM) < 0.03:
        return "hamfingermidjoint"
    if min(dist(contact, landmarks[i]) for i in MCP_LM) < 0.035:
        return "hamfingerbase"
    return None


def classify_hambetween(contact, landmarks):
    centers = {
        f: np.mean([landmarks[i] for i in ids], axis=0)
        for f, ids in FINGER_LANDMARKS.items()
    }
    close = [f for f, c in centers.items() if dist(contact, c) < 0.035]
    return "hambetween" if len(close) >= 2 else None



def classify_frame(landmarks, stable_contact):

    contact_idx = estimate_contact_index(landmarks)
    contact = landmarks[contact_idx]

    if stable_contact:
        part = classify_finger_part(contact, landmarks)
        if part:
            return part

        between = classify_hambetween(contact, landmarks)
        if between:
            return between

    return classify_finger_identity(landmarks)
def classify_fingertip_side(contact, landmarks, contact_idx):
    tip = np.array(landmarks[contact_idx])
    palm_center = np.mean([landmarks[i] for i in PALM_LM], axis=0)
    tip_to_palm = palm_center - tip
    tip_to_contact = contact - tip
    dot = np.dot(tip_to_palm, tip_to_contact)

    if dot > 0:
        return "hamfingerpad"
    else:
        return "hamfingernail"


# VIDEO → ONE LABEL (temporal stability + voting)
def classify_video(video_path):

    CONTACT_STABILITY = 4

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    cap = cv2.VideoCapture(video_path)
    votes = []
    contact_history = deque(maxlen=CONTACT_STABILITY)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            landmarks = mp_landmarks_to_xyz(hand)

            contact_idx = estimate_contact_index(landmarks)
            candidate = has_contact_candidate(contact_idx, landmarks)

            contact_history.append(candidate)
            stable_contact = (
                len(contact_history) == CONTACT_STABILITY
                and all(contact_history)
            )

            label = classify_frame(landmarks, stable_contact)
            if label:
                votes.append(label)

    cap.release()
    hands.close()

    return Counter(votes).most_common(1)[0][0] if votes else None



def main(video_path):
    return classify_video(video_path)


# In[ ]:


#label = main("/content/drive/MyDrive/pinkyvid.mp4")
#print("Final Finger Location Label:", label)


# In[ ]:


def detect_finger_location(frame):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if not result.multi_hand_landmarks:
        return None

    hand = result.multi_hand_landmarks[0]
    landmarks = mp_landmarks_to_xyz(hand)

    contact_idx = estimate_contact_index(landmarks)
    stable_contact = has_contact_candidate(contact_idx, landmarks)

    return classify_frame(landmarks, stable_contact)


# In[ ]:


def run_finger_location_module(video_path):
    """
    Finger Location Inference Wrapper
    """

    predictions = []

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 👇 IMPORTANT: CALL YOUR EXISTING LOGIC HERE
        loc = detect_finger_location(frame)   # ← CHANGE THIS if needed

        predictions.append(loc)

    cap.release()
    valid = [p for p in predictions if p]

    if valid:
      final_location = Counter(valid).most_common(1)[0][0]
    else:
      final_location = None

    final_location = max(set(predictions), key=predictions.count)

    return {
        "per_frame": predictions,
        "final": final_location
    }


# In[ ]:




