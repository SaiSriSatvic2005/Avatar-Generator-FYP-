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


#pip install mediapipe opencv-python numpy


# In[ ]:


import cv2
import mediapipe as mp
import numpy as np


# In[ ]:


def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def mean_motion(seq):
    return np.mean([dist(seq[i], seq[i+1]) for i in range(len(seq)-1)])


# In[ ]:


def is_hamtouch(p1, p2, ref_dist):
    return dist(p1, p2) < 0.25 * ref_dist


# In[ ]:


def is_hamclose(p1, p2, ref_dist):
    d = dist(p1, p2)
    return 0.25 * ref_dist < d < 0.6 * ref_dist


# In[ ]:


def is_hambrushing(seq1, seq2, ref_dist):
    distances = [dist(a, b) for a, b in zip(seq1, seq2)]
    touching = all(d < 0.25 * ref_dist for d in distances)
    motion = mean_motion(seq1) > 0.08 * ref_dist

    return touching and motion


# In[ ]:


def is_hambehind(p1, p2, ref_dist):
    return p1[2] - p2[2] > 0.3 * ref_dist


# In[ ]:


def is_hamcross(left_seq, right_seq, ref_dist):
    x_flip = (left_seq[0][0] < right_seq[0][0] and
              left_seq[-1][0] > right_seq[-1][0])

    z_cross = abs(left_seq[-1][2] - right_seq[-1][2]) > 0.2 * ref_dist

    return x_flip and z_cross


# In[ ]:


def is_haminterlock(fingers_A, fingers_B, ref_dist):
    close_pairs = 0
    depth_cross = False

    for fa in fingers_A:
        for fb in fingers_B:
            if dist(fa, fb) < 0.6 * ref_dist:
                close_pairs += 1
                if abs(fa[2] - fb[2]) > 0.1 * ref_dist:
                    depth_cross = True

    return close_pairs >= 1 and depth_cross


# In[ ]:


def classify_contact(data):
    ref_dist = data.get("ref_dist", 0.08)

    if "fingers_L" in data and "fingers_R" in data:
        if is_haminterlock(data["fingers_L"], data["fingers_R"], ref_dist):
            return "haminterlock"

    if "hand_L_seq" in data and "hand_R_seq" in data:
        if is_hamcross(data["hand_L_seq"], data["hand_R_seq"], ref_dist):
            return "hamcross"

    if "hand_L_seq" in data and "hand_R_seq" in data:
        if is_hambrushing(data["hand_L_seq"], data["hand_R_seq"], ref_dist):
            return "hambrushing"

    if "hand_L" in data and "hand_R" in data:
        if is_hamtouch(data["hand_L"], data["hand_R"], ref_dist):
            return "hamtouch"

    if "hand_L" in data and "hand_R" in data:
        if is_hamclose(data["hand_L"], data["hand_R"], ref_dist):
            return "hamclose"

    if "hand_L" in data and "hand_R" in data:
        if is_hambehind(data["hand_L"], data["hand_R"], ref_dist):
            return "hambehind"

    return "no-contact"


# In[ ]:


# Top-level demo code guarded for import safety
if __name__ == "__main__":
    mp_hands_demo = mp.solutions.hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    cap = cv2.VideoCapture("Prompt_1.mp4")

    hand_L_seq, hand_R_seq = [], []
    prev_label = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = mp_hands_demo.process(rgb)

        if not result.multi_hand_landmarks:
            continue

        hands_detected = {}
        for lm, handed in zip(result.multi_hand_landmarks,
                              result.multi_handedness):
            label = handed.classification[0].label
            hands_detected[label] = lm.landmark

        if "Left" in hands_detected and "Right" in hands_detected:
            hL = hands_detected["Left"]
            hR = hands_detected["Right"]
            p1 = (hL[8].x, hL[8].y, hL[8].z)
            p2 = (hR[8].x, hR[8].y, hR[8].z)
        elif len(hands_detected) == 1:
            h = list(hands_detected.values())[0]
            p1 = (h[8].x, h[8].y, h[8].z)
            p2 = p1
        else:
            continue

        hand_L_seq.append(p1)
        hand_R_seq.append(p2)

        if len(hand_L_seq) < 3:
            continue

        data = {
            "hand_L": p1,
            "hand_R": p2,
            "hand_L_seq": hand_L_seq[-5:],
            "hand_R_seq": hand_R_seq[-5:],
            "ref_dist": dist((hL[0].x, hL[0].y, hL[0].z), (hL[9].x, hL[9].y, hL[9].z)) if "Left" in hands_detected else 0.08
        }

        label = classify_contact(data)

        if label != prev_label:
            print(label)
            prev_label = label

    cap.release()


# In[ ]:


# Module-level hand detector for detect_contact_type / run_contact_type_module
mp_hands = mp.solutions.hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


def detect_contact_type(frame):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mp_hands.process(rgb)

    if not result.multi_hand_landmarks:
        return "no-contact"

    hands = {}

    for lm, handed in zip(result.multi_hand_landmarks,
                          result.multi_handedness):
        label = handed.classification[0].label
        hands[label] = lm.landmark

    # ---- fingertip extraction (IDENTICAL to notebook) ----
    ref_dist = 0.08 # default fallback
    
    if "Left" in hands and "Right" in hands:

        hL = hands["Left"]
        hR = hands["Right"]

        p1 = (hL[8].x, hL[8].y, hL[8].z)
        p2 = (hR[8].x, hR[8].y, hR[8].z)
        
        ref_dist = dist((hL[0].x, hL[0].y, hL[0].z), (hL[9].x, hL[9].y, hL[9].z))

    elif len(hands) == 1:

        h = list(hands.values())[0]

        p1 = (h[8].x, h[8].y, h[8].z)
        p2 = p1   # single hand → self-contact
        
        ref_dist = dist((h[0].x, h[0].y, h[0].z), (h[9].x, h[9].y, h[9].z))

    else:
        return "no-contact"

    data = {
        "hand_L": p1,
        "hand_R": p2,
        "ref_dist": ref_dist
    }

    return classify_contact(data)


# In[ ]:


def run_contact_type_module(video_path):
    """
    Contact Type Inference Wrapper using shared MediaPipe landmarks
    """
    try:
        from shared_landmarks import get_video_landmarks
        frames = get_video_landmarks(video_path)
        predictions = []
        for f in frames:
            l = f.get("left_hand")
            r = f.get("right_hand")
            if l and r:
                # Check if l and r are the same physical hand double detected
                wrist_dist = dist(
                    (l.landmark[0].x, l.landmark[0].y, l.landmark[0].z),
                    (r.landmark[0].x, r.landmark[0].y, r.landmark[0].z)
                )
                if wrist_dist < 0.08:
                    predictions.append("no-contact")
                    continue

                # Check distances between index tips (8), middle tips (12), and MCP (9)
                pts_l = [(l.landmark[i].x, l.landmark[i].y, l.landmark[i].z) for i in [0, 8, 9, 12]]
                pts_r = [(r.landmark[i].x, r.landmark[i].y, r.landmark[i].z) for i in [0, 8, 9, 12]]
                min_d = min(dist(pl, pr) for pl in pts_l for pr in pts_r)
                if min_d < 0.22:
                    predictions.append("hamtouch")
                elif min_d < 0.45:
                    predictions.append("hamclose")
                else:
                    predictions.append("no-contact")
            else:
                predictions.append("no-contact")


        if predictions:
            from collections import Counter
            valid = [p for p in predictions if p != "no-contact"]
            final_c = Counter(valid).most_common(1)[0][0] if valid else "no-contact"
            return {
                "per_frame": predictions,
                "final": final_c
            }
    except Exception as e:
        pass

    predictions = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        contact = detect_contact_type(frame)
        predictions.append(contact)

    cap.release()

    final_contact = max(set(predictions), key=predictions.count)

    return {
        "per_frame": predictions,
        "final": final_contact
    }



# In[ ]:




