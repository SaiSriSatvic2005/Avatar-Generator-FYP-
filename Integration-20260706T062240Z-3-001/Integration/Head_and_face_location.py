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
import mediapipe as mp
import numpy as np
from collections import Counter, deque


# In[ ]:


mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.4
)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# In[ ]:


def get_index_fingertip(hand_landmarks, img_w, img_h):
    tip = hand_landmarks.landmark[8]  # index fingertip
    fx = int(tip.x * img_w)
    fy = int(tip.y * img_h)
    return fx, fy


# In[ ]:


def get_face_bbox(face_detection, img_w, img_h):
    bbox = face_detection.location_data.relative_bounding_box

    x1 = int(max(0, bbox.xmin * img_w))
    y1 = int(max(0, bbox.ymin * img_h))
    x2 = int(min(img_w, (bbox.xmin + bbox.width) * img_w))
    y2 = int(min(img_h, (bbox.ymin + bbox.height) * img_h))

    return x1, y1, x2, y2


# In[ ]:


def is_near_face(face_bbox, fx, fy):
    x1, y1, x2, y2 = face_bbox
    face_h = y2 - y1
    face_w = x2 - x1

    rel_y = (fy - y1) / face_h

    # Nose / mouth projection zone
    if 0.35 <= rel_y <= 0.65:
        return True

    pad_x = 0.35 * face_w
    pad_y = 0.35 * face_h

    return (
        x1 - pad_x <= fx <= x2 + pad_x and
        y1 - pad_y <= fy <= y2 + pad_y
    )


# In[ ]:


def mouth_center(face_bbox):
    x1, y1, x2, y2 = face_bbox
    face_h = y2 - y1
    face_w = x2 - x1

    mx = x1 + 0.5 * face_w
    my = y1 + 0.55 * face_h   # mouth center (empirical)

    return mx, my


# In[ ]:


from collections import deque
fy_history = deque(maxlen=7)


# In[ ]:


def classify_face_region(face_bbox, contact_point):
    x1, y1, x2, y2 = face_bbox
    fx, fy = contact_point

    face_h = max(1, y2 - y1)
    face_w = max(1, x2 - x1)

    rel_y = (fy - y1) / face_h
    rel_x = (fx - x1) / face_w

    # ---------------- HEAD ----------------
    if rel_y < 0.05:
        return "hamheadtop"

    if 0.05 <= rel_y < 0.20:
        return "hamforehead"

    # ---------------- EYES / EARS ----------------
    if 0.20 <= rel_y < 0.35:
        if rel_x < 0.15 or rel_x > 0.85:
            return "hamear"
        return "hameyes"

    mx, my = mouth_center(face_bbox)
    mouth_x_tolerance = 0.30 * face_w
    mouth_y_tolerance = 0.12 * face_h

    # Prefer mouth-related labels whenever the contact is near the mouth center,
    # even if the point is slightly above the center line.
    if 0.35 <= rel_y <= 0.75 and abs(fx - mx) <= mouth_x_tolerance and abs(fy - my) <= 0.16 * face_h:
        fy_history.append(fy)

        if len(fy_history) >= 5:
            y_range = max(fy_history) - min(fy_history)
        else:
            y_range = 0

        if y_range < 0.02 * face_h:
            return "hamteeth"

        if fy < my - 0.04 * face_h:
            return "hamlips"

        return "hamtongue"

    # ---------------- NOSE ----------------
    if 0.35 <= rel_y < 0.48:
        return "hamnose" if rel_y < 0.42 else "hamnostrils"

    # ---------------- MOUTH (FIXED) ----------------
    if 0.48 <= rel_y < 0.75:
        if 0.30 <= rel_x <= 0.70:
            fy_history.append(fy)

            if len(fy_history) >= 5:
                y_range = max(fy_history) - min(fy_history)
            else:
                y_range = 0

            if y_range < 0.02 * face_h:
                return "hamteeth"

            if fy < my - 0.04 * face_h:
                return "hamlips"

            return "hamtongue"

    # ---------------- CHIN / NECK ----------------
    if 0.72 <= rel_y < 0.82:
        return "hamchin"

    if 0.82 <= rel_y < 0.90:
        return "hamunderchin"

    return "hamneck"


# In[ ]:


def stabilize_y(prev_y, curr_y, alpha=0.6):
    if prev_y is None:
        return curr_y
    return alpha * curr_y + (1 - alpha) * prev_y


# In[ ]:


'''
#cap = cv2.VideoCapture("/content/drive/MyDrive/teeth_testing.mp4")

labels = []
prev_fy = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_h, img_w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_res = face_detector.process(rgb)
    hand_res = hands.process(rgb)

    if not face_res.detections or not hand_res.multi_hand_landmarks:
        continue

    face_bbox = get_face_bbox(face_res.detections[0], img_w, img_h)
    fx, fy = get_index_fingertip(
        hand_res.multi_hand_landmarks[0], img_w, img_h
    )

    if not is_near_face(face_bbox, fx, fy):
        continue

    fy_stable = stabilize_y(prev_fy, fy)
    prev_fy = fy_stable

    # ✅ ADD THIS LINE HERE (IMPORTANT)
    fy_history.append(fy_stable)

    label = classify_face_region(face_bbox, (fx, fy_stable))
    labels.append(label)

cap.release()
'''


# In[ ]:


#print(cap.isOpened())


# In[ ]:


from collections import Counter

#print("Total frames classified:", len(labels))
#print(Counter(labels))


# In[ ]:


import cv2
import numpy as np
from collections import Counter

def visualize_hamnosys_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    out = None

    labels = []
    prev_fy = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img_h, img_w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_mesh_res = face_mesh.process(rgb)
        hand_res = hands.process(rgb)

        if not face_mesh_res.multi_face_landmarks or not hand_res.multi_hand_landmarks:
            continue

        face_bbox = get_face_bbox_from_mesh(
            face_mesh_res.multi_face_landmarks[0], img_w, img_h
        )

        fx, fy = get_index_fingertip(
            hand_res.multi_hand_landmarks[0], img_w, img_h
        )

        if not is_near_face(face_bbox, fx, fy):
            continue

        fy_stable = stabilize_y(prev_fy, fy)
        prev_fy = fy_stable
        fy_history.append(fy_stable)

        label = classify_face_region(face_bbox, (fx, fy_stable))
        labels.append(label)

        # ---------- DRAW ----------
        x1, y1, x2, y2 = face_bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.circle(frame, (int(fx), int(fy_stable)), 6, (0,0,255), -1)

        cv2.putText(frame, f"Frame label: {label}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255,255,0), 2)

        if out is None:
            out = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                cap.get(cv2.CAP_PROP_FPS),
                (img_w, img_h)
            )

        out.write(frame)

    cap.release()
    if out:
        out.release()

    counts = Counter(labels)
    final_label, freq = counts.most_common(1)[0]
    confidence = freq / len(labels)

    return final_label, round(confidence, 2)


# In[ ]:


import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)



# In[ ]:


def get_face_bbox_from_mesh(face_landmarks, img_w, img_h):
    xs = [lm.x * img_w for lm in face_landmarks.landmark]
    ys = [lm.y * img_h for lm in face_landmarks.landmark]

    x1 = int(min(xs))
    y1 = int(min(ys))
    x2 = int(max(xs))
    y2 = int(max(ys))

    return x1, y1, x2, y2


# In[ ]:


# Top-level demo code guarded for import safety
if __name__ == "__main__":
    final_label, conf = visualize_hamnosys_video(
        "Prompt_1.mp4",
        "nose_pre_output.mp4"
    )
    print(final_label, conf)


# In[ ]:


def detect_head_face_location(frame):

    img_h, img_w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_res = face_detector.process(rgb)
    hand_res = hands.process(rgb)

    if not face_res.detections or not hand_res.multi_hand_landmarks:
        return None

    face_bbox = get_face_bbox(face_res.detections[0], img_w, img_h)

    fx, fy = get_index_fingertip(
        hand_res.multi_hand_landmarks[0], img_w, img_h
    )

    if not is_near_face(face_bbox, fx, fy):
        return None

    fy_stable = stabilize_y(None, fy)  # smoothing optional

    fy_history.append(fy_stable)

    return classify_face_region(face_bbox, (fx, fy_stable))


# In[ ]:


def run_head_face_location_module(video_path):
    try:
        from shared_landmarks import get_video_landmarks
        frames = get_video_landmarks(video_path)
        predictions = []
        for f in frames:
            hand = f.get("right_hand") or f.get("left_hand")
            pose = f.get("pose_landmarks")
            if hand and pose:
                tip = hand.landmark[8] # index tip
                nose = pose.landmark[0] # nose
                dist = np.sqrt((tip.x - nose.x)**2 + (tip.y - nose.y)**2)
                if dist < 0.22:
                    predictions.append("hamnose" if dist < 0.12 else "hamcheek")
        if predictions:
            from collections import Counter
            return {"per_frame": predictions, "final": Counter(predictions).most_common(1)[0][0]}
    except Exception as e:
        pass

    predictions = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"per_frame": [], "final": None}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        loc = detect_head_face_location(frame)
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




