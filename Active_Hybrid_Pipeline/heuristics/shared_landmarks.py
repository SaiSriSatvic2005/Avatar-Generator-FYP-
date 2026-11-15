#!/usr/bin/env python
# coding: utf-8
"""
Shared Landmark Extractor for HamNoSys Pipeline
Unified single-pass landmark extraction for Pose, Face, and Dual Hands (Max 2 hands).
Compensates for selfie camera horizontal mirroring and transforms Viewer space to Signer space.
"""

import os
import cv2
import numpy as np
import mediapipe as mp

_LANDMARK_CACHE = {}

class OneEuroFilter:
    """Adaptive Low-Pass Filter: Reduces landmark jitter without introducing latency."""
    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev = np.array(x, dtype=np.float32)
            self.dx_prev = np.zeros_like(x, dtype=np.float32)
            self.t_prev = t
            return self.x_prev

        dt = max(t - self.t_prev, 1e-4)
        self.t_prev = t

        dx = (np.array(x, dtype=np.float32) - self.x_prev) / dt
        edx = self._alpha(dt, self.d_cutoff) * dx + (1 - self._alpha(dt, self.d_cutoff)) * self.dx_prev
        self.dx_prev = edx

        cutoff = self.min_cutoff + self.beta * np.linalg.norm(edx)
        alpha = self._alpha(dt, cutoff)
        x_filtered = alpha * np.array(x, dtype=np.float32) + (1 - alpha) * self.x_prev
        self.x_prev = x_filtered
        return x_filtered

    def _alpha(self, dt, cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)


class LandmarkPoint:
    """Lightweight MediaPipe-compatible Landmark object."""
    def __init__(self, x, y, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

class LandmarkList:
    """Lightweight container mimicking MediaPipe LandmarkList."""
    def __init__(self, points):
        self.landmark = [LandmarkPoint(p[0], p[1], p[2] if len(p) > 2 else 0.0) for p in points]

class SharedLandmarkExtractor:
    def __init__(self):
        try:
            self.mp_holistic = mp.solutions.holistic
            self.mp_hands = mp.solutions.hands
        except (AttributeError, Exception):
            try:
                import mediapipe.python.solutions.holistic as mp_holistic
                import mediapipe.python.solutions.hands as mp_hands
                self.mp_holistic = mp_holistic
                self.mp_hands = mp_hands
            except Exception as e:
                raise ImportError(f"MediaPipe solutions module initialization failed: {e}")
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.filter_right = OneEuroFilter()
        self.filter_left = OneEuroFilter()
        self._rtm_model = None

    def _get_rtm_model(self):
        if self._rtm_model is None:
            try:
                from rtmlib import Wholebody
                self._rtm_model = Wholebody(mode='balanced')
                print("[SharedLandmarks] Initialized RTMPose Wholebody Model.")
            except Exception as e:
                print(f"[SharedLandmarks] RTMPose unavailable ({e}), using MediaPipe Holistic...")
                self._rtm_model = False
        return self._rtm_model



    def _preprocess_frame(self, frame):
        """Enhance frame contrast using CLAHE."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        enhanced_lab = cv2.merge((l_enhanced, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

    def extract_video_landmarks(self, video_path, mirror_correction=True):
        if video_path in _LANDMARK_CACHE:
            return _LANDMARK_CACHE[video_path]
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[SharedLandmarks] Warning: Cannot open video {video_path}")
            return []

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        frames_data = []
        frame_idx = 0

        rtm = self._get_rtm_model()

        if rtm:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    break
                if frame_idx % 2 != 0:
                    frame_idx += 1
                    continue
                t = frame_idx / fps
                h, w = frame.shape[:2]
                keypoints, scores = rtm(frame)

                if len(keypoints.shape) == 3:
                    keypoints, scores = keypoints[0], scores[0]

                # Scale keypoints back to normalized [0, 1] range relative to original frame
                keypoints[:, 0] /= float(w)
                keypoints[:, 1] /= float(h)


                # keypoints shape: (133, 2), scores shape: (133,)
                l_score = np.mean(scores[91:112]) if (scores is not None and len(scores) >= 133) else 0.0
                r_score = np.mean(scores[112:133]) if (scores is not None and len(scores) >= 133) else 0.0

                l_hand_pts = keypoints[91:112] if (len(keypoints) >= 133 and l_score >= 0.3) else None
                r_hand_pts = keypoints[112:133] if (len(keypoints) >= 133 and r_score >= 0.3) else None


                # One-Euro Filtering
                if l_hand_pts is not None:
                    l_hand_pts = self.filter_left(l_hand_pts, t)
                if r_hand_pts is not None:
                    r_hand_pts = self.filter_right(r_hand_pts, t)

                l_hand_obj = LandmarkList(l_hand_pts) if l_hand_pts is not None else None
                r_hand_obj = LandmarkList(r_hand_pts) if r_hand_pts is not None else None

                body_pts = keypoints[0:17] if len(keypoints) >= 17 else None
                pose_obj = LandmarkList(body_pts) if body_pts is not None else None

                frame_info = {
                    "width": width,
                    "height": height,
                    "pose_landmarks": pose_obj,
                    "pose_world_landmarks": pose_obj,
                    "face_landmarks": None,
                    "left_hand": l_hand_obj,
                    "right_hand": r_hand_obj,
                    "primary_hand": r_hand_obj or l_hand_obj,
                    "primary_handedness": "Right" if r_hand_obj else ("Left" if l_hand_obj else None)
                }
                frames_data.append(frame_info)
                frame_idx += 1

            cap.release()
            _LANDMARK_CACHE[video_path] = frames_data
            return frames_data

        # Fallback to MediaPipe Holistic + Hands (model_complexity=1 for low memory & CPU efficiency)
        with self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic, self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands_detector:

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if mirror_correction:
                    frame = cv2.flip(frame, 1)

                rgb_frame = self._preprocess_frame(frame)
                holistic_res = holistic.process(rgb_frame)
                hands_res = hands_detector.process(rgb_frame)

                frame_info = {
                    "width": width,
                    "height": height,
                    "pose_landmarks": holistic_res.pose_landmarks,
                    "pose_world_landmarks": getattr(holistic_res, "pose_world_landmarks", None),
                    "face_landmarks": holistic_res.face_landmarks,
                    "left_hand": None,
                    "right_hand": None,
                    "hands": []
                }

                if hands_res.multi_hand_landmarks and hands_res.multi_handedness:
                    for lm, handedness in zip(hands_res.multi_hand_landmarks, hands_res.multi_handedness):
                        label = handedness.classification[0].label
                        score = handedness.classification[0].score
                        if label == "Right" and frame_info["right_hand"] is None:
                            frame_info["right_hand"] = lm
                        elif label == "Left" and frame_info["left_hand"] is None:
                            frame_info["left_hand"] = lm

                if frame_info["left_hand"] is None and holistic_res.left_hand_landmarks:
                    frame_info["left_hand"] = holistic_res.left_hand_landmarks
                if frame_info["right_hand"] is None and holistic_res.right_hand_landmarks:
                    frame_info["right_hand"] = holistic_res.right_hand_landmarks

                if frame_info["right_hand"] is None and frame_info["left_hand"] is not None:
                    frame_info["primary_hand"] = frame_info["left_hand"]
                    frame_info["primary_handedness"] = "Left"
                elif frame_info["right_hand"] is not None:
                    frame_info["primary_hand"] = frame_info["right_hand"]
                    frame_info["primary_handedness"] = "Right"
                else:
                    frame_info["primary_hand"] = None
                    frame_info["primary_handedness"] = None

                frames_data.append(frame_info)

        cap.release()
        if len(_LANDMARK_CACHE) > 5:
            _LANDMARK_CACHE.clear()
        _LANDMARK_CACHE[video_path] = frames_data
        return frames_data

# Singleton helper instance
_extractor = SharedLandmarkExtractor()

def get_video_landmarks(video_path, mirror_correction=True):
    return _extractor.extract_video_landmarks(video_path, mirror_correction=mirror_correction)

