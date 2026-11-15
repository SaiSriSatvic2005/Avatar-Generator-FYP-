import cv2
import numpy as np
import mediapipe as mp
import os
import shutil
import tempfile
try:
    from scipy.signal import savgol_filter
except ImportError:
    savgol_filter = None

def smooth_trajectory(traj, window_size=5, polyorder=2):
    """
    Applies Savitzky-Golay filter (or moving average fallback) to smooth a 2D or 3D trajectory array.
    """
    if traj is None or len(traj) < window_size:
        return traj

    traj = np.array(traj, dtype=np.float32)
    smoothed = np.zeros_like(traj)
    for i in range(traj.shape[1]):
        col = traj[:, i]
        if savgol_filter is not None:
            ws = window_size if window_size % 2 == 1 else window_size - 1
            if ws > len(traj):
                ws = len(traj) if len(traj) % 2 == 1 else len(traj) - 1
            if ws >= 3:
                smoothed[:, i] = savgol_filter(col, ws, min(polyorder, ws - 1))
                continue
        # Pure NumPy moving average fallback
        kernel = np.ones(min(window_size, len(col))) / min(window_size, len(col))
        smoothed[:, i] = np.convolve(col, kernel, mode='same')
    return smoothed

def detect_dominant_hand(video_path):
    """
    Analyzes the video to determine which hand moves more (variance).
    Returns 'left', 'right', or 'both' (from the camera's perspective).
    Returns 'both' when both hands show significant movement (two-handed sign).
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "right" # Default fallback

    left_wrist_pts = []
    right_wrist_pts = []

    frame_count = 0
    max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 60)
    sample_limit = min(max_frames, 60)  # Sample up to 60 frames for better coverage
    
    while cap.isOpened() and frame_count < sample_limit:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(frame_rgb)

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            lw = lm[mp_pose.PoseLandmark.LEFT_WRIST]
            rw = lm[mp_pose.PoseLandmark.RIGHT_WRIST]

            left_wrist_pts.append((lw.x, lw.y, lw.z))
            right_wrist_pts.append((rw.x, rw.y, rw.z))

        frame_count += 1
    
    cap.release()
    pose.close()

    if len(left_wrist_pts) < 5:
        return "right"

    left_var = np.var(left_wrist_pts, axis=0).sum()
    right_var = np.var(right_wrist_pts, axis=0).sum()

    # Minimum variance threshold: hand must actually be moving
    min_var_threshold = 1e-4
    left_active = left_var > min_var_threshold
    right_active = right_var > min_var_threshold
    
    # Both hands active: if variance ratio is within 3:1, treat as two-handed
    if left_active and right_active:
        ratio = max(left_var, right_var) / max(min(left_var, right_var), 1e-8)
        if ratio < 3.0:
            return "both"

    return "left" if left_var > right_var else "right"

def get_mirrored_video(input_path):
    """
    Creates a temporary mirrored version of the video.
    Returns the path to the temporary video.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return input_path
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    temp_dir = tempfile.gettempdir()
    base_name = os.path.basename(input_path)
    out_path = os.path.join(temp_dir, "mirrored_" + base_name)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        mirrored_frame = cv2.flip(frame, 1)
        out.write(mirrored_frame)
        
    cap.release()
    out.release()
    
    return out_path
