import cv2
import numpy as np
import mediapipe as mp
import os
import shutil
import tempfile
try:
    from scipy.signal import savgol_filter
except ImportError:
    import subprocess
    import sys
    print("Installing scipy...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
    from scipy.signal import savgol_filter

def smooth_trajectory(traj, window_size=5, polyorder=2):
    """
    Applies Savitzky-Golay filter to smooth a 2D or 3D trajectory array.
    """
    if traj is None or len(traj) < window_size:
        return traj

    traj = np.array(traj)
    smoothed = np.zeros_like(traj)
    for i in range(traj.shape[1]):
        # Ensure window_size is odd and less than or equal to the sequence length
        ws = window_size if window_size % 2 == 1 else window_size - 1
        if ws > len(traj):
            ws = len(traj) if len(traj) % 2 == 1 else len(traj) - 1
            if ws < 3: # Cannot apply filter with window size < 3 for polyorder 2
                return traj 
        
        smoothed[:, i] = savgol_filter(traj[:, i], ws, polyorder)
    return smoothed

def detect_dominant_hand(video_path):
    """
    Analyzes the video to determine which hand moves more (variance).
    Returns 'left' or 'right' (from the camera's perspective).
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
    # Check first 30 frames to save time
    while cap.isOpened() and frame_count < 30:
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
