"""
DGS Frontal Video Cropper & Upscaler
Isolates the frontal signer view (left 60% of frame) from multi-view Dicta-Sign videos,
upscales from 192x180 to 384x360 with bicubic interpolation for optimal landmark tracking,
and saves the clean frontal videos directly into downloaded_dgs_dataset/videos/.
Original multi-view videos are preserved in downloaded_dgs_dataset/raw_multi_view_videos/.
"""

import os
import cv2
import shutil
import time
from typing import Tuple

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "downloaded_dgs_dataset")
RAW_DIR = os.path.join(DATASET_DIR, "raw_multi_view_videos")
VIDEOS_DIR = os.path.join(DATASET_DIR, "videos")

def crop_video(src_path: str, dst_path: str, target_size: Tuple[int, int] = (384, 360)) -> bool:
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        return False
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 120:
        fps = 25.0
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        h, w = frame.shape[:2]
        # Frontal view boundary: exactly 60% of width (192 for 320x180)
        split_x = int(w * 0.6)
        frontal_crop = frame[:, :split_x]
        
        # Bicubic upscale for crisp hand and finger landmark tracking
        resized = cv2.resize(frontal_crop, target_size, interpolation=cv2.INTER_CUBIC)
        
        if out is None:
            out = cv2.VideoWriter(dst_path, fourcc, fps, target_size)
            
        out.write(resized)
        
    cap.release()
    if out:
        out.release()
        return True
    return False

def main():
    print("=== DGS Frontal Video Batch Cropper & Upscaler ===")
    
    # 1. Prepare raw archive directory
    if not os.path.exists(RAW_DIR):
        print(f"[*] Archiving raw multi-view videos to: {RAW_DIR}")
        os.rename(VIDEOS_DIR, RAW_DIR)
    else:
        print(f"[*] Raw multi-view archive already exists: {RAW_DIR}")

    os.makedirs(VIDEOS_DIR, exist_ok=True)
    
    # 2. Get list of raw videos to process
    raw_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".mp4")]
    total = len(raw_files)
    print(f"[*] Found {total} videos to process...")
    
    start_time = time.time()
    success_count = 0
    
    for i, fname in enumerate(raw_files, start=1):
        src = os.path.join(RAW_DIR, fname)
        dst = os.path.join(VIDEOS_DIR, fname)
        
        if crop_video(src, dst):
            success_count += 1
            
        if i % 100 == 0 or i == total:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            eta = (total - i) / rate if rate > 0 else 0
            print(f"[{i:4d}/{total}] Cropped & upscaled {success_count} videos ({rate:.1f} vid/s, ETA: {eta:.0f}s)")
            
    total_time = time.time() - start_time
    print(f"\n[DONE] Successfully processed {success_count}/{total} videos in {total_time:.1f}s!")
    print(f"Clean frontal videos: {VIDEOS_DIR}")
    print(f"Raw archive videos:   {RAW_DIR}")

if __name__ == "__main__":
    main()
