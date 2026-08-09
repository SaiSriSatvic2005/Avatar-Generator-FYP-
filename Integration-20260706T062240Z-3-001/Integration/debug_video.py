import sys, os, numpy as np
sys.path.insert(0, r'd:\academics\HamNoSys_Group14_V2\Integration-20260706T062240Z-3-001\Integration')
from shared_landmarks import get_video_landmarks
from build_wlasl_landmarks import extract_normalized_features

v_apple = r'd:\academics\HamNoSys_Group14_V2\Integration-20260706T062240Z-3-001\Integration\WLASL_videos\archive\videos\00381.mp4'

frames = get_video_landmarks(v_apple)
print('Total frames extracted:', len(frames))

l_hands_found = sum(1 for f in frames if f['left_hand'] is not None)
r_hands_found = sum(1 for f in frames if f['right_hand'] is not None)
print(f'Right hands detected: {r_hands_found}/{len(frames)}')
print(f'Left hands detected:  {l_hands_found}/{len(frames)}')

vecs = [extract_normalized_features(f) for f in frames]
left_hand_energy = [np.sum(np.abs(v[63:126])) for v in vecs]
print('Left hand feature magnitude per frame:', [round(e, 2) for e in left_hand_energy])
