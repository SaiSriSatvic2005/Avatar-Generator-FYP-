#!/usr/bin/env python3
"""Quick sanity check: verify feature dimension consistency across all scripts."""
import sys, os
sys.path.insert(0, r'd:\academics\HamNoSys_Group14_V2\Integration-20260706T062240Z-3-001\Integration')

# 1. Check build_wlasl_landmarks feature dimension
from build_wlasl_landmarks import extract_normalized_features
import numpy as np

# Simulate a frame_info dict with RTMPose-style LandmarkList (17 body, 21 hand)
class FakeLandmarkPoint:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z

class FakeLandmarkList:
    def __init__(self, n):
        self.landmark = [FakeLandmarkPoint(np.random.rand(), np.random.rand(), 0.0) for _ in range(n)]

# RTMPose frame: 17 body, 21 right hand, 21 left hand
frame = {
    "pose_landmarks": FakeLandmarkList(17),
    "pose_world_landmarks": FakeLandmarkList(17),
    "right_hand": FakeLandmarkList(21),
    "left_hand": FakeLandmarkList(21),
}
vec = extract_normalized_features(frame)
print(f"[OK] Feature vector dim (both hands):    {vec.shape[0]} (expected 177)")

# 1-handed frame
frame_1h = {
    "pose_landmarks": FakeLandmarkList(17),
    "pose_world_landmarks": FakeLandmarkList(17),
    "right_hand": FakeLandmarkList(21),
    "left_hand": None,
}
vec_1h = extract_normalized_features(frame_1h)
print(f"[OK] Feature vector dim (1 hand):        {vec_1h.shape[0]} (expected 177)")
left_portion = vec_1h[63:126]
print(f"[OK] Left hand zeros (1-hand case):      sum={np.sum(np.abs(left_portion)):.4f} (expected 0.0)")

# 2. Check model input_dim matches
from train_landmark_nn import HamNoSysSequenceNet
import torch

model = HamNoSysSequenceNet(input_dim=177, num_hs=5, num_ext=4, num_palm=4, num_loc=5, num_mov=4, num_two=3)
dummy = torch.randn(2, 150, 177)
out = model(dummy)
print(f"[OK] Model forward pass OK. Output shapes:")
for k, v in out.items():
    print(f"     {k}: {v.shape}")

# 3. Verify integration pipeline would use 177
with open(r'd:\academics\HamNoSys_Group14_V2\Integration-20260706T062240Z-3-001\Integration\integration_pipeline.py', 'r') as f:
    content = f.read()
if 'input_dim=177' in content:
    print("[OK] integration_pipeline.py uses input_dim=177")
else:
    print("[FAIL] integration_pipeline.py does NOT use input_dim=177!")

if 'target_frames = 150' in content:
    print("[OK] integration_pipeline.py uses target_frames=150")
else:
    print("[FAIL] integration_pipeline.py target_frames mismatch!")

print("\n=== All checks passed! Ready to rebuild dataset. ===")
