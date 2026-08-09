#!/usr/bin/env python3
import os
import json
import numpy as np
import torch
from shared_landmarks import get_video_landmarks
from build_wlasl_landmarks import extract_normalized_features
from train_landmark_nn import HamNoSysSequenceNet

def run_quick_neural_test(video_path):
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(_SCRIPT_DIR, "wlasl_landmark_dataset", "hamnosys_net_v2.pth")
    map_path = os.path.join(_SCRIPT_DIR, "wlasl_landmark_dataset", "nn_class_mappings.json")

    with open(map_path, "r", encoding="utf-8") as f:
        mappings = json.load(f)

    print(f"[QuickNeuralTest] Extracting landmarks from WLASL video: {video_path}...")
    frames = get_video_landmarks(video_path)
    print(f"[QuickNeuralTest] Extracted {len(frames)} frames.")

    frame_vecs = [extract_normalized_features(f) for f in frames]
    frame_matrix = np.array(frame_vecs, dtype=np.float32)

    target_frames = 150
    T_curr, D = frame_matrix.shape
    if T_curr < target_frames:
        pad_width = ((0, target_frames - T_curr), (0, 0))
        padded_matrix = np.pad(frame_matrix, pad_width, mode="edge")
    else:
        padded_matrix = frame_matrix[:target_frames]

    tensor_in = torch.tensor(padded_matrix).unsqueeze(0) # (1, 150, 225)

    model = HamNoSysSequenceNet(
        input_dim=225,
        num_hs=len(mappings["handshape"]),
        num_ext=len(mappings["ext_finger"]),
        num_palm=len(mappings["palm_ori"]),
        num_loc=len(mappings["location"]),
        num_mov=len(mappings["movement"]),
        num_two=len(mappings["two_handed"])
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        preds = model(tensor_in)

    hs = mappings["handshape"][preds["handshape"].argmax().item()]
    ext = mappings["ext_finger"][preds["ext_finger"].argmax().item()]
    palm = mappings["palm_ori"][preds["palm_ori"].argmax().item()]
    loc = mappings["location"][preds["location"].argmax().item()]
    mov = mappings["movement"][preds["movement"].argmax().item()]
    two = mappings["two_handed"][preds["two_handed"].argmax().item()]

    sequence = []
    if two and two != "none":
        sequence.append(two)

    sequence.extend([hs, ext, palm, loc])
    if mov and mov != "none":
        sequence.append(mov)

    hamnosys_str = " ".join(sequence)
    return hamnosys_str, {
        "two_handed": two,
        "handshape": hs,
        "ext_finger": ext,
        "palm_ori": palm,
        "location": loc,
        "movement": mov
    }

if __name__ == "__main__":
    import sys
    vpath = sys.argv[1] if len(sys.argv) > 1 else "WLASL_videos/archive/videos/12312.mp4"
    ham_str, components = run_quick_neural_test(vpath)
    print("\n=======================================================")
    print(f" INPUT VIDEO: {vpath}")
    print(f" PREDICTED HAMNOSYS TAGS: {ham_str}")
    print(f" BREAKDOWN: {components}")
    print("=======================================================\n")
