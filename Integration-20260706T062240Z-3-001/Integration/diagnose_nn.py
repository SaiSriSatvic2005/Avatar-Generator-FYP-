import json
import numpy as np
import torch
from train_landmark_nn import HamNoSysSequenceNet

def diagnose():
    dataset_path = "wlasl_landmark_dataset/dataset_landmarks.npz"
    map_path = "wlasl_landmark_dataset/nn_class_mappings.json"
    model_path = "wlasl_landmark_dataset/hamnosys_net_v2.pth"

    data = np.load(dataset_path)
    X = data["tensors"] # (N, 150, 225)

    with open(map_path, "r") as f:
        mappings = json.load(f)

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

    preds_list = []
    with torch.no_grad():
        for i in range(min(20, len(X))):
            t_in = torch.tensor(X[i]).unsqueeze(0)
            preds = model(t_in)
            hs = mappings["handshape"][preds["handshape"].argmax().item()]
            two = mappings["two_handed"][preds["two_handed"].argmax().item()]
            loc = mappings["location"][preds["location"].argmax().item()]
            preds_list.append((two, hs, loc))

    print("[Diagnostic Predictions across 20 samples]:")
    for i, p in enumerate(preds_list):
        print(f"  Sample {i:02d}: two_handed={p[0]}, handshape={p[1]}, location={p[2]}")

if __name__ == "__main__":
    diagnose()
