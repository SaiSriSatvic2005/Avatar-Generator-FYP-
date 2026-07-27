#!/usr/bin/env python3
"""
PyTorch Multi-Head Sequence Classifier for HamNoSys Prediction — v3.

Architecture (matches Kaggle training script v3):
- 2x Conv1d temporal feature extraction + BatchNorm
- 2-Layer Bidirectional GRU with dropout
- Attention-weighted temporal pooling
- LayerNorm + Dropout before classification
- 6 Independent Multi-Head Classifiers:
  1. Handshape
  2. Extended Finger Direction
  3. Palm Orientation
  4. Body Location
  5. Movement
  6. Two-Handed Structure
"""

import os
import sys
import csv
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class LandmarkDataset(Dataset):
    def __init__(self, npz_path, csv_path):
        data = np.load(npz_path)
        self.tensors = torch.tensor(data["tensors"], dtype=torch.float32) # (N, T, 177)
        
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.metadata = list(reader)

        self.handshapes = sorted(list(set(r["handshape"] for r in self.metadata)))
        self.ext_fingers = sorted(list(set(r["ext_finger"] for r in self.metadata)))
        self.palm_oris = sorted(list(set(r["palm_ori"] for r in self.metadata)))
        self.locations = sorted(list(set(r["location"] for r in self.metadata)))
        self.movements = sorted(list(set(r["movement"] for r in self.metadata)))
        self.two_handed_types = sorted(list(set(r["two_handed"] for r in self.metadata)))

        self.hs_map = {v: i for i, v in enumerate(self.handshapes)}
        self.ext_map = {v: i for i, v in enumerate(self.ext_fingers)}
        self.palm_map = {v: i for i, v in enumerate(self.palm_oris)}
        self.loc_map = {v: i for i, v in enumerate(self.locations)}
        self.mov_map = {v: i for i, v in enumerate(self.movements)}
        self.two_handed_map = {v: i for i, v in enumerate(self.two_handed_types)}

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        row = self.metadata[idx]
        return {
            "tensor": self.tensors[idx],
            "handshape": torch.tensor(self.hs_map[row["handshape"]], dtype=torch.long),
            "ext_finger": torch.tensor(self.ext_map[row["ext_finger"]], dtype=torch.long),
            "palm_ori": torch.tensor(self.palm_map[row["palm_ori"]], dtype=torch.long),
            "location": torch.tensor(self.loc_map[row["location"]], dtype=torch.long),
            "movement": torch.tensor(self.mov_map[row["movement"]], dtype=torch.long),
            "two_handed": torch.tensor(self.two_handed_map[row["two_handed"]], dtype=torch.long)
        }


class HamNoSysSequenceNet(nn.Module):
    """v3 architecture — must match Kaggle training script exactly."""

    def __init__(self, input_dim=177, num_hs=11, num_ext=3, num_palm=3,
                 num_loc=8, num_mov=8, num_two=3, dropout=0.3):
        super().__init__()

        # Temporal feature extraction
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

        # Bidirectional GRU
        self.gru = nn.GRU(128, 64, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=dropout)

        # Attention pooling
        self.attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # LayerNorm + Dropout
        self.layer_norm = nn.LayerNorm(128)
        self.dropout = nn.Dropout(dropout)

        # 6 independent classification heads
        self.fc_hs = nn.Linear(128, num_hs)
        self.fc_ext = nn.Linear(128, num_ext)
        self.fc_palm = nn.Linear(128, num_palm)
        self.fc_loc = nn.Linear(128, num_loc)
        self.fc_mov = nn.Linear(128, num_mov)
        self.fc_two = nn.Linear(128, num_two)

    def forward(self, x):
        # x: (B, T, 177)
        x_t = x.transpose(1, 2)                    # (B, 177, T)
        c1 = self.relu(self.bn1(self.conv1(x_t)))   # (B, 128, T)
        c2 = self.relu(self.bn2(self.conv2(c1)))     # (B, 128, T)
        c2 = c2.transpose(1, 2)                      # (B, T, 128)

        gru_out, _ = self.gru(c2)                    # (B, T, 128)

        # Attention-weighted pooling
        attn_weights = self.attention(gru_out)       # (B, T, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        pooled = (gru_out * attn_weights).sum(dim=1) # (B, 128)

        # LayerNorm + Dropout
        pooled = self.layer_norm(pooled)
        pooled = self.dropout(pooled)

        return {
            "handshape": self.fc_hs(pooled),
            "ext_finger": self.fc_ext(pooled),
            "palm_ori": self.fc_palm(pooled),
            "location": self.fc_loc(pooled),
            "movement": self.fc_mov(pooled),
            "two_handed": self.fc_two(pooled),
        }


def train_model(data_dir=os.path.join(_SCRIPT_DIR, "wlasl_landmark_dataset"), epochs=20, batch_size=32):
    npz_path = os.path.join(data_dir, "dataset_landmarks.npz")
    csv_path = os.path.join(data_dir, "metadata.csv")
    
    if not os.path.exists(npz_path) or not os.path.exists(csv_path):
        print(f"[Training Error] Dataset files not found in: {data_dir}")
        print("Please run build_wlasl_landmarks.py first!")
        return

    dataset = LandmarkDataset(npz_path, csv_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    mappings = {
        "handshape": dataset.handshapes,
        "ext_finger": dataset.ext_fingers,
        "palm_ori": dataset.palm_oris,
        "location": dataset.locations,
        "movement": dataset.movements,
        "two_handed": dataset.two_handed_types
    }
    with open(os.path.join(data_dir, "nn_class_mappings.json"), "w", encoding="utf-8") as f:
        json.dump(mappings, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training] Device: {device} | Dataset Samples: {len(dataset)}")

    model = HamNoSysSequenceNet(
        input_dim=177,
        num_hs=len(dataset.handshapes),
        num_ext=len(dataset.ext_fingers),
        num_palm=len(dataset.palm_oris),
        num_loc=len(dataset.locations),
        num_mov=len(dataset.movements),
        num_two=len(dataset.two_handed_types)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct_hs = 0
        correct_two = 0
        total_samples = 0


        for batch in dataloader:
            tensors = batch["tensor"].to(device)
            optimizer.zero_grad()
            
            preds = model(tensors)
            
            loss_hs = criterion(preds["handshape"], batch["handshape"].to(device))
            loss_ext = criterion(preds["ext_finger"], batch["ext_finger"].to(device))
            loss_palm = criterion(preds["palm_ori"], batch["palm_ori"].to(device))
            loss_loc = criterion(preds["location"], batch["location"].to(device))
            loss_mov = criterion(preds["movement"], batch["movement"].to(device))
            loss_two = criterion(preds["two_handed"], batch["two_handed"].to(device))
            
            loss = loss_hs + loss_ext + loss_palm + loss_loc + loss_mov + loss_two
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(tensors)
            correct_hs += (preds["handshape"].argmax(dim=1) == batch["handshape"].to(device)).sum().item()
            correct_two += (preds["two_handed"].argmax(dim=1) == batch["two_handed"].to(device)).sum().item()
            total_samples += len(tensors)

        avg_loss = total_loss / total_samples
        acc_hs = (correct_hs / total_samples) * 100.0
        acc_two = (correct_two / total_samples) * 100.0
        print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {avg_loss:.4f} | Handshape Acc: {acc_hs:.1f}% | 2-Hand Acc: {acc_two:.1f}%")


    model_save_path = os.path.join(data_dir, "hamnosys_net_v2.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"\n[Training Complete] Saved trained Neural Network to: {os.path.abspath(model_save_path)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    train_model(epochs=args.epochs)
