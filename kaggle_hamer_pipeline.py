"""
Kaggle HaMeR 3D Hand Mesh & V4 Hybrid Sign Language Pipeline
Platform: Kaggle Notebook with GPU (Tesla T4 or P100)
"""

import os
import sys
import cv2
import glob
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Device] Using compute hardware: {device}")

# =====================================================================
# 1. Models Definition: 3D Handshape ResNet & 3D Movement BiGRU
# =====================================================================

class HandKinematicResNet(nn.Module):
    """3D Handshape Classifier on HaMeR 84D Kinematic Vectors (B, 84)."""
    def __init__(self, in_features=84, num_classes=12):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        self.res1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128)
        )
        self.res2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128)
        )
        self.out_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        x = self.in_proj(x)
        x = x + self.res1(x)
        x = x + self.res2(x)
        return self.out_head(x)

class MovementBiGRU(nn.Module):
    """3D Movement Trajectory Classifier on 3D velocity curves (B, 3, 30)."""
    def __init__(self, in_channels=3, hidden_size=64, num_classes=14):
        super().__init__()
        self.bigru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.25
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        x = x.transpose(1, 2) # (B, 30, 3)
        out, _ = self.bigru(x)
        pooled = torch.max(out, dim=1)[0]
        return self.classifier(pooled)

# =====================================================================
# 2. Main Kaggle Extraction & Training Flow
# =====================================================================

def main():
    # 1. Dataset Resolution
    candidate_dirs = [
        "/kaggle/input/isl-sign-dataset",
        "/kaggle/input/isl_dataset",
        "/kaggle/input"
    ]
    videos_dir = None
    labels_csv = None
    
    for c in candidate_dirs:
        v_matches = glob.glob(f"{c}/**/ISL_videos", recursive=True)
        l_matches = glob.glob(f"{c}/**/isl_ground_truth_labels.csv", recursive=True)
        if len(v_matches) > 0 and len(l_matches) > 0:
            videos_dir = v_matches[0]
            labels_csv = l_matches[0]
            break
            
    if not videos_dir or not os.path.exists(videos_dir):
        print(f"[Error] Could not find ISL_videos directory in {candidate_dirs}")
        return

    print(f"[Dataset] Videos Directory: {videos_dir}")
    print(f"[Dataset] Labels CSV: {labels_csv}")
    
    df = pd.read_csv(labels_csv).dropna(subset=['handshape', 'movement']).reset_index(drop=True)
    print(f"[Dataset] Loaded {len(df)} labeled video entries.")

if __name__ == '__main__':
    main()
