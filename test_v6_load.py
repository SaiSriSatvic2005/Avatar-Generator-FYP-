#!/usr/bin/env python3
import os
import sys
import cv2
import json
import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import pdist

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
INTEGRATION_DIR = os.path.join(ROOT_DIR, "Integration-20260706T062240Z-3-001", "Integration")

# 1. Model Definitions
class ResidualBlock_V6(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        res = x
        x = torch.nn.functional.mish(self.ln1(self.fc1(x)))
        x = self.drop(x)
        x = self.ln2(self.fc2(x))
        return torch.nn.functional.mish(x + res)

class SkeletalHandResNet_V6(nn.Module):
    def __init__(self, in_features=113, hidden_dim=256, num_classes=12):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish())
        self.b1 = ResidualBlock_V6(hidden_dim, 0.2)
        self.b2 = ResidualBlock_V6(hidden_dim, 0.2)
        self.head = nn.Sequential(nn.Linear(hidden_dim, 128), nn.Mish(), nn.Dropout(0.25), nn.Linear(128, num_classes))
    def forward(self, x):
        return self.head(self.b2(self.b1(self.proj(x))))

class AttentionBiGRU_V6(nn.Module):
    def __init__(self, in_channels=18, hidden_dim=128, num_classes=19, num_heads=4):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(in_channels, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.Mish())
        self.gru = nn.GRU(64, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=0.25)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim * 2, num_heads=num_heads, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128), nn.LayerNorm(128), nn.Mish(), nn.Dropout(0.3), nn.Linear(128, num_classes)
        )
    def forward(self, x):
        if x.dim() == 3 and x.shape[1] != 18 and x.shape[2] == 18:
            x = x.transpose(1, 2)
        h = self.conv(x).transpose(1, 2)
        gru_out, _ = self.gru(h)
        attn_out, _ = self.attn(gru_out, gru_out, gru_out)
        pooled = torch.mean(attn_out, dim=1)
        return self.classifier(pooled)

def test_load():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enc_path = os.path.join(INTEGRATION_DIR, 'v6_encoders.json')
    with open(enc_path, 'r') as f:
        enc = json.load(f)
    print(f"[*] Loaded encoders: {len(enc['handshape_classes'])} Handshape classes, {len(enc['movement_classes'])} Movement classes.")

    model_a = SkeletalHandResNet_V6(in_features=113, num_classes=len(enc['handshape_classes'])).to(device)
    model_a.load_state_dict(torch.load(os.path.join(INTEGRATION_DIR, 'v6_dgs_handshape_resnet.pth'), map_location=device))
    model_a.eval()
    print("[*] Successfully loaded Model A (SkeletalHandResNet_V6)!")

    model_b = AttentionBiGRU_V6(in_channels=18, num_classes=len(enc['movement_classes'])).to(device)
    model_b.load_state_dict(torch.load(os.path.join(INTEGRATION_DIR, 'v6_dgs_movement_attention_bigru.pth'), map_location=device))
    model_b.eval()
    print("[*] Successfully loaded Model B (AttentionBiGRU_V6)!")

    # Test dummy inference
    dummy_x_hs = torch.randn(1, 113).to(device)
    out_a = model_a(dummy_x_hs)
    pred_hs = enc['handshape_classes'][out_a.argmax().item()]
    print(f"[*] Test inference Model A: pred class = {pred_hs}")

    dummy_x_mv = torch.randn(1, 18, 32).to(device)
    out_b = model_b(dummy_x_mv)
    pred_mv = enc['movement_classes'][out_b.argmax().item()]
    print(f"[*] Test inference Model B: pred class = {pred_mv}")

if __name__ == "__main__":
    test_load()
