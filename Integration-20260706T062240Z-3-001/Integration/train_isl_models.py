#!/usr/bin/env python3
"""
Phase 3: Train Modular Neural Networks (Model A & Model B)
Model A: Spatial Handshape Classifier (Frame-Level MLP)
Model B: Movement Trajectory Classifier (1D-CNN) with 10x Data Augmentation

Reference: HYBRID_NN_ARCHITECTURE_AND_TRAINING_SPECIFICATION.md - Section 7
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.interpolate import interp1d

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_NPZ = os.path.join(_SCRIPT_DIR, "isl_dwpose_dataset.npz")
ENCODERS_JSON = os.path.join(_SCRIPT_DIR, "isl_label_encoders.json")

MODEL_A_PTH = os.path.join(_SCRIPT_DIR, "isl_handshape_mlp.pth")
MODEL_B_PTH = os.path.join(_SCRIPT_DIR, "isl_movement_cnn.pth")

# Device selection: Use CUDA if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================================
# Model A: Spatial Handshape Classifier (Frame-Level MLP)
# =====================================================================

class HandshapeMLP(nn.Module):
    """
    Frame-Level Handshape MLP operating on 63D normalized hand coordinates.
    Specification: Section 7.1
    Linear(63 -> 128) -> LayerNorm -> ReLU -> Dropout(0.4)
    Linear(128 -> 64) -> LayerNorm -> ReLU -> Dropout(0.3)
    Linear(64 -> N_classes)
    """
    def __init__(self, in_features=63, num_classes=11, dropout1=0.4, dropout2=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout1),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

# =====================================================================
# Model B: Movement Trajectory Classifier (1D-CNN)
# =====================================================================

class Movement1DCNN(nn.Module):
    """
    Temporal 1D-CNN for velocity trajectory curves (B, 3, 30).
    Specification: Section 7.2
    Conv1d(3 -> 32, k=5, p=2) -> BatchNorm1d -> ReLU -> MaxPool1d(2)
    Conv1d(32 -> 64, k=3, p=1) -> BatchNorm1d -> ReLU -> AdaptiveAvgPool1d(1)
    Flatten -> Linear(64 -> N_classes)
    """
    def __init__(self, in_channels=3, num_classes=15):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(output_size=1)
        )
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x):
        feat = self.conv_block(x) # (B, 64, 1)
        feat = feat.view(feat.size(0), -1) # (B, 64)
        return self.fc(feat)

# =====================================================================
# Model B Trajectory Augmentation (Section 7.3)
# =====================================================================

def augment_trajectory(traj, multiplier=10):
    """
    Augments trajectory (3, 30) using:
      1. Temporal resampling (speed 0.8x - 1.2x)
      2. 3D spatial rotation around Z and Y (-12 to +12 deg)
      3. Gaussian jitter (std=0.005)
      4. Scale perturbation (+-10%)
    """
    augmented = [traj]
    T = traj.shape[1]
    
    for _ in range(multiplier - 1):
        aug = traj.copy()
        
        # 1. Scale perturbation
        scale = np.random.uniform(0.90, 1.10)
        aug = aug * scale
        
        # 2. 3D Spatial rotation
        theta_z = np.radians(np.random.uniform(-12, 12))
        cos_z, sin_z = np.cos(theta_z), np.sin(theta_z)
        R_z = np.array([
            [cos_z, -sin_z, 0],
            [sin_z,  cos_z, 0],
            [0,      0,     1]
        ])
        aug = np.dot(R_z, aug)
        
        # 3. Gaussian jitter
        noise = np.random.normal(0, 0.005, size=aug.shape)
        aug = aug + noise
        
        # 4. Temporal warp
        warp_rate = np.random.uniform(0.85, 1.15)
        new_len = max(5, int(T * warp_rate))
        t_orig = np.linspace(0, 1, T)
        t_warp = np.linspace(0, 1, new_len)
        f = interp1d(t_orig, aug, axis=1, kind='linear', fill_value='extrapolate')
        warped = f(t_warp)
        
        # Resample back to T=30
        t_new = np.linspace(0, 1, T)
        f_res = interp1d(np.linspace(0, 1, new_len), warped, axis=1, kind='linear', fill_value='extrapolate')
        aug = f_res(t_new)
        
        augmented.append(aug.astype(np.float32))
        
    return augmented

# =====================================================================
# Training Routines
# =====================================================================

def train_model_a(X, y, num_classes, epochs=35, batch_size=32, lr=1e-3):
    print("\n" + "=" * 60)
    print(f"  Training Model A: Spatial Handshape Classifier (MLP)")
    print(f"  Input: {X.shape} | Classes: {num_classes} | Device: {DEVICE}")
    print("=" * 60)
    
    # Train / Val Split (80 / 20)
    indices = np.random.permutation(len(X))
    split = int(0.85 * len(X))
    train_idx, val_idx = indices[:split], indices[split:]
    
    X_train, y_train = torch.FloatTensor(X[train_idx]), torch.LongTensor(y[train_idx])
    X_val, y_val = torch.FloatTensor(X[val_idx]), torch.LongTensor(y[val_idx])
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    
    model = HandshapeMLP(in_features=63, num_classes=num_classes).to(DEVICE)
    
    # Balanced inverse-frequency class weights
    class_counts = np.bincount(y, minlength=num_classes)
    weights = len(y) / (num_classes * np.maximum(class_counts, 1).astype(np.float32))
    weights = np.clip(weights, 0.2, 5.0)
    weights = weights / np.mean(weights)
    weight_tensor = torch.FloatTensor(weights).to(DEVICE)

    # Label smoothing epsilon = 0.05 with class weights
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            # Apply light joint jitter data augmentation
            noise = torch.randn_like(batch_x) * 0.004
            optimizer.zero_grad()
            preds = model(batch_x + noise)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(batch_y)
            correct += (preds.argmax(dim=1) == batch_y).sum().item()
            total += len(batch_y)
            
        scheduler.step()
        train_acc = correct / max(total, 1)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                preds = model(batch_x)
                val_correct += (preds.argmax(dim=1) == batch_y).sum().item()
                val_total += len(batch_y)
                
        val_acc = val_correct / max(val_total, 1)
        if val_acc > best_acc or epoch == epochs:
            best_acc = max(best_acc, val_acc)
            torch.save(model.state_dict(), MODEL_A_PTH)
            
        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] Loss: {train_loss/total:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")
            
    print(f"Model A Handshape MLP saved -> {MODEL_A_PTH} (Best Val Acc: {best_acc*100:.2f}%)")
    return model

def train_model_b(X, y, num_classes, epochs=30, batch_size=32, lr=5e-4):
    print("\n" + "=" * 60)
    print(f"  Training Model B: Movement Trajectory Classifier (1D-CNN)")
    print(f"  Base Trajectories: {len(X)} | Target Augmented: ~{len(X)*10}")
    print("=" * 60)
    
    # Apply synthetic trajectory augmentation (10x Multiplier)
    X_aug = []
    y_aug = []
    for traj, label in zip(X, y):
        aug_trajs = augment_trajectory(traj, multiplier=10)
        for t in aug_trajs:
            X_aug.append(t)
            y_aug.append(label)
            
    X_aug = np.array(X_aug, dtype=np.float32)
    y_aug = np.array(y_aug, dtype=np.int64)
    print(f"  Augmented dataset size: {X_aug.shape}")
    
    indices = np.random.permutation(len(X_aug))
    split = int(0.85 * len(X_aug))
    train_idx, val_idx = indices[:split], indices[split:]
    
    X_train, y_train = torch.FloatTensor(X_aug[train_idx]), torch.LongTensor(y_aug[train_idx])
    X_val, y_val = torch.FloatTensor(X_aug[val_idx]), torch.LongTensor(y_aug[val_idx])
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    
    model = Movement1DCNN(in_channels=3, num_classes=num_classes).to(DEVICE)
    
    # Balanced movement class weights
    mv_counts = np.bincount(y_aug, minlength=num_classes)
    mv_weights = len(y_aug) / (num_classes * np.maximum(mv_counts, 1).astype(np.float32))
    mv_weights = np.clip(mv_weights, 0.2, 5.0)
    mv_weights = mv_weights / np.mean(mv_weights)
    weight_tensor = torch.FloatTensor(mv_weights).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(batch_y)
            correct += (preds.argmax(dim=1) == batch_y).sum().item()
            total += len(batch_y)
            
        train_acc = correct / max(total, 1)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                preds = model(batch_x)
                val_correct += (preds.argmax(dim=1) == batch_y).sum().item()
                val_total += len(batch_y)
                
        val_acc = val_correct / max(val_total, 1)
        if val_acc > best_acc or epoch == epochs:
            best_acc = max(best_acc, val_acc)
            torch.save(model.state_dict(), MODEL_B_PTH)
            
        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] Loss: {train_loss/total:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")
            
    print(f"Model B Movement 1D-CNN saved -> {MODEL_B_PTH} (Best Val Acc: {best_acc*100:.2f}%)")
    return model

def main():
    if not os.path.exists(DATASET_NPZ):
        raise FileNotFoundError(f"Missing {DATASET_NPZ}. Run extract_isl_dwpose_dataset.py first.")
        
    data = np.load(DATASET_NPZ)
    X_hs = data['handshape_features']
    y_hs = data['handshape_labels']
    X_mv = data['movement_trajectories']
    y_mv = data['movement_labels']
    
    with open(ENCODERS_JSON, "r") as f:
        meta = json.load(f)
        
    n_hs_classes = len(meta['handshape_classes'])
    n_mv_classes = len(meta['movement_classes'])
    
    print(f"Loaded ISL DWPose Dataset: {len(X_hs)} handshape frames, {len(X_mv)} movement trajectories")
    
    train_model_a(X_hs, y_hs, num_classes=n_hs_classes, epochs=30)
    train_model_b(X_mv, y_mv, num_classes=n_mv_classes, epochs=25)
    
    print("\n" + "=" * 60)
    print("  Phase 3 Training Complete! All neural network weights saved.")
    print("=" * 60)

if __name__ == "__main__":
    main()
