# Guide to Retraining the HamNoSys Neural Network

Since you want to execute training manually so you can inspect terminal logs and progress directly, follow the steps below in your terminal.

---

## Step 1: Build the Dataset
Extract landmarks from your WLASL/video dataset into normalized NumPy tensors and metadata CSVs:

```powershell
cd d:\academics\HamNoSys_Group14_V2\Integration-20260706T062240Z-3-001\Integration
python build_wlasl_landmarks.py --max_samples 500
```

*(You can adjust `--max_samples` depending on how many videos you want to extract features from).*

---

## Step 2: Run the Neural Network Training
Train the multi-head PyTorch model (`HamNoSysSequenceNet`):

```powershell
python train_landmark_nn.py --epochs 30
```

---

## What Happens During Training:
1. `build_wlasl_landmarks.py` reads `gloss_to_hamnosys_dict.json` and maps gloss labels to 6 HamNoSys target heads (`handshape`, `ext_finger`, `palm_ori`, `location`, `movement`, `two_handed`).
2. Normalized 177-dimensional landmark vectors are saved to `wlasl_landmark_dataset/dataset_landmarks.npz`.
3. `train_landmark_nn.py` outputs per-epoch loss, handshape accuracy, and two-handed detection accuracy.
4. The trained weights are automatically saved to `wlasl_landmark_dataset/hamnosys_net_v2.pth` and `nn_class_mappings.json`.
5. `app.py` and `integration_pipeline.py` will automatically load the new `hamnosys_net_v2.pth` model at runtime.
