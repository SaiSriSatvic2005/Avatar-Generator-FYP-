# Comprehensive Installation & User Guide
## Universal Sign Language to 3D Avatar System (ISL & DGS)

This document provides complete, step-by-step instructions for installing, configuring, and executing the **Sign Language to 3D Avatar System** across Windows, Linux, and macOS environments.

---

## 📋 System Requirements

### Hardware Requirements
- **CPU**: Intel Core i5 / AMD Ryzen 5 (4+ physical cores) or Apple Silicon M-series.
- **RAM**: Minimum 8 GB (16 GB recommended for HaMeR 3D mesh reconstruction).
- **GPU (Optional but Recommended)**: NVIDIA GPU with CUDA 11.8 or 12.1 (4 GB+ VRAM). The system includes automatic CPU fallbacks if no GPU is detected.
- **Disk Space**: ~4.5 GB free space (includes Python packages, MANO hand models, and pretrained checkpoints).

### Software Requirements
- **Operating System**: Windows 10/11 (64-bit), Ubuntu 20.04/22.04 LTS, or macOS 12+.
- **Python**: Version **3.10.x** or **3.11.x** (Python 3.12 is not recommended due to upstream C++ wheel compatibility with MediaPipe 0.10.x).
- **Git**: Git 2.30+ installed.

---

## 🚀 Step-by-Step Installation

### Step 1: Clone the Repository
Open a terminal (or PowerShell on Windows) and clone the repository:
```bash
git clone https://github.com/SaiSriSatvic2005/Avatar-Generator-FYP-.git
cd Avatar-Generator-FYP-
```

### Step 2: Configure Virtual Environment

#### Option A: Using Python `venv` (Standard)
**Windows (PowerShell / Command Prompt):**
```powershell
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Option B: Using Anaconda / Miniconda
```bash
conda create -n avatar-fyp python=3.10 -y
conda activate avatar-fyp
```

---

### Step 3: Upgrade Packaging Tools & Install Core Packages
Run:
```bash
python -m pip install --upgrade pip setuptools wheel
```

Install all required production dependencies:
```bash
pip install -r requirements.txt
```

#### Verifying PyTorch with CUDA (Optional GPU Acceleration)
If you have an NVIDIA GPU, verify that PyTorch detects CUDA:
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```
*If CUDA is False on a machine with an NVIDIA GPU, reinstall PyTorch matching your CUDA version:*
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

### Step 4: Download HaMeR 3D Mesh Assets & Weights
The system uses the HaMeR (Hand Mesh Recovery) neural network to reconstruct 3D MANO meshes. Download the required asset bundle and checkpoints using our automated utility:
```bash
python download_hamer_assets.py
```
This utility:
1. Validates the `_DATA/` directory structure.
2. Downloads the MANO hand model definitions and SMPL parameters.
3. Downloads the pretrained HaMeR model checkpoint (`hamer_ckpts.tar.gz`).
4. Extracts and places checkpoints into `_DATA/hamer_ckpts/`.

---

## 🖥️ Running the System

### Mode 1: Interactive Full-Stack Web Application (Recommended)
Launch the Flask backend server:
```bash
python webapp/app.py
```
- Open your web browser and navigate to: **`http://localhost:5000`**
- Upload any `.mp4`, `.mov`, or `.avi` sign language video.
- Click **"Process & Animate"** to view real-time synchronized:
  1. Input Sign Video Playback
  2. 10-Module Phonetic Diagnostic Matrix
  3. Interactive CWASA 3D WebGL Avatar performing the sign

### Mode 2: Command-Line Interface (CLI Video Translation)
To translate a video directly to SiGML XML via the command line:
```bash
python translate_video_to_sigml.py --input "ISL_videos/sample.mp4" --output "output_sign.sigml"
```

### Mode 3: Automated Benchmark & Accuracy Evaluation
To evaluate system accuracy across the full 30-video validation benchmark:
```bash
python benchmark_v8_evaluation.py
```
This outputs a detailed quantitative performance matrix reporting:
- Base Handshape Accuracy (%)
- Finger Direction Precision (%)
- Palm Orientation Cosine Distance
- CFG Grammar Syntactic Validity (%)
- End-to-End Pipeline Latency per frame (ms)

---

## 🐳 Docker Deployment (Optional)

You can also run the complete application inside a standardized Docker container:

```bash
# 1. Build Docker image
docker build -t avatar-generator-fyp .

# 2. Run container mapping port 5000
docker run -d -p 5000:5000 --name avatar-app avatar-generator-fyp

# 3. Access in browser
# Navigate to http://localhost:5000
```

---

## ❓ Troubleshooting & FAQs

### 1. `ImportError: cannot import name 'solve_triangular' from 'scipy.linalg'`
- **Cause**: Incompatible SciPy/NumPy versions.
- **Fix**: Reinstall with:
  ```bash
  pip install "numpy<2.0.0" "scipy>=1.11.0,<1.14.0"
  ```

### 2. `MediaPipe Holistic tracking failed / missing landmarks`
- **Cause**: Video lighting is too dark or performer is out of frame.
- **Fix**: Ensure the sign performer's hands, shoulders, and face are clearly visible. The hybrid pipeline will automatically fall back to HaMeR 3D mesh inference if 2D landmarks drop below threshold.

### 3. `CORS error when opening gui_clean.html directly`
- **Cause**: Browsers block local file access (`file:///`) for WebGL asset loading (CWASA avatars).
- **Fix**: Run via the Flask server (`python webapp/app.py`) or start a local HTTP server:
  ```bash
  python -m http.server 8000
  ```
  Then open `http://localhost:8000/gui_clean.html`.
