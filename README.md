# Universal Sign Language to 3D Avatar System (ISL & DGS)
### 3D Hand Mesh Estimation (HaMeR), Hybrid Neural Sequence Modeling & HamNoSys/SiGML Synthesis for Sign Language Animation

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)](https://developers.google.com/mediapipe)
[![HaMeR](https://img.shields.io/badge/HaMeR-3D%20Mesh-purple.svg)](https://github.com/geopavlakos/hamer)
[![CWASA](https://img.shields.io/badge/CWASA-3D%20WebGL%20Avatar-green.svg)](http://vhg.cmp.uea.ac.uk/tech/jas/cwasa/)
[![Status](https://img.shields.io/badge/Status-FYP%20Final%20Submission-brightgreen.svg)]()

---

## 📌 Project Overview & Abstract

Sign language is the primary medium of communication for over 70 million deaf and hard-of-hearing individuals worldwide. However, automated translation between spoken/written language and sign language has historically suffered from 2D depth ambiguities, finger self-occlusion, and poor spatial reproducibility.

This **Final Year Project (Group 14)** presents an end-to-end, multi-stage sign language translation and 3D avatar animation platform. Our system takes continuous sign language video streams in **Indian Sign Language (ISL)** and **German Sign Language (DGS)**, reconstructs robust **3D MANO hand meshes using HaMeR (Hand Mesh Recovery)** alongside **MediaPipe 177-dimensional skeletal landmarks**, and feeds these spatial representations into a **Hybrid Deep Neural Network (BiGRU + 1D-CNN + Self-Attention)**.

The network predicts six discrete phonetic sign components, which are parsed through a validated **Context-Free Grammar (CFG)** compiler into **Hamburg Notation System (HamNoSys)** and **Signing Gesture Markup Language (SiGML)**. The resulting XML instructions drive a real-time **JASigning CWASA 3D WebGL Avatar**, producing seamless, anatomically accurate sign language synthesis at 60 FPS in any standard web browser.

---

## 👥 Project Team & Contribution Matrix (Group 14)

| Member | Academic & System Role | Core Technical Ownership | Primary Deliverables |
| :--- | :--- | :--- | :--- |
| **Vivin Rakul** (`@vivin888`) | **Co-Lead: Deep Learning Systems, Model Generation & Pipeline Integration** | • 3D Hand Mesh integration (HaMeR & MANO 778-vertex surface extraction)<br>• Self-Attention mechanism & Conv1D temporal feature extractor<br>• End-to-end pipeline integration (`local_hamer_pipeline.py`)<br>• Benchmark validation & multi-version accuracy evaluation | • 44 Commits<br>• Equal Technical Weightage<br>• Core Pipeline & Model Generation |
| **Sai Sri Satvic** (`@SaiSriSatvic2005`) | **Co-Lead: Sequence Modeling, Cloud Architecture & SiGML Synthesis** | • Bidirectional GRU (BiGRU) temporal sequence architecture<br>• Context-Free Grammar (CFG) phonetic compiler<br>• SiGML XML synthesis engine & CWASA WebGL player integration<br>• Flask REST API backend & cloud containerization | • 44 Commits<br>• Equal Technical Weightage<br>• Architecture & Sequence Engine |
| **Snigdha** (`@snigdhasirivalli`) | **Lead: 3D Spatial Geometry, Multi-Task Loss & Avatar Rendering** | • Multi-task cross-entropy loss balancing across 6 classification heads<br>• 3D finger direction & palm normal vector orientation modeling<br>• WebGL avatar viewport integration & motion replay dynamics<br>• Subsystem token precision validation suite (`tests/`) | • 34 Commits<br>• Geometry & Loss Specialist<br>• Precision Suite & 3D Orientation |
| **Vijay Saravana Pandi** (`@VijaySaravanaPandi`) | **Lead: Video Ingestion, Feature Engineering & Spatial Datasets** | • 177-D wrist-centered, scale-invariant landmark engineering<br>• Anatomical body & facial anchor detection (Da Vinci facial third ratios)<br>• Dataset ingestion & ground-truth alignment for ISL and DGS<br>• Multi-threaded frame preprocessing & video ingestion pipeline | • 34 Commits<br>• Ingestion & Feature Specialist<br>• Dataset Manifest & Face/Body |

---

## 🏗️ System Architecture & Workflow Pipeline

```text
+-------------------------------------------------------------------------------------------------+
|                                     END-TO-END PIPELINE WORKFLOW                                |
+-------------------------------------------------------------------------------------------------+

 [ Raw Sign Video ]  (ISL / DGS Multi-angle Video Stream, 30 FPS)
         |
         v
+-------------------------------------------------------------------------------------------------+
| STAGE 1: DUAL-STREAM 3D SPATIAL & SKELETAL FEATURE EXTRACTION                                   |
|  - MediaPipe Holistic: 177-D Wrist-Centered, Scale-Invariant Landmarks (Body, Hands, Face)      |
|  - HaMeR (Hand Mesh Recovery): 3D MANO Mesh (778 Vertices, 21 Joint Rotations, 3D Depth)       |
+-------------------------------------------------------------------------------------------------+
         |
         v
+-------------------------------------------------------------------------------------------------+
| STAGE 2: HYBRID DEEP NEURAL NETWORK (BiGRU + Conv1D + Self-Attention)                           |
|  - 1D Temporal Convolutions (Short-term phonetic motion primitive extraction)                   |
|  - Multi-Head Self-Attention (Phonetic inflection frame weighting & keyframe focus)             |
|  - Bidirectional GRU (BiGRU) (Long-range forward/backward temporal sequence modeling)           |
+-------------------------------------------------------------------------------------------------+
         |
         +---> [ Head 1: Base Handshape (Flat, Fist, Pinch, Index, Victory, Open) ]
         +---> [ Head 2: Extended Finger Direction (18 3D Spherical Vectors) ]
         +---> [ Head 3: Palm Orientation (Normal Vectors & Plane Rotations) ]
         +---> [ Head 4: Body Location Anchors (Face Thirds, Chest, Shoulders, Neutral) ]
         +---> [ Head 5: Movement Dynamics (Linear, Arc, Wavy, Circular Trajectories) ]
         +---> [ Head 6: Two-Handed Structure (Symmetry, Dominance, Alternating) ]
         |
         v
+-------------------------------------------------------------------------------------------------+
| STAGE 3: CONTEXT-FREE GRAMMAR (CFG) & PHONETIC SYNTHESIS                                        |
|  - Strict Sign Phonology Validation (99.4% Syntactic Compliance)                                |
|  - Hamburg Notation System (HamNoSys v4) Token Assembly                                         |
+-------------------------------------------------------------------------------------------------+
         |
         v
+-------------------------------------------------------------------------------------------------+
| STAGE 4: SiGML XML COMPILATION & 3D AVATAR ANIMATION                                            |
|  - Signing Gesture Markup Language (SiGML) XML Document Generation                              |
|  - JASigning / CWASA 3D WebGL Avatar Engine (Real-time Hardware-Accelerated Animation)           |
+-------------------------------------------------------------------------------------------------+
```

---

## 📁 Repository Structure & Organization

```
Avatar-Generator-FYP/
├── webapp/                                 # Full-Stack Web Application
│   ├── app.py                              # Flask REST API Server & Video Processing Endpoint
│   ├── static/                             # Frontend Static Assets (CSS, JS, CWASA Avatars)
│   │   ├── css/styles.css                  # Modern UI Stylesheet & Responsive Layouts
│   │   ├── js/app.js                       # Real-Time UI Controller & Diagnostic Matrix
│   │   └── cwasa/                          # JASigning 3D WebGL Avatar Engine
│   └── templates/                          # Application HTML Views
│       └── index.html                      # Main Interactive Dashboard
├── hamer/                                  # HaMeR 3D Hand Mesh Estimation Module
│   ├── models/                             # HaMeR Neural Architecture & MANO Mesh Decoders
│   └── utils/                              # 3D Vector Math & Geometry Utilities
├── ISL_videos/                             # Indian Sign Language Video Dataset & Manifest
├── downloaded_dgs_dataset/                 # German Sign Language (DGS) Dataset & Ground Truth
├── SignFiles/                              # Verified SiGML XML Reference Signs
├── dgs_sigml_outputs/                      # Benchmark Generated SiGML Output Files
│
├── local_hamer_pipeline.py                 # Core End-to-End Video-to-3D-Mesh Translation Pipeline
├── integration_pipeline.py                 # CFG Grammar Compiler & Phonetic Assembler
├── translate_video_to_sigml.py             # CLI Tool for Video-to-SiGML Conversion
├── download_hamer_assets.py                # Pretrained Model Checkpoint & Asset Downloader
│
├── train_v7_hybrid_kaggle.py               # V7 Hybrid Neural Network Training Script
├── train_v6_hybrid_kaggle.py               # V6 Neural Network Baseline Training Script
│
├── benchmark_v8_evaluation.py              # Automated 30-Video Precision & Benchmark Suite
├── benchmark_v7_evaluation.py              # V7 Model Evaluation & Ablation Script
├── compare_sigml.py                        # SiGML Similarity & Tree-Distance Evaluator
├── accuracy.ipynb                          # Interactive Evaluation & Error Analysis Notebook
│
├── README.md                               # Primary Project Documentation
├── INSTALLATION.md                         # Detailed Setup & Installation Guide
├── PROJECT_CONTRIBUTIONS.md                # 4-Person Contribution & Responsibility Matrix
├── PROJECT_ORGANIZATION.md                 # Detailed Codebase Architecture & Directory Map
├── CURRENT_PIPELINE_EXPLANATION.md         # Comprehensive Theoretical & Algorithmic Guide
├── V8_BENCHMARK_AND_ENHANCEMENT_REPORT.md  # Final Benchmark, Accuracy & F1 Evaluation Report
│
├── requirements.txt                        # Production Python Package Dependencies
└── Dockerfile                              # Production Docker Container Specification
```

---

## ⚡ Quickstart & Installation

For full installation instructions, dependency resolution, and troubleshooting, refer to **[INSTALLATION.md](INSTALLATION.md)**.

### 1. Prerequisites
- **OS**: Windows 10/11, Ubuntu 20.04+, or macOS
- **Python**: 3.10 or 3.11
- **Hardware**: 8 GB RAM minimum (16 GB + NVIDIA GPU recommended for HaMeR 3D mesh acceleration)

### 2. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/SaiSriSatvic2005/Avatar-Generator-FYP-.git
cd Avatar-Generator-FYP-

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download Pretrained Model Weights & Assets
```bash
python download_hamer_assets.py
```

### 5. Launch the Web Application
```bash
python webapp/app.py
```
Open your browser and navigate to **`http://localhost:5000`** to access the interactive translation dashboard and 3D avatar viewport.

### 6. Run CLI Video Translation
```bash
python translate_video_to_sigml.py --input ISL_videos/sample_sign.mp4 --output sample_output.sigml
```

### 7. Run Benchmark & Accuracy Evaluation Suite
```bash
python benchmark_v8_evaluation.py
```

---

## 📊 Benchmark & Evaluation Results

Our system has undergone rigorous quantitative evaluation across 30 multi-dialect sign videos against manual phonetic ground truth:

| Pipeline Iteration | Hand Mesh Accuracy | Token Precision | Overall Sequence F1 | CFG Syntactic Validity | 3D Avatar Cosine Similarity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **V1 (Baseline Heuristic)** | 54.2% | 58.1% | 0.56 | 74.2% | 0.6210 |
| **V6 (MediaPipe + BiGRU)** | 71.8% | 76.4% | 0.74 | 91.5% | 0.7840 |
| **V7 (Self-Attention Hybrid)** | 79.4% | 81.9% | 0.80 | 97.2% | 0.8415 |
| **V8 (HaMeR 3D MANO + Hybrid)** | **88.6%** | **85.5%** | **0.84** | **99.4%** | **0.8812** |

### Key Performance Highlights:
- **99.4% CFG Validity**: Context-Free Grammar compiler eliminates ungrammatical sign combinations.
- **88.12% Avatar Cosine Similarity**: High kinematic fidelity between original human performer and 3D avatar joints.
- **Occlusion Robustness**: HaMeR 3D hand mesh reconstruction solves severe finger self-occlusion where standard 2D tracking fails.

---

## 📖 Technical Documentation Index

Detailed engineering specifications, research methodologies, and study guides are documented in the following files:
* **[INSTALLATION.md](INSTALLATION.md)**: Exhaustive setup guide for Windows, Linux, CUDA, and Docker.
* **[PROJECT_CONTRIBUTIONS.md](PROJECT_CONTRIBUTIONS.md)**: Official 4-person academic role and subsystem ownership matrix.
* **[PROJECT_ORGANIZATION.md](PROJECT_ORGANIZATION.md)**: Comprehensive architectural module documentation.
* **[CURRENT_PIPELINE_EXPLANATION.md](CURRENT_PIPELINE_EXPLANATION.md)**: In-depth mathematical formulation of feature extraction, neural classifiers, and HamNoSys mapping.
* **[V8_BENCHMARK_AND_ENHANCEMENT_REPORT.md](V8_BENCHMARK_AND_ENHANCEMENT_REPORT.md)**: Full quantitative benchmark tables, ablation experiments, and latency profiling.
* **[HamNoSys_Study_Guide.md](HamNoSys_Study_Guide.md)**: Reference guide to Hamburg Notation System phonetics and SiGML XML structures.

---

## 📜 Version History & Development Milestones

The project was developed sequentially from **June 2026 to October 2026** across **156 version-controlled commits**:
* **Phase 1 (June – July 2026)**: Baseline architecture, dataset collection, MediaPipe landmark ingestion, and CFG compiler initialization.
* **Phase 2 (August – September 2026)**: V6 and V7 hybrid neural network implementation, self-attention layer integration, and multi-task loss balancing.
* **Phase 3 (October 2026)**: V8 HaMeR 3D MANO mesh integration, full 30-video accuracy benchmarking, WebGL CWASA avatar optimization, and production hardening.

---

## ⚖️ License & Acknowledgements

This project is developed as an academic **Final Year Project (Group 14)**.
- 3D Avatar Rendering powered by **JASigning / CWASA** (University of East Anglia).
- 3D Hand Mesh Estimation powered by **HaMeR** (University of California, Berkeley).
- Skeletal tracking powered by **Google MediaPipe**.
