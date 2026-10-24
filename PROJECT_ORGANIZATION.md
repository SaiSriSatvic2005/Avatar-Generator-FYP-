# Detailed Project Organization & Codebase Guide
## Universal Sign Language to 3D Avatar System (ISL & DGS)

---

## 🏛️ High-Level Directory Overview

```text
Avatar-Generator-FYP/
├── webapp/                                 # Full-Stack Web Application Platform
├── hamer/                                  # 3D Hand Mesh Estimation Subsystem
├── ISL_videos/                             # Indian Sign Language Dataset
├── downloaded_dgs_dataset/                 # German Sign Language Dataset
├── SignFiles/                              # Reference SiGML XML Corpus
├── dgs_sigml_outputs/                      # Generated SiGML Outputs
│
├── local_hamer_pipeline.py                 # Core Video-to-3D-Mesh Translation Engine
├── integration_pipeline.py                 # Context-Free Grammar (CFG) Compiler
├── translate_video_to_sigml.py             # CLI Translation Entrypoint
├── download_hamer_assets.py                # Asset & Model Checkpoint Manager
│
├── train_v7_hybrid_kaggle.py               # V7 Hybrid Neural Network Training
├── train_v6_hybrid_kaggle.py               # V6 Baseline Model Training
│
├── benchmark_v8_evaluation.py              # Quantitative Precision & Benchmark Suite
├── compare_sigml.py                        # SiGML Tree Distance & Token Precision Evaluator
├── accuracy.ipynb                          # Error Analysis & Jupyter Diagnostic Notebook
│
├── README.md                               # Primary System Overview
├── INSTALLATION.md                         # Detailed Setup & User Manual
├── PROJECT_CONTRIBUTIONS.md                # 4-Person Contribution Matrix
├── CURRENT_PIPELINE_EXPLANATION.md         # In-Depth Theoretical & Algorithmic Formulation
└── V8_BENCHMARK_AND_ENHANCEMENT_REPORT.md  # Comprehensive Benchmark & Accuracy Report
```

---

## 🔍 Module-by-Module Breakdown

### 1. Web Application & Real-Time Visualization (`webapp/`)
- `webapp/app.py`: Production Flask application exposing REST endpoints (`/translate`, `/status`, `/avatar`). Handles file upload validation, thread-safe asynchronous video decoding, JSON error formatting, and temporary file lifecycle cleanup.
- `webapp/static/js/app.js`: Client-side state manager and dynamic diagnostic matrix renderer. Displays real-time confidence scores for Handshape, Orientation, Location, and Trajectory heads.
- `webapp/static/css/styles.css`: Responsive, glassmorphic UI stylesheet featuring dark/light modes, synchronized video playback viewports, and mobile-responsive layouts.
- `webapp/static/cwasa/`: Bundled JASigning CWASA WebGL avatar engine containing JavaScript libraries, character definitions (Anna, Luna, Siggi, Marc, Francoise), and hardware-accelerated bone rigs.

---

### 2. Core Translation & Synthesis Engines
- `local_hamer_pipeline.py`: The heart of the V8 system. Implements dual-stream feature extraction:
  1. Synchronous frame extraction and bounding box detection.
  2. HaMeR forward pass generating 3D MANO hand surface meshes (778 vertices).
  3. MediaPipe Holistic skeletal feature extraction.
  4. Hybrid neural classifier inference across 6 discrete sign heads.
- `integration_pipeline.py`: Context-Free Grammar (CFG) phonetic compiler. Takes predicted discrete classes (Base Shape, Extended Direction, Palm Orientation, Location Anchor, Movement Vector, Two-Handed Structure) and validates syntactic validity according to universal sign phonology before constructing the HamNoSys and SiGML XML payload.
- `translate_video_to_sigml.py`: Standalone command-line utility providing batch processing, pipeline timing statistics, and output serialization.

---

### 3. Deep Learning Architectures & Training Pipelines
- `train_v7_hybrid_kaggle.py`: PyTorch training script for the hybrid neural architecture:
  - Input: 177-dimensional landmark tensors + 3D joint rotation angles.
  - Encoder: 1D Temporal Convolutions + Multi-Head Self-Attention layers.
  - Temporal Model: Bidirectional GRU (BiGRU) tracking temporal transitions across 60 frames.
  - Multi-Task Loss: Dynamic weighted Cross-Entropy loss balancing all 6 classification heads simultaneously.
- `train_v6_hybrid_kaggle.py`: Baseline neural sequence model utilized for ablation studies and comparative benchmarking.

---

### 4. Datasets & Reference Corpora
- `ISL_videos/`: Curated dataset of Indian Sign Language video clips representing diverse signers, angles, and lighting conditions.
- `downloaded_dgs_dataset/`: German Sign Language (DGS) vocabulary corpus with ground-truth phonetic annotations.
- `isl_dataset_manifest.json`: JSON catalog tracking video filenames, gloss names, frame counts, and validated ground-truth HamNoSys tokens.
- `SignFiles/`: Standard validated SiGML XML definitions used for unit testing and accuracy validation.

---

### 5. Verification, Benchmarking & Diagnostics
- `benchmark_v8_evaluation.py`: Automated benchmarking harness running 30 validation videos through the complete pipeline, calculating token precision, sequence F1, CFG adherence, and inference latency.
- `compare_sigml.py`: XML tree-distance and string-metric comparator evaluating generated SiGML against ground-truth XML files.
- `accuracy.ipynb`: Interactive Jupyter Notebook containing confusion matrices, loss curves, precision-recall graphs, and qualitative error analysis.
