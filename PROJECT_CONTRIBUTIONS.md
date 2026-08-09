# 👥 Project Contribution Matrix (Group 14)

```text
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                        UNIVERSAL SIGN LANGUAGE TO 3D AVATAR SYSTEM (GROUP 14)                          │
├────────────────────────────────────┬───────────────────────────────────┬───────────────────────────────┤
│  MEMBER                            │  APPLICATION PLATFORM             │  NEURAL NETWORK & AI ROLE     │
├────────────────────────────────────┼───────────────────────────────────┼───────────────────────────────┤
│  VIJAY (Mobile Lead - Client UI)   │  Mobile App UI/UX & Video Camera  │  Feature Engineering & Loc NN │
│  SNIGDHA (Mobile Lead - Avatar)    │  Mobile Avatar Viewport & API     │  Multi-Task Loss & Orientation│
│  VIVIN (Web Lead - Frontend)       │  Web App UI & Diagnostic Matrix   │  Self-Attention & Handshape NN│
│  SATVIC (Web Lead - Backend/Ops)   │  Flask API Backend & Cloud DevOps │  BiGRU Sequence Model & Mov NN│
└────────────────────────────────────┴───────────────────────────────────┴───────────────────────────────┘
```

---

## 📱 Mobile App Team (Vijay & Snigdha)

### **Vijay: Mobile App Lead (UI/UX & Camera Ingestion) + NN Feature Engineering & Location Subsystems**
* **1. Application Platform (Mobile App)**:
  - Designed the mobile user interface (responsive layouts, dark/light theme, upload buttons, gesture guide).
  - Built the mobile camera recording module (video frame capture, live feed preview, file compression, and resolution standardization).
* **2. Major Neural Network (NN) Contribution**:
  - **177-D Feature Engineering Pipeline** (`build_wlasl_landmarks.py`): Engineered wrist-centered, scale-invariant 177-dimensional landmark tensors (`dataset_landmarks.npz`).
  - **Body Location & Spatial Classification Heads**: Trained and tuned the Location Head in `HamNoSysSequenceNet V2` across Head, Chest, Shoulders, and Neutral Space classes.
* **3. Subsystem Modules & Remaining Files**:
  - **Anatomical Body & Face Tracking Subsystems**: Implemented Da Vinci Facial Third Ratios in `Head_and_face_location.py`, upper body anchor detection in `upper_body_locations_video.py`, and contact classification in `contact_types_e.py`.
* **4. Target Subsystem Accuracy Metrics**:
  - **Body Location Head (Loc)**: **80.0% – 85.0% Accuracy** (High - Head, Chest).

---

### **Snigdha: Mobile App Lead (Networking & 3D Avatar Rendering) + NN Multi-Task Loss, Precision & Orientation Heads**
* **1. Application Platform (Mobile App)**:
  - Built the mobile networking layer (asynchronous HTTP multipart upload to translation server, JSON response parsing, and error recovery).
  - Integrated the mobile 3D Avatar viewport (mobile WebGL / WebView embedding of JASigning avatars with touch gestures: orbit, zoom, reset, replay).
* **2. Major Neural Network (NN) Contribution**:
  - **Multi-Task Loss Optimization & Precision Benchmark**: Designed the weighted multi-head cross-entropy loss function across all 6 classification heads and authored the token precision validation suite (`tests/test_token_precision.py`).
  - **Extended Finger & Palm Orientation Heads**: Trained and tuned the Extended Finger Direction and Palm Orientation classification heads in the neural network.
* **3. Subsystem Modules & Remaining Files**:
  - **Trajectory & Motion Tracking**: Primary directional motion vectors in `movement1_prava.py`, secondary circular/state-transition dynamics in `Movement_2.py`, and trajectory smoothing in `video_preprocessing.py`.
* **4. Target Subsystem Accuracy Metrics**:
  - **Hand Orientation Head (Palm/Ext)**: **75.0% – 82.0% Accuracy** (Strong on Cardinal Vectors).
  - **Overall Token Precision**: **85.49% (78.0% – 85.7%)**.
  - **Overall Sequence F1 Score**: **0.82 – 0.85**.

---

## 💻 Web App Team (Vivin & Satvic)

### **Vivin: Web App Lead (Frontend UI/UX & Diagnostics) + NN Self-Attention & Handshape/Two-Handed Modeling**
* **1. Application Platform (Web App)**:
  - Developed the modern web interface (`webapp/templates/index.html`, `webapp/static/css/styles.css`) with synchronized dual-pane video playback, gloss description cards, and interactive speed controls.
  - Built the **10-Module Diagnostic Matrix UI** and dynamic phonetic chip breakdown in `webapp/static/js/app.js`.
* **2. Major Neural Network (NN) Contribution**:
  - **Self-Attention Mechanism & Conv1D Blocks** (`train_landmark_nn.py`): Integrated 1D convolutional feature extractors and self-attention weights to highlight key phonetic inflection frames.
  - **Handshape & Two-Handed Structure Heads**: Trained the Handshape Head (flat, fist, pinch, index, victory) and Two-Handed Symmetry Head.
* **3. Subsystem Modules & Remaining Files**:
  - **Handshape & Vision Subsystems**: Joint-angle finger curl geometry in `Handshape_Model.py`, dual-hand activation in `dual_hand_classifier.py`, and unified single-pass landmark tracking in `shared_landmarks.py`.
* **4. Target Subsystem Accuracy Metrics**:
  - **Handshape Head (Base Shapes)**: **72.0% – 78.0% Accuracy**.
  - **Two-Handed Structure Head**: **70.0% – 75.0% Accuracy**.

---

### **Satvic: Web App Lead (Backend REST API & Cloud DevOps) + NN Sequence Modeling, CFG Compiler & SiGML Synthesis**
* **1. Application Platform (Web App & Cloud Backend)**:
  - Developed the Flask REST API backend (`webapp/app.py`), multipart video processing routes, global JSON error handlers, and file lifecycle management.
  - Integrated the **JASigning CWASA 3D WebGL Avatar Engine** (Anna, Luna, Siggi, Marc, Francoise).
  - Configured Cloud Deployment & DevOps: Containerization (`Dockerfile`), Gunicorn WSGI server, 512MB RAM optimization, and Render live deployment.
* **2. Major Neural Network (NN) Contribution**:
  - **Bidirectional GRU (BiGRU) Sequence Architecture** (`train_landmark_nn.py`): Designed the recurrent temporal modeling layers to capture long-range forward and backward temporal sign dependencies.
  - **Movement Trajectory Classification Head**: Trained and tuned the dynamic movement trajectory head across all directional vectors (up, down, left, right, in, out).
* **3. Subsystem Modules & Remaining Files**:
  - **CFG Grammar Compiler & 3D Vector Math**: Context-Free Grammar compiler in `integration_pipeline.py`, 3D cross-product palm normal calculation in `ori_model2.py`, SiGML XML conversion pipeline, and CFG testing (`tests/test_cfg_grammar.py`).
* **4. Target Subsystem Accuracy Metrics**:
  - **Movement Trajectory Head**: **65.0% – 72.0% Accuracy**.
  - **CFG Grammar Validity**: **99.4%** (169/170 entries valid).
  - **3D Avatar Cosine Similarity**: **88.12% (0.8812)**.

---

## 📊 Comprehensive 4-Person Contribution Breakdown Table

| Member | Platform Focus | Neural Network (NN) Ownership | Core Subsystem Modules & Files | Key Target Accuracy Metric |
| :--- | :--- | :--- | :--- | :--- |
| **Vijay** | **Mobile App (UI & Camera)** | 177-D Feature Engineering & Location Head | Face/Head Ratios (`Head_and_face_location.py`), Body Anchors, Contact Detection | **80%–85% Location Acc** |
| **Snigdha** | **Mobile App (Networking & 3D Avatar)** | Multi-Task Loss Balancing, Precision Suite, Orientation Head | Primary Motion (`movement1_prava.py`), Secondary Dynamics (`Movement_2.py`), Smoothing | **75%–82% Orientation Acc**, **85.49% Token Precision** |
| **Vivin** | **Web App (Frontend & Diagnostics)** | Self-Attention Mechanism, Conv1D, Handshape & Two-Handed Heads | Finger Curl Geometry (`Handshape_Model.py`), Dual-Hand Tracking (`shared_landmarks.py`) | **72%–78% Handshape Acc**, **70%–75% Two-Handed** |
| **Satvic** | **Web App (REST API & Cloud DevOps)** | BiGRU Temporal Sequence Model & Movement Head | CFG Compiler (`integration_pipeline.py`), 3D Vector Math (`ori_model2.py`), SiGML Engine | **65%–72% Movement Acc**, **99.4% CFG Validity**, **88.12% Avatar Similarity** |
