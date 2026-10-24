# 🏆 Project Contribution Matrix (Group 14)
## Universal Sign Language to 3D Avatar System (ISL & DGS)

---

## 👥 Executive 4-Person Role Summary

```text
+---------------------------------------------------------------------------------------------------+
|                     UNIVERSAL SIGN LANGUAGE TO 3D AVATAR SYSTEM (GROUP 14)                        |
+---------------------------------------------------------------------------------------------------+
|  MEMBER                           |  CORE SYSTEM RESPONSIBILITY      |  PRIMARY SPECIALIZATION    |
+---------------------------------------------------------------------------------------------------+
|  VIVIN RAKUL (Co-Lead)            |  Deep Learning & Pipeline Integr.|  3D Mesh, Self-Attention   |
|  SAI SRI SATVIC (Co-Lead)         |  Sequence Modeling & SiGML Synt. |  BiGRU, CFG, Flask Engine  |
|  SNIGDHA (Lead)                   |  3D Spatial Geometry & Rendering |  Multi-Task Loss, CWASA    |
|  VIJAY SARAVANA PANDI (Lead)      |  Video Ingestion & Feature Eng.  |  177-D Landmarks, Body/Face|
+---------------------------------------------------------------------------------------------------+
```

---

## 🎯 Individual Contribution Breakdown

### **1. Vivin Rakul (`@vivin888`): Co-Lead (Deep Learning Systems, Model Generation & Pipeline Integration)**
* **Role & Weightage**: Equal Technical Co-Lead with Satvic (44 Commits).
* **Core Responsibilities**:
  1. **3D Hand Mesh Recovery (HaMeR) Integration**:
     - Engineered the dual-stream feature extraction pipeline (`local_hamer_pipeline.py`) incorporating HaMeR's 3D MANO mesh (778 vertices, 21 joint rotations) to eliminate 2D finger self-occlusion.
     - Implemented 3D hand bounding box normalization, camera translation compensation, and depth projection.
  2. **Neural Network Architecture & Model Generation**:
     - Designed the **Multi-Head Self-Attention mechanism** and 1D Convolutional temporal blocks in `train_v7_hybrid_kaggle.py` to weight key phonetic inflection frames.
     - Trained and tuned the Handshape Classification Head (flat, fist, pinch, index, victory, open) and Two-Handed Structure Head.
  3. **End-to-End Pipeline Integration & Validation**:
     - Integrated the complete video-to-avatar processing pipeline combining vision feature extraction, neural inference, and syntax serialization.
     - Conducted multi-version quantitative accuracy evaluation (`benchmark_v8_evaluation.py`) demonstrating a progression from 54.2% (V1) to 88.6% (V8).

---

### **2. Sai Sri Satvic (`@SaiSriSatvic2005`): Co-Lead (Sequence Modeling, Cloud Architecture & SiGML Synthesis)**
* **Role & Weightage**: Equal Technical Co-Lead with Vivin (44 Commits).
* **Core Responsibilities**:
  1. **Recurrent Sequence Modeling**:
     - Architected the **Bidirectional GRU (BiGRU)** temporal sequence layers in `train_v7_hybrid_kaggle.py` to model forward and backward temporal phonetic transitions across continuous frames.
     - Trained the dynamic Movement Trajectory Head across directional spatial vectors (up, down, left, right, in, out).
  2. **Context-Free Grammar (CFG) & SiGML Compiler**:
     - Developed the strict **Context-Free Grammar (CFG) compiler** in `integration_pipeline.py`, achieving **99.4% syntactic validity** across universal sign phonology.
     - Built the SiGML XML document generator mapping HamNoSys phonetic tokens into valid gesture markup.
  3. **Full-Stack Backend & Cloud DevOps**:
     - Built the production Flask REST API backend (`webapp/app.py`) with asynchronous video processing and error handling.
     - Configured containerization (`Dockerfile`) and optimized memory allocation for cloud deployment.

---

### **3. Snigdha (`@snigdhasirivalli`): Lead (3D Spatial Geometry, Multi-Task Loss & Avatar Rendering)**
* **Role & Weightage**: Specialized Subsystem Lead (34 Commits).
* **Core Responsibilities**:
  1. **Multi-Task Loss Balancing & Neural Optimization**:
     - Designed and calibrated the weighted multi-task cross-entropy loss function in `train_v7_hybrid_kaggle.py` across all 6 classification heads simultaneously.
     - Authored the subsystem token precision validation suite (`compare_sigml.py`).
  2. **3D Spatial Orientation & Normal Vector Modeling**:
     - Formulated 3D palm normal cross-product math and finger directional vector calculation in spherical coordinates (`ori_model2.py`).
     - Trained the Extended Finger Direction and Palm Orientation classification heads.
  3. **3D Avatar Viewport & Motion Dynamics**:
     - Integrated the JASigning CWASA 3D WebGL avatar viewport, bone joint kinematics, and trajectory smoothing.
     - Achieved **88.12% Avatar Cosine Similarity** against human kinematic motion.

---

### **4. Vijay Saravana Pandi (`@VijaySaravanaPandi`): Lead (Video Ingestion, Feature Engineering & Spatial Datasets)**
* **Role & Weightage**: Specialized Subsystem Lead (34 Commits).
* **Core Responsibilities**:
  1. **177-D Feature Engineering Pipeline**:
     - Engineered wrist-centered, scale-invariant 177-dimensional skeletal landmark tensors combining hand, pose, and facial markers.
     - Implemented multi-threaded frame extraction, cropping, and resolution standardization.
  2. **Anatomical Body & Face Location Subsystems**:
     - Implemented Da Vinci facial third ratios in `Head_and_face_location.py` to establish precise facial anchor points.
     - Developed upper body anchor detection and contact point classification for sign localization.
  3. **Dataset Ingestion & Manifest Management**:
     - Curated and validated the Indian Sign Language (ISL) and German Sign Language (DGS) corpora (`isl_dataset_manifest.json`).
     - Aligned ground-truth phonetic annotations for quantitative benchmark evaluation.

---

## 📊 Comprehensive 4-Person Contribution Breakdown Table

| Member | Primary Subsystem Ownership | Deep Learning / AI Role | Key Code Files & Deliverables | Quantitative Metric |
| :--- | :--- | :--- | :--- | :--- |
| **Vivin Rakul** | **3D Hand Mesh & Pipeline Integration** | Self-Attention, Conv1D & Handshape Head | `local_hamer_pipeline.py`, `train_v7_hybrid_kaggle.py`, `benchmark_v8_evaluation.py` | **88.6% Mesh Acc**, **85.5% Token Prec.** |
| **Sai Sri Satvic** | **Sequence Model & SiGML Synthesis** | BiGRU Temporal Model & CFG Compiler | `integration_pipeline.py`, `webapp/app.py`, `translate_video_to_sigml.py` | **99.4% CFG Valid.**, **88.1% Avatar Sim.** |
| **Snigdha** | **3D Spatial Geometry & Avatar Viewport** | Multi-Task Loss Balancing & Orientation | `compare_sigml.py`, `ori_model2.py`, `webapp/static/cwasa/` | **85.49% Token Prec.**, **75%–82% Ori. Acc.** |
| **Vijay Saravana Pandi** | **Video Ingestion & Feature Engineering** | 177-D Landmarks & Location Head | `Head_and_face_location.py`, `isl_dataset_manifest.json`, `crop_dgs_frontal_videos.py`| **80%–85% Loc. Acc.**, **177-D Tensors** |
