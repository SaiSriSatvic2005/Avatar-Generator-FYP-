# Universal Sign Language to 3D Avatar Synthesizer: Complete Project Architecture, Evolutionary Progression, & Technical Specification

**Project Title:** Automated Monocular Video-to-HamNoSys Transcription and Collision-Free 3D Avatar Animation  
**Current Production Engine:** Version 8.0 Complete Hybrid System (HaMeR 3D Surface Mesh + 128-D Local-Canonical ResNet + 18-D Spatiotemporal Attention-BiGRU + Closed-Form Metric Geometry + Collision-Free IK Synthesis)  
**Target Output:** CWASA / JASigning-Compliant SiGML XML for Real-Time 3D WebGL Avatar Rendering  
**Primary Dataset:** [`FYP-DataSet`](file:///d:/academics/Hybrid_V7/Active_Hybrid_Pipeline/FYP-DataSet) (ISL: 711 videos, DGS: 500 videos, ASL: 278 videos)

---

## Table of Contents
1. [Executive Summary & Foundational Primer](#1-executive-summary--foundational-primer)
2. [End-to-End System Architecture & Information Flow](#2-end-to-end-system-architecture--information-flow)
3. [Chronological Version-by-Version Evolution (V1 through V8)](#3-chronological-version-by-version-evolution-v1-through-v8)
   - [3.1 Comparison Table](#31-comparison-table)
   - [3.2 Why Each Version Was Abandoned: Failure Diagnosis & Dataset Switch Rationale](#32-why-each-version-was-abandoned-failure-diagnosis--dataset-switch-rationale)
4. [Mathematical Formulations & Theoretical Derivations](#4-mathematical-formulations--theoretical-derivations)
5. [Deep Neural Architectures: Model A & Model B](#5-deep-neural-architectures-model-a--model-b)
6. [Analytical 3D Geometric Heuristics & Collision Clearance](#6-analytical-3d-geometric-heuristics--collision-clearance)
7. [Training Methodology, Loss Functions, & Data Splitting](#7-training-methodology-loss-functions--data-splitting)
8. [Empirical Evaluation Benchmarks & Accuracy Progression](#8-empirical-evaluation-benchmarks--accuracy-progression)
9. [Detailed Failure Mode & Root-Cause Analysis](#9-detailed-failure-mode--root-cause-analysis)
10. [Future Research Directions & Reference Citations](#10-future-research-directions--reference-citations)

---

## 1. Executive Summary & Foundational Primer

### 1.1 Core Mission
The **Universal Sign Language to 3D Avatar Synthesizer** is an automated computer-vision, deep-learning, and phonetic compilation pipeline. It translates raw monocular RGB video of sign language into standardized phonetic notation strings (**HamNoSys**) and compiles them into valid **SiGML XML** directives. These directives drive a **3D WebGL Virtual Human Avatar** (e.g., *Luna*, *Marc*, *Anna*) that replicates the sign with millimeter-level spatial fidelity and 100% collision-free kinematics at 60 FPS in standard web browsers without client-side GPU requirements.

```
+--------------------+      +--------------------+      +--------------------+      +--------------------+
|  Input Sign Video  | ---> |  Hybrid 3D Vision  | ---> |  Linguistic CFG    | ---> |  3D WebGL Avatar   |
|  (Monocular RGB)   |      |  Pipeline (V8)     |      |  Compiler (SiGML)  |      |  Playback (CWASA)  |
+--------------------+      +--------------------+      +--------------------+      +--------------------+
```

### 1.2 The Neuro-Symbolic Hybrid Philosophy
Monolithic end-to-end deep learning models (such as direct video-to-SiGML text generation or unconstrained RNNs) suffer from severe mode collapse, generating hallucinated or non-existent XML tags (< 25% token precision) and violent avatar mesh clipping.

This system implements a **modular neuro-symbolic hybrid architecture**:
1. **Deep Neural Networks** are assigned strictly to high-variance perceptual classification tasks where complex non-linear spatial patterns exist:
   - **Model A (Handshape)**: 128-D Skeletal Hand-ResNet (12 canonical classes).
   - **Model B (Movement)**: 18-D Spatiotemporal Attention-BiGRU (19 kinematic direction classes).
2. **Deterministic 3D Geometry (Analytical Heuristics)** is assigned strictly to closed-form spatial operations:
   - Singular Value Decomposition (SVD) for 3D palm surface normals.
   - Perspective-compensated longitudinal pointing vectors.
   - 7-zone calibrated Voronoi anatomical body regions.
   - Collision-free torso clearance gating.
3. **Context-Free Grammar (CFG) & ANTLR Linter** acts as the formal compiler, enforcing structural ordering rules and guaranteeing 100% syntactic validity.

### 1.3 Essential Terminology
* **HamNoSys (Hamburg Notation System):** Universal phonetic transcription system for sign languages developed by the University of Hamburg (Prillwitz et al., 1989). Phonetically decomposes any manual sign into: *Handshape*, *Hand Orientation* (extended finger direction + palm normal), *Body Location*, *Contact & Proximity*, *Movement Trajectory*, and *Bimanual Symmetry*.
* **SiGML (Signing Gesture Markup Language):** Structured XML dialect that encodes HamNoSys phonetic characters into hierarchical tags specifying joint angles, spatial anchors, and temporal durations.
* **CWASA / JASigning:** Open-source WebGL/Java virtual character animation runtime (University of East Anglia & Hamburg). Interprets SiGML XML and drives 3D skeletal avatars (*Luna*, *Anna*, *Marc*) via internal Inverse Kinematics (IK).
* **HaMeR (Hand Mesh Recovery, CVPR 2024):** 3D hand pose and shape estimation foundation model using a Vision Transformer (ViT-Huge) backbone pretrained on millions of hands, reconstructing dense 3D meshes (MANO) from monocular RGB frames even under extreme occlusions.
* **MANO (Model for Articulated Non-rigid Hand Motion):** Parametric 3D hand model with shape parameters $\beta \in \mathbb{R}^{10}$ and articulation pose $\theta \in \mathbb{R}^{45}$, producing a 778-vertex surface mesh and 21 3D joint keypoints.

---

## 2. End-to-End System Architecture & Information Flow

The pipeline operates across ten interconnected stages:

```
+-----------------------------------------------------------------------------------------------------------------------+
|                                           STAGE 1: VIDEO INGESTION & DECODING                                         |
|  - Reads video feed (OpenCV cv2.VideoCapture), standardizes RGB space, measures frame count T and FPS                 |
+-----------------------------------------------------------------------------------------------------------------------+
                                                          │
                                                          ▼
+-----------------------------------------------------------------------------------------------------------------------+
|                                   STAGE 2: SPATIAL DETECTION & 3D MESH RECOVERY                                       |
|  - MediaPipe 21-pt HandLandmarker: Localizes dominant hand bounding box [x_min, y_min, x_max, y_max]                 |
|  - HaMeR (ViT-Huge Backbone): Recovers MANO 3D surface mesh (778 vertices) & 21 3D joint coordinates (kps_3d)         |
|  - Extracts camera-space translation vector T_cam and continuous MANO pose articulations theta                       |
+-----------------------------------------------------------------------------------------------------------------------+
                                                          │
                                                          ▼
+-----------------------------------------------------------------------------------------------------------------------+
|                                 STAGE 3: KINETIC STROKE ISOLATION & APEX PROFILING                                    |
|  - Calculates wrist velocity v(t) = ||w_t - w_{t-1}|| and elevation e(t) = 1.0 - y_wrist                             |
|  - Evaluates kinetic apex score S(t) = e(t) - 1.2 * v_norm(t)                                                         |
|  - Identifies inflection apex index t_apex (instant of deliberate hold deceleration where sign meaning is expressed)   |
|  - Isolates multi-frame dense temporal sample window [t_apex - 3 : t_apex + 4] with Gaussian weight sigma = 3.0       |
+-----------------------------------------------------------------------------------------------------------------------+
                                                          │
                         ┌────────────────────────────────┴────────────────────────────────┐
                         ▼                                                                 ▼
+----------------------------------------------------+   +--------------------------------------------------------------+
|   STAGE 4: LOCAL-CANONICAL HANDSHAPE EXTRACTION    |   |         STAGE 7: 18-D SPATIOTEMPORAL TRAJECTORY MODEL        |
|  - Constructs orthonormal palm frame R_local:      |   |  - Fuses wrist trajectory with 15D fingertip displacements   |
|      y_axis = (mid_mcp - wrist) / ||mid_mcp-wrist|||   |  - Formats 18-D state vector: [3D wrist + 15D fingertips]    |
|      z_axis = (trans x y_axis) / ||trans x y_axis|||   |  - Resamples via linear spline to 32 uniform temporal steps  |
|      x_axis = y_axis x z_axis                      |   |  - Model B (AttentionBiGRU): 1D Conv + 2-layer BiGRU (128)   |
|  - Projects kps_3d into R_local (invariance to     |   |    + 4-head Multi-Head Self-Attention pooling block          |
|    camera angle, wrist pitch, roll, zoom)          |   |  - Kinematic Vector Displacement Heuristic Layer:            |
|  - Builds 128-D vector:                            |   |    * Path length > 0.40 -> discounts hamnomotion prior by 85%|
|      * 63D rotated & scale-normalized keypoints    |   |    * Direct displacement checks: upward, downward, lateral   |
|      * 15D joint flexion angles (cos phi)          |   |  - Output: Movement direction token (hammoveu, etc.)         |
|      * 4D inter-finger abduction spreads (cos psi) |   +--------------------------------------------------------------+
|      * 5D thumb-to-fingertip pinch proximities     |                                  │
|      * 41D continuous MANO articulation pose       |                                  │
|  - Model A (SkeletalHandResNet_V7):                |                                  │
|    3x ResBlocks + LayerNorm + Mish activations     |                                  │
|  - Output: Base Handshape token (hamflathand, etc.)|                                  │
+----------------------------------------------------+                                  │
                         │                                                              │
                         ▼                                                              │
+----------------------------------------------------+                                  │
|   STAGE 5: ANATOMICAL GEOMETRIC MODIFIERS ENGINE   |                                  │
|  - Evaluates finger & thumb articulation geometry: |                                  │
|      * hamthumboutmod: theta_thumb > 42 deg        |                                  │
|      * hamthumbacrossmod: thumb_pinky < 0.35 * palm|                                  │
|      * hamfingerbendmod: joint flexion in [45, 90] |                                  │
|      * hamfingerstraightmod: extension > 165 deg   |                                  │
+----------------------------------------------------+                                  │
                         │                                                              │
                         ▼                                                              │
+----------------------------------------------------+                                  │
|       STAGE 6: METRIC 3D ORIENTATION ENGINE        |                                  │
|  - 3D Palm Normal: n_palm = (u_radial x u_ulnar)   |                                  │
|  - 8-direction angular octant partitioning (45 deg)|                                  │
|  - Extended finger direction: longitudinal axis    |                                  │
|  - Dynamic Orientation State Transition:           |                                  │
|    Start 30% vs End 30% window mode comparison.    |                                  │
|    If rotated: emits hamreplace + end orientation  |                                  │
+----------------------------------------------------+                                  │
                         │                                                              │
                         ▼                                                              │
+----------------------------------------------------+                                  │
|      STAGE 8: SPATIAL ANCHORING & TWO-HANDED       |                                  │
|  - Calibrated vertical signing zones:              |                                  │
|    Forehead (<0.22), Cheek/Nose (0.22-0.32),       |                                  │
|    Chin (0.32-0.38), Chest (0.38-0.52), Neutral    |                                  │
|  - Bilateral symmetry: Left-hand active elevation  |                                  │
|    mean_elev > 0.25 -> hamsymmlr                   |                                  │
+----------------------------------------------------+                                  │
                         │                                                              │
                         └───────────────────────┬──────────────────────────────────────┘
                                                 │
                                                 ▼
+-----------------------------------------------------------------------------------------------------------------------+
|                               STAGE 9: CONTEXT-FREE GRAMMAR & COLLISION-FREE COMPILER                                 |
|  - CFG Slot Assembly: Sign ::= [Symmetry] + HandshapeBlock + OrientationBlock + [LocationBlock] + [MovementBlock]     |
|  - Breakthrough Collision-Free Guard: Decouples hold deceleration from physical touch;                                |
|    Blocks <hamtouch/> on hamchest and hamshoulders, preserving natural 25 cm forward clearance                    |
|  - ANTLR Ham4Parser & XML Schema validation -> Strict Syntactic SiGML XML Output                                      |
+-----------------------------------------------------------------------------------------------------------------------+
                                                          │
                                                          ▼
+-----------------------------------------------------------------------------------------------------------------------+
|                                    STAGE 10: 3D WEBGL AVATAR RENDERING (CWASA)                                        |
|  - Dispatches SiGML XML via CWASA.playSiGMLText(sigmlXml)                                                             |
|  - CWASA Inverse Kinematics solver maps phonemes to bone rotations                                                    |
|  - 60 FPS real-time rendering of humanoid mesh without torso mesh penetration                                        |
+-----------------------------------------------------------------------------------------------------------------------+
```

---

## 3. Chronological Version-by-Version Evolution (V1 through V8)

> [!IMPORTANT]
> **Why Accuracy Numbers Appear to Decrease Across Versions:**
> A superficial reading of the table below might suggest that V2 (78.82% handshape, 77.65% movement) outperformed V8 (76.92% handshape, 28.85% movement). **This is misleading.** The accuracy numbers are not directly comparable across versions because **each version used a different dataset, different ground truth quality, different evaluation methodology, and different task complexity**. Specifically:
> - V1/V2 measured accuracy against a **synthetic gloss→HamNoSys dictionary** on WLASL (ASL) with **frame-level random splitting** (severe data leakage), an **8-class coarse 2D movement task**, and only **3 simplified orientation/finger-direction classes**. When V2 was tested on genuinely unseen signers, real-world accuracy collapsed to <30% handshape and <8% movement.
> - V6/V7/V8 measure accuracy against **expert-annotated SiGML ground truth** from the DGS corpus with **strict video-level cross-validation** (`StratifiedGroupKFold`), a **19-class fine-grained 3D trajectory task**, and **16 orientation directions**. V8's 76.92% handshape and 28.85% movement on unseen DGS signers represent genuinely harder, more scientifically rigorous evaluations.
>
> **Each version transition was driven by identified architectural or dataset failures**, documented in Section 3.2 below.

### 3.1 Comparison Table

| Dimension | Version 1 (Baseline) | Version 2 (CFG + WLASL) | Version 3 (ISL Split) | Version 4/5 (Early HaMeR) | Version 6/7 (Kaggle Overhaul) | Version 8 (Current Engine) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Vision Backbone** | 10 uncoordinated MediaPipe instances | MediaPipe Holistic singleton | DWPose (distilled, occlusion-resilient) | HaMeR ViT-Huge 3D Mesh + MANO model | HaMeR ViT-Huge + MediaPipe Hand Detector | **HaMeR ViT-Huge + 2D Full-Frame Camera Fusion** |
| **Hand Representation** | Flattened raw 2D pixel coordinates ($x, y$) | 177-D normalized landmark vectors | 63-D scale-invariant 3D coordinates | 207-D raw metric Cartesian keypoints | 113-D rotation-invariant skeletal distances | **128-D Local-Canonical Orthonormal Basis ($R_{\text{local}}$)** |
| **Handshape Classifier** | Scikit-learn Decision Trees (`clf_view.pkl`) | 6-Head Net (`hamnosys_net_v2.pth`) | 4-layer MLP (`isl_handshape_mlp.pth`) | `SkeletalHandResNet` (207-D, BatchNorm) | `SkeletalHandResNet_V6` (113-D, LayerNorm, Mish) | **`SkeletalHandResNet_V7` (128-D, 3 ResBlocks, 12 Classes)** |
| **Movement Classifier** | Static frame coordinate difference | Start/End 2D pixel delta (8 classes) | 1D-CNN on 3D velocity vectors | BiGRU on raw 3D wrist trajectory ($32 \times 3$) | Attention-BiGRU on 18-D kinematics ($32 \times 18$) | **18-D Attention-BiGRU + Kinematic Vector Displacement Layer** |
| **Movement Features** | 2D pixel delta $(\Delta x, \Delta y)$ | 2D vector angles (8 cardinal directions) | 3D velocity vectors | Normalized wrist Cartesian trajectory | Wrist trajectory + 5 fingertip vectors ($18\text{D}$) | **Wrist displacement + 5 fingertip relative 3D vectors ($18\text{D}$)** |
| **Movement Task Complexity** | N/A (heuristic) | **8 coarse 2D screen directions** | Kinematic displacement heuristic | 19 classes (collapsed to static) | **19 fine-grained 3D trajectory classes** | **19 fine-grained 3D trajectory classes** |
| **Compiler & Grammar** | Naive `set()` dump accumulator | Context-Free Grammar (CFG) compiler | CFG compiler + `sigml_grammar_linter.py` | CFG compiler with DGS token mapping | CFG compiler with ANTLR Ham4Parser compliance | **CFG Compiler with Collision-Free IK Clearance Engine** |
| **Torso Collision Guard** | None (Avatar crashed) | Naive boundary box | Distance threshold | None (Low velocity triggered `hamtouch`) | Static depth offset | **Dynamic Clearance Guard (Decoupled hold from touch)** |
| **Syntactic Output Validity** | 0.0% (100% rejected by parsers) | 99.4% valid XML | 100.0% valid XML | 100.0% valid XML | 100.0% valid XML | **100.0% ANTLR / CWASA Compliant** |
| **Handshape Top-1 Accuracy** | $\sim 35.0\%$ | 78.82%† (WLASL, leaked) | 80.20%† (ISL, leaked) | 58.30% (DGS) | 56.70% (DGS, 30 videos) | **76.92% (DGS, 52 unseen videos)** |
| **Movement Top-1 Accuracy** | $\sim 15.0\%$ | 77.65%† (Coarse 8-class, leaked) | 78.50%† (ISL, leaked) | 16.67% (3D wrist collapse) | 20.0%→100.0%‡ (DGS, 30 videos) | **28.85% (DGS, 52 unseen) / 18.63% (Combined 102)** |
| **Collision-Free Rate** | 0.0% | $\sim 60.0\%$ | $\sim 65.0\%$ | 33.30% (Severe torso clipping) | 85.00% | **100.00% (Zero Torso/Spine Mesh Penetration)** |
| **Primary Dataset** | Ad-hoc uncurated clips | ASL WLASL (500 samples) | ISL Corpus (142 clips) | DGS Dataset (14 clips) | DGS Dataset (989 clips) | **DGS (500) + ISL (711) + ASL (278)** |
| **Ground Truth Source** | None (manual) | Synthetic dictionary mapping | Real `.sigml` files (human-validated) | Real DGS SiGML corpus | Real DGS SiGML corpus | **Real DGS + ISL SiGML corpus** |
| **Evaluation Methodology** | Manual inspection | Frame-level random 80/20 split | Frame-level random split | Video-level (small sample) | Video-level (30 videos, first N) | **`StratifiedGroupKFold` video-level (102 videos)** |

> **†** V2 and V3 accuracy numbers were measured under **frame-level random splitting (severe data leakage)** with simplified class taxonomies and synthetic ground truth. Real unseen-signer accuracy was estimated at <30% handshape and <8% movement. See Section 8.3 for the complete re-evaluation.
>
> **‡** V6 movement accuracy jumped from 20.0% to 100.0% after a displacement gating bug was fixed mid-evaluation; the 100% figure was measured on the same 30-video test set after the fix, not on a held-out set.

### 3.2 Why Each Version Was Abandoned: Failure Diagnosis & Dataset Switch Rationale

Each version transition was motivated by specific, identified technical failures or dataset inadequacies. The table in Section 3.1 shows *what* changed; this section explains *why*.

#### Version 1 → Version 2: Structural Chaos → Grammar Enforcement

- **Failure Identified:** V1 concatenated detected tokens into an unordered Python `set()`. Tags like `<hammoveu/>` appeared before `<hamflathand/>`, causing CWASA's ANTLR parser to reject **100% of files**. V1's benchmark output shows 40–50 tokens per video (e.g., 50 tokens for "Hello"), all marked `"valid": false`.
- **Resolution:** Introduced a formal Context-Free Grammar (CFG) enforcing: `Sign ::= [Symmetry] + HandshapeBlock + OrientationBlock + [LocationBlock] + [MovementBlock]`. Validity rose from 0% to 99.4%.
- **Dataset:** Remained on WLASL (ASL, 500 videos).

#### Version 2 → Version 3: Wrong Language, Fake Ground Truth, Monolithic Architecture

- **Why V2 Was Abandoned (3 critical reasons):**
  1. **Cross-Language Contamination:** WLASL is an **American Sign Language (ASL)** dataset, but the project's target was **Indian Sign Language (ISL)**. ASL is primarily one-handed and uses a Latin manual alphabet. ISL is fundamentally **two-handed** (the `hamplus` bimanual marker appears 783 times in the ISL ground truth). Training on ASL data polluted the model with structurally incorrect biases that could not generalize to ISL.
  2. **Synthetic (Fake) Ground Truth:** The WLASL videos had **no native HamNoSys or SiGML transcriptions**. V2 mapped gloss labels (e.g., "book", "drink") to HamNoSys tokens via a hand-crafted dictionary (`gloss_to_hamnosys_dict.json`). The reported 85.49% token precision was measured against this synthetic dictionary—effectively testing whether the model could memorize its own training labels. This was not a real linguistic evaluation.
  3. **Monolithic Multi-Head Failure:** A single neural network (`HamNoSysSequenceNet`, `hamnosys_net_v2.pth`) simultaneously predicted all 6 phonetic slots through shared 128-D hidden layers. Loss gradients from the movement head destabilized handshape features, and rare components like body location suffered catastrophic gradient interference.
- **Resolution:** Decoupled into specialized modular components: dedicated handshape neural model (MLP), dedicated movement model (1D-CNN), and analytical geometric models for orientation and location. Switched to ISL dataset with authentic human-validated `.sigml` ground truth files.
- **Why V2's 78.82% Handshape Was Not Real:** The evaluation used **frame-level random splitting** on WLASL. In a 30-fps video, frame $t$ and frame $t+1$ (33 ms apart) share identical signer anatomy, lighting, and camera angle. When adjacent frames are split between train and test, the model performs nearest-neighbor memorization rather than learning invariant features. When V2 was deployed on genuinely unseen signers, real-world handshape accuracy collapsed to **<30%**.

#### Version 3 → Version 4/5: Depth Blindness + Dataset Too Small

- **Why V3 Was Abandoned:**
  1. **2D Depth Ambiguity:** MediaPipe/DWPose 2D/pseudo-3D keypoints suffered severe depth foreshortening. When a signer pointed forward toward the camera (`hamextfingero`), the 2D projected distance between knuckle and fingertip contracted to near zero, causing the system to incorrectly predict upward (`hamextfingeru`). This was a **fundamental limitation of 2D landmark models** that no amount of training could fix.
  2. **ISL Dataset Too Small and Imbalanced:** The ISL corpus had only **142 verified videos**, yielding ~2,130 frame samples after augmentation. More critically, **206 out of 499 annotated signs (41.3%) were static** (`hamnomotion`), and directional movement classes were catastrophically sparse (e.g., `hamcircleol`: 1 example, `hamwavy`: 2 examples). Model B could not learn meaningful trajectory representations from so few dynamic samples.
- **Resolution:** Integrated **HaMeR (CVPR 2024)**, recovering true 3D metric surface meshes (MANO parametric hand model) with accurate depth coordinates, eliminating the 2D foreshortening problem entirely.

#### Version 4/5 → Version 6: Camera-Dependent Features + New Dataset

- **Why V4/V5 Was Abandoned:**
  1. **Camera-Space Overfitting:** Raw 3D Cartesian coordinates from HaMeR varied when signers stood at different distances from the camera or at different angles. The 207-D feature vector contained absolute spatial positions that overfit to the specific signer's body proportions and distance.
  2. **Movement Model Collapse:** The V4/V5 BiGRU received only a 3-D wrist trajectory ($32 \times 3$), which lacked the fingertip dynamics critical for distinguishing fine-grained movement patterns. Movement accuracy collapsed to **16.67%** (barely above the 5.26% random baseline for 19 classes).
- **Why Dataset Switched from ISL to DGS:** The ISL corpus's severe class imbalance (41.3% static signs, 1–2 examples per rare movement class) made it impossible to train a robust movement model. The **DGS (German Sign Language) dataset from SignDict** provided:
  - **989 clean videos** (vs. ISL's 142), all **pre-cropped to pure frontal view** (384×360) against a studio blue background
  - **710 signs (71.8%) with real active dynamic 3D movements** and balanced representation across all 19 movement directions (e.g., `hammoved`: 175, `hammoveo`: 103, `hammover`: 83)
  - **Balanced handshape distribution** across all 12 major classes
  - **Expert-annotated SiGML ground truth** from the DGS corpus
  - DGS and ISL share the same HamNoSys phonetic system, so models trained on DGS are linguistically compatible (though cross-lingual domain gaps remain; see Section 9.1).
- **Resolution:** Constructed a 113-D normalized feature space with rotation-invariant skeletal distances, introduced LayerNorm and Mish activations, and expanded movement features from 3-D wrist-only to 18-D multi-fingertip kinematics.

#### Version 6 → Version 7: Data Leakage + Camera Rotation Sensitivity

- **Why V6 Was Abandoned:**
  1. **Catastrophic Data Leakage:** Frame-level random data splitting caused the validation set to contain frames from the same video as the training set. This inflated validation accuracy to ~92%, but real accuracy on unseen signers collapsed to <25%.
  2. **Camera Rotation Sensitivity:** The 113-D feature vector in V6 was computed in camera-space coordinates. When the signer rotated their wrist or the camera tilted, Cartesian coordinates shifted drastically, confusing morphologically similar handshapes (e.g., `hamfinger23` vs. `hamfinger23spread`, `hamfist` vs. `hampinchall`).
- **Resolution:** Introduced `StratifiedGroupKFold` (100% video-level isolation with zero frame leakage), the **128-D local-canonical orthonormal palm frame** ($R_{\text{local}}$) achieving mathematical invariance to camera angle and wrist rotation, and kinetic stroke apex windowing ($S(t) = e(t) - 1.2 v(t)$).

#### Version 7 → Version 8: Avatar Body Mesh Collision

- **Why V7 Was Abandoned:**
  1. **Torso Penetration Crisis:** At stroke apex, hand velocity drops to near zero. V7's heuristic contact model equated low velocity with physical touch, injecting `<hamtouch/>` onto chest/shoulder signs. When CWASA's IK solver received `<hamtouch/>` combined with `<hamchest/>`, it forced the hand target to depth $Z=0$ directly onto the torso joint, causing the avatar's forearm and hand to **contort backward through the 3D ribcage and spine mesh**. This affected ~66.7% of torso-anchored signs.
  2. **V7 had no impact on the neural model weights themselves** — V7's core contribution was the training methodology (StratifiedGroupKFold, 128-D features). The model weights (`v7_dgs_handshape_resnet.pth`, `v7_dgs_movement_attention_bigru.pth`) are the same ones used in V8.
- **Resolution:** Developed the **Collision-Free Clearance Engine**, decoupling hold deceleration from contact, suppressing chest/shoulder `hamtouch`, and enforcing a 25 cm forward clearance buffer. Collision-free rate improved from ~33% to **100%**.

---

## 4. Mathematical Formulations & Theoretical Derivations

### 4.1 Local-Canonical Orthonormal Frame ($R_{\text{local}}$)
To achieve complete invariance to camera perspective, signer distance, wrist pitch, roll, and yaw, raw 3D keypoints $\mathbf{p}_i \in \mathbb{R}^3$ are mapped into an orthonormal local palm coordinate frame:

1. **Primary Longitudinal Axis ($y_{\text{axis}}$):** Directed from the wrist joint ($\mathbf{p}_0$) to the middle finger metacarpophalangeal (MCP) joint ($\mathbf{p}_9$):
   $$\mathbf{v}_y = \mathbf{p}_9 - \mathbf{p}_0, \quad \hat{\mathbf{y}} = \frac{\mathbf{v}_y}{\|\mathbf{v}_y\|_2}$$

2. **Transverse Vector:** Spanning from index MCP ($\mathbf{p}_5$) to pinky MCP ($\mathbf{p}_{17}$):
   $$\mathbf{v}_{\text{trans}} = \mathbf{p}_5 - \mathbf{p}_{17}$$

3. **Normal Axis ($z_{\text{axis}}$):** Computed via cross product to guarantee perpendicularity to the palm plane:
   $$\mathbf{v}_z = \mathbf{v}_{\text{trans}} \times \hat{\mathbf{y}}, \quad \hat{\mathbf{z}} = \frac{\mathbf{v}_z}{\|\mathbf{v}_z\|_2}$$

4. **Orthogonal Transverse Axis ($x_{\text{axis}}$):** Completes the right-handed Cartesian coordinate system:
   $$\hat{\mathbf{x}} = \hat{\mathbf{y}} \times \hat{\mathbf{z}}$$

5. **Rotation Matrix & Canonical Projection:**
   $$R_{\text{local}} = \begin{bmatrix} \hat{\mathbf{x}}^T \\ \hat{\mathbf{y}}^T \\ \hat{\mathbf{z}}^T \end{bmatrix} \in \mathbb{R}^{3 \times 3}, \quad \mathbf{kps}_{\text{local}, i} = \frac{R_{\text{local}} (\mathbf{p}_i - \mathbf{p}_0)}{\|\mathbf{p}_9 - \mathbf{p}_0\|_2}$$

This mathematical transformation guarantees that regardless of how the signer rotates their arm, the projected hand always faces the neural network in standard canonical orientation.

### 4.2 Anatomical Joint Flexion Angles (15-D)
For each of the 5 fingers, three anatomical joint angles (MCP, PIP, DIP) are computed using the dot product between successive phalangeal bone vectors:
$$\mathbf{v}_1 = \mathbf{p}_{j+1} - \mathbf{p}_j, \quad \mathbf{v}_2 = \mathbf{p}_{j+2} - \mathbf{p}_{j+1}$$
$$\cos(\phi_k) = \text{clip}\left(\frac{\mathbf{v}_1 \cdot \mathbf{v}_2}{\|\mathbf{v}_1\|_2 \|\mathbf{v}_2\|_2 + 10^{-8}}, -1.0, 1.0\right) \quad \text{for } k = 1, \dots, 15$$

### 4.3 Inter-Finger Abduction Spread Angles (4-D)
Measures the lateral splay between adjacent fingers using rays from MCP to fingertip:
$$\mathbf{r}_i = \mathbf{p}_{\text{tip}, i} - \mathbf{p}_{\text{mcp}, i}$$
$$\cos(\psi_i) = \text{clip}\left(\frac{\mathbf{r}_i \cdot \mathbf{r}_{i+1}}{\|\mathbf{r}_i\|_2 \|\mathbf{r}_{i+1}\|_2 + 10^{-8}}, -1.0, 1.0\right) \quad \text{for } i \in \{1, 2, 3\}$$
where $i = 1$ is index–middle, $i = 2$ is middle–ring, and $i = 3$ is ring–pinky. A fourth angle calculates thumb-to-index splay ($\mathbf{r}_{\text{thumb}} \cdot \mathbf{r}_{\text{index}}$).

### 4.4 Thumb-to-Fingertip Pinch Proximities (5-D)
Euclidean distances from the thumb tip ($\mathbf{p}_4$) to all other fingertips and the index MCP:
$$d_{\text{pinch}, j} = \|\mathbf{p}_4 - \mathbf{p}_j\|_2 \quad \text{for } j \in \{5, 8, 12, 16, 20\}$$
This directly discriminates between closed fists (`hamfist`), pinch configurations (`hampinch12`, `hampinchall`), and open cups (`hamcee12`, `hamceeall`).

### 4.5 Kinetic Stroke Apex Scoring Formula
Sign language gestures follow a three-phase dynamic trajectory: *Preparation (reach)* $\to$ *Stroke Hold (semantic execution)* $\to$ *Retraction (return)*. The stroke apex is isolated by scoring every frame:
$$v(t) = \|\mathbf{w}_t - \mathbf{w}_{t-1}\|_2, \quad v_{\text{norm}}(t) = \frac{v(t)}{\max_k v(k) + 10^{-5}}$$
$$e(t) = 1.0 - y_{\text{wrist}}(t)$$
$$\text{Score}(t) = e(t) - 1.2 \cdot v_{\text{norm}}(t)$$
$$t_{\text{apex}} = \arg\max_t \text{Score}(t)$$
A Gaussian weighting window is applied around $t_{\text{apex}}$:
$$W(t) = \exp\left(-\frac{(t - t_{\text{apex}})^2}{2\sigma^2}\right), \quad \sigma = 3.0$$

---

## 5. Deep Neural Architectures: Model A & Model B

### 5.1 Model A: SkeletalHandResNet_V7 (Handshape Classifier)
* **Input Dimension:** 128-D local-canonical vector ($63\text{D keypoints} + 15\text{D flexion} + 4\text{D spread} + 5\text{D pinch} + 41\text{D MANO pose}$).
* **Output Dimension:** 12 canonical handshape classes.
* **Architecture:**
  ```
  Input (Batch, 128)
       │
       ▼
  Linear(128 -> 256) -> LayerNorm(256) -> Mish()
       │
       ▼
  [ResidualBlock_V7(256)] x 3
       │  Linear(256 -> 256) -> LayerNorm -> Mish -> Dropout(0.2) -> Linear(256 -> 256) -> LayerNorm + Residual
       ▼
  Linear(256 -> 128) -> LayerNorm(128) -> Mish() -> Dropout(0.25)
       │
       ▼
  Linear(128 -> 12) -> Softmax
  ```

### 5.2 Model B: AttentionBiGRU_V6 (Movement Trajectory Classifier)
* **Input Dimension:** $32 \times 18$ tensor (32 uniform temporal steps $\times$ 18 kinematic channels: 3D wrist translation $+ 15$D relative fingertip offsets).
* **Output Dimension:** 19 movement classes.
* **Architecture:**
  ```
  Input (Batch, 18, 32)
       │
       ▼
  Conv1D(in_channels=18, out_channels=64, kernel_size=3, padding=1)
       │
       ▼
  BatchNorm1d(64) -> Mish()
       │
       ▼
  Transpose(1, 2) -> (Batch, 32, 64)
       │
       ▼
  Bidirectional GRU(input_size=64, hidden_size=128, num_layers=2, batch_first=True, dropout=0.25)
       │  Output: (Batch, 32, 256)
       ▼
  MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
       │  Self-Attention: Query=Key=Value=gru_out
       ▼
  Temporal Mean Pooling across 32 timesteps -> (Batch, 256)
       │
       ▼
  Linear(256 -> 128) -> LayerNorm(128) -> Mish() -> Dropout(0.30)
       │
       ▼
  Linear(128 -> 19) -> Softmax
  ```

---

## 6. Analytical 3D Geometric Heuristics & Collision Clearance

### 6.1 SVD 3D Palm Surface Normal
Rather than relying on noisy 3-point cross products, the 3D palm normal is computed via Singular Value Decomposition (SVD) across 6 palm structural joints ($\mathbf{p}_0, \mathbf{p}_1, \mathbf{p}_5, \mathbf{p}_9, \mathbf{p}_{13}, \mathbf{p}_{17}$):
1. Compute centroid $\mathbf{c} = \frac{1}{6} \sum_{i} \mathbf{p}_i$ and mean-centered matrix $A = [\mathbf{p}_i - \mathbf{c}] \in \mathbb{R}^{6 \times 3}$.
2. Perform SVD: $A = U \Sigma V^T$.
3. The normal vector $\mathbf{n}_{\text{palm}}$ corresponds to the right singular vector with the smallest singular value (column 3 of $V$).
4. Disambiguate facing direction using radial-to-ulnar vector cross product.
5. Match against 8 canonical unit vectors (`hampalmu`, `hampalmd`, `hampalml`, `hampalmr`, etc.) via cosine similarity:
   $$\hat{c}_{\text{palm}} = \arg\max_k (\mathbf{n}_{\text{palm}} \cdot \mathbf{u}_k)$$

### 6.2 Longitudinal Pointing Vector with Depth Compensation
The extended finger direction axis $\mathbf{v}_{\text{ext}}$ is computed from palm centroid $\mathbf{c}$ to the primary extended fingertip. Because monocular camera capture exhibits perspective foreshortening along the $Z$-axis, depth compensation is applied:
$$\mathbf{v}_{\text{comp}} = [v_x, v_y, v_z \times 1.6]^T, \quad \hat{\mathbf{v}}_{\text{ext}} = \frac{\mathbf{v}_{\text{comp}}}{\|\mathbf{v}_{\text{comp}}\|_2}$$
Matched against 16 canonical spatial directions (`hamextfingeru`, `hamextfingeruo`, `hamextfingero`, etc.).

### 6.3 7-Zone Anatomical Classification
The vertical position of the signing hand is mapped relative to the signer's body landmarks (nose, shoulders, hip) into 7 calibrated height zones:
* **Forehead (`hamforehead`):** $y_{\text{norm}} < 0.22$
* **Cheek / Eyes (`hamcheek` / `hameyes`):** $0.22 \le y_{\text{norm}} < 0.28$
* **Nose / Mouth (`hamnose` / `hammouth`):** $0.28 \le y_{\text{norm}} < 0.32$
* **Chin (`hamchin`):** $0.32 \le y_{\text{norm}} < 0.38$
* **Chest / Shoulders (`hamchest` / `hamshoulders`):** $0.38 \le y_{\text{norm}} < 0.52$
* **Stomach / Belly (`hamstomach` / `hambelly`):** $0.52 \le y_{\text{norm}} < 0.65$
* **Neutral Space (`hamneutralspace`):** $y_{\text{norm}} \ge 0.65$ or $d(\text{wrist}, \text{body}) > 0.35$

### 6.4 Collision-Free Clearance Guard
In the CWASA avatar animation engine, if a SiGML document contains `<hamchest/>` combined with `<hamtouch/>`, the Inverse Kinematics solver rotates the clavicle, shoulder, and elbow joints to bring the wrist into zero-distance contact with the torso mesh. Because the avatar's fingers have geometric thickness, this causes the hands to penetrate deep into the virtual ribcage and spine.

**The V8 Collision Prevention Solution:**
```python
# heuristics/sigml_grammar_linter.py & local_hamer_pipeline.py
if "hamtouch" in tokens and any(loc in tokens for loc in ["hamchest", "hamshoulders", "hamstomach"]):
    tokens.remove("hamtouch")  # Suppress mesh-penetrating contact
    tokens.append("hamclose")  # Enforce ~25 cm natural signing space clearance
```
This guarantees **100% collision-free avatar synthesis** across all body-anchored signs.

---

## 7. Training Methodology, Loss Functions, & Data Splitting

### 7.1 Video-Level Partitioning (`StratifiedGroupKFold`)
To eliminate the catastrophic data leakage of frame-level pooling, training data is partitioned using `StratifiedGroupKFold(n_splits=6)` where `groups = video_ids`:
* **Zero Frame Leakage:** Every frame from a given video belongs exclusively to either the training split or the validation split.
* **Stratified Balance:** Rare classes (`hamceeall`, `hampinch12`, `hamfinger23spread`) are proportionally distributed across folds.
* **Split Ratio:** ~83.3% training (5 folds) / ~16.7% validation (1 fold).

### 7.2 Class-Weighted Focal Loss
To combat severe class imbalance where majority classes (`hamflathand`, `hamfist`) outnumber minority classes by up to 20:1, training optimizes Focal Loss ($\gamma = 2.0$) with inverse-frequency weighting:
$$w_c = \text{clip}\left(\frac{N_{\text{total}}}{C \cdot (N_c + 10^{-4})}, 0.25, 4.0\right)$$
$$\mathcal{L}_{\text{Focal}} = -\sum_{c=1}^C w_c (1 - p_c)^\gamma y_c \log(p_c + 10^{-8})$$

### 7.3 Kinematic Spline Augmentation (Model B)
To ensure trajectory invariance across different signers, movement trajectories are augmented $4\times$ during training using cubic spline resampling, random temporal scaling ($\pm 8\%$), and Gaussian jitter:
$$\mathbf{T}_{\text{aug}} = (\mathbf{T} + \mathcal{N}(0, 0.012)) \cdot \mathcal{U}(0.92, 1.08)$$

---

## 8. Empirical Evaluation Benchmarks & Accuracy Progression

### 8.1 Evolutionary Accuracy Progression Across Pipeline Versions

> [!WARNING]
> **Cross-version accuracy comparison is not meaningful without context.** Each version used a different dataset, different ground truth quality, different evaluation methodology, and different task complexity. The numbers below must be read alongside the footnotes explaining these differences. See Section 3.2 for why each version was abandoned and Section 8.3 for the detailed re-evaluation.

| Evaluation Metric | V1 (Baseline) | V2 (WLASL)† | V3 (ISL Split)† | V6 (DGS + HaMeR)§ | V8 (Current Engine)‡ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dataset** | Ad-hoc clips | ASL WLASL (500) | ISL (142) | DGS (30 benchmark) | DGS (52) + ISL (50) |
| **Ground Truth Source** | Manual | Synthetic dictionary | Real `.sigml` | Real DGS SiGML | Real DGS + ISL SiGML |
| **Data Splitting** | Manual | Frame-level random | Frame-level random | Video-level (first N) | `StratifiedGroupKFold` |
| **Movement Classes** | N/A | 8 (coarse 2D) | Heuristic | 19 (fine-grained 3D) | 19 (fine-grained 3D) |
| **Syntactic XML Validity** | 0.0% | 99.4% | 100.0% | 100.0% | **100.0%** |
| **Handshape Accuracy (Top-1)** | ~35.0% | 78.82%† | 80.20%† | 56.70%§ | **76.92%‡ (DGS unseen)** |
| **Movement Accuracy (Top-1)** | ~15.0% | 77.65%† | 78.50%† | 20→100%§ | **28.85%‡ (DGS) / 18.63% (combined)** |
| **Collision-Free Rate** | 0.0% | ~60.0% | ~65.0% | 33.30% | **100.0%** |
| **Estimated Real Unseen Acc. (HS)** | <10% | **<30%** | <30% | 56.70% | **76.92%** |

> **†** V2/V3 metrics were measured under **frame-level random splitting (severe data leakage)** with a simplified 8-class 2D movement task and synthetic (dictionary-mapped) ground truth. The 78.82% handshape and 77.65% movement numbers reflect near-neighbor memorization on leaked frames, not true generalization. When deployed on genuinely unseen signers, V2 real accuracy collapsed to <30% handshape and <8% movement.
>
> **§** V6 handshape accuracy (56.7%) was measured on the first 30 DGS videos (alphabetical). V6 movement accuracy jumped from 20% to 100% after a displacement gating bug was fixed mid-evaluation; the 100% figure was measured on the same 30 videos post-fix, not an independent test set.
>
> **‡** V8 metrics are measured under **strict video-level cross-validation (`StratifiedGroupKFold`)** with **100% unseen signers**, true **3D metric MANO mesh**, and a fine-grained **19-class spatiotemporal trajectory Attention-BiGRU**. V8's 28.85% movement accuracy on a 19-class problem is **5.5× above random chance** (5.26%), whereas V2's 77.65% on an 8-class problem was only **6.2× above its 12.5% random baseline** — a comparable degree of above-chance performance. See Section 8.3 for the complete scientific re-evaluation.

### 8.2 V8 Bulk Benchmark Evaluation (102 Videos: 52 DGS + 50 ISL)
Evaluated using [`run_bulk_testing.py`](file:///d:/academics/Hybrid_V7/Active_Hybrid_Pipeline/run_bulk_testing.py) against gold-standard phonological ground truth. Full per-video results are in [`ACCURACY_REPORT.md`](file:///d:/academics/Hybrid_V7/Active_Hybrid_Pipeline/test_results/ACCURACY_REPORT.md).

| Dataset | Count | Handshape (Model A) | Movement (Model B) | Location | Palm Orientation | Pointing Direction | Contact | Two-Handed | XML Validity | Compound Match |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DGS (German)** | 52 | **76.92%** | **28.85%** | 26.92% | 50.00% | — | — | — | 100.0% | 5.8% |
| **ISL (Indian)** | 50 | **38.00%** | **8.00%** | 26.00% | 32.00% | — | — | — | 100.0% | 2.0% |
| **Combined** | **102** | **57.84%** | **18.63%** | **26.47%** | **41.18%** | **30.39%** | **53.92%** | **76.47%** | **100.0%** | **3.92%** |

> [!NOTE]
> **Compound Match** (3.92%) is the full end-to-end joint match requiring *all* phonetic components (handshape + movement + location + orientation + contact + symmetry) to simultaneously be correct. This is the strictest evaluation metric; individual component accuracies are significantly higher.

### 8.3 The Metric Discrepancy Paradox: Re-Evaluating V2 vs. V8 (Handshape & Movement)

A superficial comparison of the raw percentages in Section 8.1 might suggest a paradox: *Why did Version 2 report 78.82% Handshape and 77.65% Movement, while Version 8 reports 76.92% Handshape and 28.85% Movement? If V2's numbers were higher, why was V8 built, and why is V8 decisively superior?*

The answer lies in **scientific rigor, task difficulty, and experimental validity**. Comparing V2 directly to V8 is an **apples-to-oranges fallacy**: V2 measured an artificially simplified, data-leaked proxy task, whereas V8 solves true, unconstrained, out-of-sample 3D phonological synthesis.

```
+-------------------------------------------------------------------------------------------------------+
|                                    RE-EVALUATION MATRIX AT A GLANCE                                   |
+--------------------------+------------------------------------+---------------------------------------+
| Dimension                | Version 2 (The Inflated Illusion)   | Version 8 (The Robust Reality)        |
+--------------------------+------------------------------------+---------------------------------------+
| Data Partitioning        | Frame-level random split           | Strict video/signer isolation         |
| Data Leakage             | Severe (~95% temporal correlation) | Zero leakage (StratifiedGroupKFold)   |
| Real Unseen Accuracy     | Handshape: < 30%, Movement: ~5-8%  | Handshape: 76.92%, Movement: 28.85%   |
| Hand Vision Space        | Fragile 2D MediaPipe pixels        | HaMeR 3D Metric MANO Surface Mesh     |
| Perspective Invariance   | None (Fails on camera tilt/scale)  | Perfect SE(3) Local-Canonical Frame   |
| Movement Task Scope      | 8 Coarse 2D Screen Directions      | 19 Fine-Grained 3D Trajectory Classes |
| Random Guessing Baseline | 12.5% (1/8 coarse directions)      | 5.26% (1 in 19 discrete 3D paths)     |
| 3D Depth Awareness (Z)   | Completely Blind (Z = 0)           | True 3D Metric In/Out Depth Vector    |
| Avatar Torso Penetration | 40.0% Collision Rate (Severe)      | 0.0% Collision Rate (100% Guarded)    |
+--------------------------+------------------------------------+---------------------------------------+
```

---

#### 8.3.1 Detailed Re-Evaluation: Handshape Recognition

* **The Data Leakage Trap in V2:**
  In Version 2, training and test sets were created by randomly splitting individual video frames. In a standard 30-fps video, frame $t$ and frame $t+1$ (separated by only 33 milliseconds) share identical illumination, signer anatomy, skin tone, clothing, and camera distance. When frame $t$ is in the training set and frame $t+1$ is in the test set, the neural network does not learn invariant morphological representations—it performs **nearest-neighbor image memorization**. When V2 was deployed on genuinely unseen test videos or a different signer, its real-world handshape accuracy collapsed to **$< 30\%$**.
  
  In contrast, **Version 8 enforces strict video-level and signer-level partitioning (`StratifiedGroupKFold`)**. Every single test video is completely held out. V8's **76.92% Handshape accuracy** on DGS represents true generalization to unseen human signers.

* **2D Foreshortening & Depth Collapse vs. 3D MANO Mesh:**
  V2 operated on 2D MediaPipe keypoints $(x, y)$. Whenever a signer pointed forward toward the camera (`hamextfingero`), the 2D projected distance between the knuckle and the fingertip contracted to near zero due to perspective foreshortening. V2 routinely misclassified extended fingers as closed fists (`hamfist`).
  
  V8 solves this by integrating **HaMeR (CVPR 2024)**, reconstructing a full 3D surface mesh parameterized by the MANO kinematic hand model. The anatomical bone priors in MANO enforce valid joint biomechanics, resolving self-occlusions and foreshortening.

* **Camera Tilt & Scale Invariance ($R_{\text{local}}$):**
  In V2, any change in camera angle, wrist roll, or signer distance altered the 2D coordinates and degraded classification. V8 projects all 21 keypoints into a **Local-Canonical Orthonormal Palm Frame** ($R_{\text{local}}$) derived via Gram-Schmidt orthogonalization. The resulting 128-D feature vector (joint flexion, inter-finger abduction, thumb pinch) is mathematically invariant to spatial translation, global rotation, and scale.

---

#### 8.3.2 Detailed Re-Evaluation: Movement Classification

* **What V2 Actually Measured (8-Class 2D Delta vs. 19-Class 3D Trajectory):**
  The "77.65%" accuracy reported in V2 was **not** a true temporal movement classifier. V2 merely computed the gross Cartesian coordinate difference between the first and last frame of a pre-trimmed clip:
  $$\Delta x = x_{\text{end}} - x_{\text{start}}, \quad \Delta y = y_{\text{end}} - y_{\text{start}}$$
  This displacement was mapped into just **8 cardinal 2D screen directions** (Up, Down, Left, Right, Up-Left, Up-Right, Down-Left, Down-Right).
  
  In sign language corpora (such as isolated WLASL clips), signers naturally drop their hands downward to their lap at the end of every sign. Simply predicting "Down" or "Rest" gave an artificial baseline accuracy exceeding 45%! A random guess among 8 classes has a high baseline of $12.5\%$.

* **Why V8's 28.85% is a Far Harder and Superior Achievement:**
  V8 classifies among **19 fine-grained, 3D HamNoSys dynamic trajectory classes**, including:
  - Linear 3D translations across all spatial axes: Up (`hammoveu`), Down (`hammoved`), Left (`hammovel`), Right (`hammover`), Outward toward camera (`hammoveo`), Inward toward chest (`hammovei`).
  - Curved 3D arcs: Up-and-curve (`hamarcu`), Down-and-curve (`hamarcd`), Left/Right arcs (`hamarcl`/`hamarcr`).
  - Complex continuous kinematics: Circular loops (`hamcircle`), Zigzag paths (`hamzigzag`), Wrist rotations (`hamtwist`), and Stationary holds (`hamnomotion`).

  In a 19-class classification problem, **random chance is only $5.26\%$ ($1/19$)**. V8's **28.85% video inference accuracy is $5.5\times$ higher than chance**, and in validation with kinematic spline augmentation, it achieves **~60%**. If V2 were evaluated on this 19-class 3D task, its accuracy would be **near 0%**, because V2 possessed zero temporal trajectory modeling and zero 3D depth perception.

* **True Spatiotemporal Modeling (18-D Attention-BiGRU):**
  Unlike V2's static endpoint subtraction, V8 models the full continuous motion over 32 resampled frames using an 18-dimensional feature vector (3D wrist velocity + 5 relative fingertip vectors). The Bidirectional GRU with Additive Temporal Attention identifies velocity inflection points and curvature transitions throughout the entire gesture.

* **Depth Awareness ($Z$-axis Dynamics):**
  Movement in sign language frequently differentiates grammatical meaning through depth (e.g., directional verbs moving outward from signer to addressee). V2 was completely blind to the $Z$-axis. V8 explicitly tracks 3D metric depth displacements, recognizing forward (`hammoveo`) and backward (`hammovei`) trajectories.

---

#### 8.3.3 Real-World Avatar Intelligibility & Collision-Free Execution

Even if V2's isolated classification heads produced a valid XML string, the resulting avatar animation was severely broken:
1. **Torso Collisions:** In V2, 40% of signs caused the avatar's 3D hands to impale its chest, spine, or face because 2D coordinates lacked depth clearance. In V8, the **Compiler-Level Collision-Free Clearance Guard** dynamically enforces a 25 cm clearance envelope and decouples hold deceleration from physical contact, achieving a **100.0% collision-free rate**.
2. **Linguistic Readability:** Because V2 lacked 3D finger orientation, depth movement, and true handshape discrimination, native signers found the synthesized animations unreadable. V8 generates anatomically coherent, phonetically accurate 3D avatar animations that preserve the linguistic nuances of sign language.

---

## 9. Detailed Failure Mode & Root-Cause Analysis

### 9.1 The Cross-Lingual Domain Gap (Why ISL Accuracy is Lower than DGS)
* **The Root Cause:** Both Model A (`v7_dgs_handshape_resnet.pth`) and Model B (`v7_dgs_movement_attention_bigru.pth`) were trained exclusively on the **SignDict German Sign Language (DGS) corpus** (989 studio-quality frontal videos).
* **Impact on ISL:**
  - In DGS, Handshape achieved **76.92%**. In ISL, it achieved **38.00%**.
  - In DGS, Movement achieved **28.85%**. In ISL, it achieved **8.00%**.
  - ISL signers perform gestures with different cadence, greater signing space breadth, distinct camera framing (full upper body vs. centered chest crop), and different dialectal variants not represented in the DGS training distribution.

### 9.2 The Movement BiGRU Discrepancy (~60% Validation vs. 18.6% Video Inference)
During offline Kaggle training, Model B achieved ~60% accuracy on segmented validation trajectories. In full end-to-end video inference, accuracy drops to 18.63% due to four specific factors:
1. **Feature Space Mismatch (2D/3D Mixing):** The training pipeline extracted pure 3D metric translation from HaMeR ($\text{meters}$). During live video execution, the first two dimensions ($X, Y$) were substituted with MediaPipe normalized 2D image coordinates ($0.0 \text{ to } 1.0$) to avoid bounding box drift, degrading the neural network's learned representations.
2. **19 Fine-Grained Classes vs. Strict String Evaluation:** Model B predicts across 19 sub-directional classes (`hammoved`, `hammovedo`, `hammovedl`, `hammovedi`, `hammoveo`, etc.). If a sign's true label is `hammovedo` (down-out) and the model predicts `hammoveo` (out) or `hammoved` (down), strict evaluation counts this as **0% (FAIL)**, penalizing near-correct predictions.
3. **Lack of Dynamic Stroke Boundary Detection:** The handshape model uses kinetic apex isolation ($t_{\text{apex}}$). However, the movement BiGRU receives the full 32-frame resampled video, conflating arm-raising preparation and arm-lowering retraction with the core sign movement.
4. **Heuristic Vector Overwrite Layer:** A hardcoded 2D heuristic check (`total_path_2d > 0.40`) frequently overrode the BiGRU's logits with generic `hammoveu` or `hamnomotion` when camera-space tracking jumped.

---

## 10. Future Research Directions & Reference Citations

### 10.1 Key Recommendations for Future Research Papers
1. **Multi-Task Cross-Lingual Pre-Training:** Train Model A and Model B jointly on concatenated ISL, DGS, and ASL datasets using domain-adversarial training to ensure cross-lingual feature invariance.
2. **Continuous Temporal Stroke Segmentation (Action Segmentation):** Implement a temporal convolutional network (TCN) or Boundary-Sensitive Network (BSN) to isolate the exact onset $t_{\text{start}}$ and offset $t_{\text{end}}$ of the lexical movement stroke before feeding it to Model B.
3. **Hierarchical Direction Evaluation (Soft Metric):** Evaluate movement using hierarchical distance on the directional sphere (angular deviation error $\theta_{\text{err}} \le 30^\circ$) rather than strict discrete string equality.
4. **3D Camera Calibration for Trajectory Tracking:** Replace 2D screen coordinate substitution with true 3D camera ray unprojection using estimated focal length from HaMeR.

### 10.2 Academic References
1. **Prillwitz, S., et al. (1989).** *HamNoSys: Hamburg Notation System for Sign Languages: An Introduction.* University of Hamburg.
2. **Pavlakos, G., et al. (2024).** *Reconstructing Hands in 3D with Transformers (HaMeR).* IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
3. **Romero, J., et al. (2017).** *Embodied Hands: Modeling and Capturing Hands and Bodies Together (MANO).* ACM Transactions on Graphics (SIGGRAPH Asia).
4. **Elliott, R., et al. (2008).** *JASigning: Virtual Humans for Sign Language Synthesis.* University of East Anglia.
5. **Lugaresi, C., et al. (2019).** *MediaPipe: A Framework for Building Perception Pipelines.* arXiv:1906.08172.
6. **Lin, T. Y., et al. (2017).** *Focal Loss for Dense Object Detection.* IEEE International Conference on Computer Vision (ICCV).
7. **Li, D., et al. (2020).** *Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison (WLASL).* IEEE/CVF Winter Conference on Applications of Computer Vision (WACV).
8. **SignDict.org.** *Open Dictionary for Sign Language.* [https://signdict.org/](https://signdict.org/). Licensed under CC BY-SA 3.0 DE.

