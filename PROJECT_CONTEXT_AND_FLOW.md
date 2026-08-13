# Project Technical Architecture, Pipeline Flow & UI Feature Context Blueprint
**System Name:** Universal Sign Language to 3D Avatar Synthesizer (HamNoSys Pipeline)  
**Repository:** `HamNoSys_Group14_V2` / `Avatar-Generator-FYP-`  
**Purpose:** Provide complete, unabridged technical context, data flow, pipeline architecture, and UI feature specifications to any AI Agent or UI Designer Model for generating optimal UI/UX designs, wireframes, and layouts.

---

## 1. Project Overview & Core Mission
The **Universal Sign Language to 3D Avatar Synthesizer** is an AI-powered sign language translation and rendering platform. It converts input sign language videos (e.g., Indian Sign Language ISL, American Sign Language ASL, WLASL, European Sign Language) into standard phonetic sign language notation known as **HamNoSys** (Hamburg Notation System). 

The system then converts the HamNoSys notation into **SiGML XML** (Signing Gesture Markup Language) and renders a real-time **3D WebGL Animated Avatar** performing the exact sign gesture.

### Key Objectives:
1. **Bridge Communication Barriers**: Provide visual 3D avatar rendering of sign language gestures alongside human-readable English glosses and translations.
2. **Standardized Phonetic Notation**: Translate raw video pixels into international HamNoSys phonetic tokens, capturing handshape, orientation, location, contact, and movement.
3. **Cross-Language Translation Foundation**: Build a universal framework capable of translating any input sign language into any target sign language gesture.

---

## 2. End-to-End System Pipeline & Data Flow

```
[Input Sign Video / Camera Feed / Sample Clip]
                       │
                       ▼
[Stage 1: Frame Extraction & Preprocessing (OpenCV)]
                       │
                       ▼
[Stage 2: Landmark Tracking Engine (RTMPose / MediaPipe)]
 ├── 21 Hand Keypoints per hand (Left & Right)
 ├── 468 Facial Mesh Landmarks
 └── 33 Body Pose Landmarks
                       │
                       ▼
[Stage 3: Temporal Smoothing & Feature Extraction (temporal_utils.py)]
                       │
                       ▼
[Stage 4: 10-Module AI Feature Classification Engine]
 ├── Module 1: Handshape Model (clf_finger_signer/right/bird.pkl)
 ├── Module 2: Orientation Model (ori_model2.py)
 ├── Module 3: Upper Body Location Tracker (upper_body_locations_video.py)
 ├── Module 4: Head & Face Location Tracker (Head_and_face_location.py)
 ├── Module 5: Hand Relative Location Tracker (hand_location_video_P.py)
 ├── Module 6: Finger Placement Tracker (fing_locations_d.py)
 ├── Module 7: Contact Type Recognizer (contact_types_e.py)
 ├── Module 8: Arm & Space Position Model (Arm_and_Space_positions.py)
 ├── Module 9: Primary Movement Model (movement1_prava.py)
 └── Module 10: Secondary Movement & Transition Model (Movement_2.py)
                       │
                       ▼
[Stage 5: Context-Free Grammar (CFG) Assembly Engine (integration_pipeline.py)]
 └── Sign ::= Handshape InitialOrientation BodyLocation [Contact] [Movement] [StateTransition]
                       │
                       ▼
[Stage 6: Phonetic Token Sequence & Unicode Glyph Mapping]
 └── HamNoSys String Tags -> conversionSpreadSheet.txt -> Unicode Hex Glyphs
                       │
                       ▼
[Stage 7: SiGML XML Conversion Subprocess (HamNoSys2SiGML.py)]
 └── Generates formal <sigml><hns_sign>...</hns_sign></sigml> XML markup
                       │
                       ▼
[Stage 8: JASigning CWASA WebGL 3D Avatar Rendering Engine]
 └── Renders Luna, Siggi, Anna, Marc, or Francoise in 3D WebGL Viewport
```

---

## 3. Deep-Dive into Technical Modules & AI Components

### A. Dual Pose & Facial Landmark Tracking Engine
* **Primary Engine**: `rtmlib` (RTMPose whole-body pose estimation). High-speed multi-person keypoint extraction.
* **Hardware Acceleration**: Accelerated via **Intel Iris Xe iGPU** using Microsoft DirectML (`onnxruntime-directml` with `DmlExecutionProvider`) or Intel OpenVINO (`openvino` with `OpenVINOExecutionProvider`) + Python multi-processing (`multiprocessing.Pool`), boosting inference speeds by **10x to 20x**.
* **Fallback Engine**: MediaPipe Holistic (`mp.solutions.holistic`) for CPU fallback.
* **Tracked Keypoints**:
  * 42 Hand keypoints (21 per hand: wrist, thumb joints, index, middle, ring, pinky).
  * 468 Facial mesh keypoints (eyebrows, eyes, nose bridge, lips, cheeks, jawline).
  * 33 Body pose keypoints (shoulders, elbows, wrists, hips).

### B. The 10-Module AI Feature Classifier Suite
1. **Handshape Model (`Handshape_Model.py` & Scikit-Learn Classifiers)**:
   * Classifies hand postures: Flat Hand (`hamflathand`), Open Hand (`hamthumbopenmod`), Index Finger (`hamfinger2`), Victory / Two Fingers (`hamfinger23`), Pinch (`hampinch12`, `hampinchall`), Fist (`hamfist`), Thumb Position (`hamthumboutmod`, `hamthumbacrossmod`).
2. **Orientation Model (`ori_model2.py`)**:
   * Computes 3D spatial direction vectors: Extended Finger Direction (`hamextfingeru` up, `hamextfingerd` down, `hamextfingerr` right, `hamextfingerl` left) and Palm Orientation (`hampalmd` down, `hampalmu` up, `hampalml` left, `hampalmr` right).
3. **Upper Body Location Tracker (`upper_body_locations_video.py`)**:
   * Detects spatial body anchors: Chest (`hamchest`), Shoulders (`hamshoulders`), Head (`hamhead`), Neutral Space (`hamneutralspace`).
4. **Head & Face Location Tracker (`Head_and_face_location.py`)**:
   * Detects precise anatomical facial regions: Cheek (`hamcheek`), Forehead, Chin, Nose, Mouth, Eyes.
5. **Hand Location Tracker (`hand_location_video_P.py`)**:
   * Maps relative 3D coordinate space between dominant and non-dominant hands.
6. **Finger Location & Proximity Tracker (`fing_locations_d.py`)**:
   * Tracks individual finger extensions and inter-digital distance (middle, ring, pinky).
7. **Contact Type Recognizer (`contact_types_e.py`)**:
   * Detects physical contact vs spatial proximity: Touch (`hamtouch`), Near, No-contact.
8. **Arm & Space Position Model (`Arm_and_Space_positions.py`)**:
   * Calculates elbow elevation, arm spread angle, and spatial plane depth.
9. **Primary Movement Model (`movement1_prava.py`)**:
   * Tracks primary directional motion vectors: Upward (`hammoveu`), Downward (`hammoved`), Leftward (`hammovel`), Rightward (`hammover`), Outward (`hammoveo`), Inward (`hammovei`).
10. **Secondary Movement & Transition Model (`Movement_2.py`)**:
    * Tracks complex dynamics: Circular Motion (`hamcircleo`), Wrist Rotation Transition (`hamreplace`), Repeating Nodding (`hamnodding`), Speed/Wavering.

### C. Context-Free Grammar (CFG) Rule Engine (`integration_pipeline.py`)
Combines predictions from all 10 modules into a syntactically valid HamNoSys sentence following standard sign linguistic grammar:
$$\text{Sign} ::= \text{HandshapeStructure} \times \text{InitialOrientation} \times \text{BodyLocation} \times [\text{Contact}] \times [\text{Movement}] \times [\text{StateTransition}]$$
* Inserts two-handed symmetry modifiers (`hamsymmlr`, `hamplus`, `hamnonipsi`) when symmetrical or dual-hand movements are detected.

### D. HamNoSys to SiGML XML & CWASA 3D Avatar System
* **Spreadsheet Mapping (`conversionSpreadSheet.txt`)**: Maps token strings (e.g. `hamflathand`) to custom HamNoSys Unicode hexadecimal codes (e.g. `0xe000`).
* **SiGML Script (`HamNoSys2SiGML.py`)**: Transforms Unicode sequence into valid SiGML XML string:
  ```xml
  <sigml>
    <hns_sign gloss="DYNAMICALLY PREDICTED SIGN">
      <hamnosys_nonmanual/>
      <hamnosys_manual>
        <hamflathand/>
        <hamextfingeru/>
        <hampalmd/>
        <hamchest/>
        <hamtouch/>
        <hammoveo/>
      </hamnosys_manual>
    </hns_sign>
  </sigml>
  ```
* **JASigning CWASA 3D Avatar**: WebGL rendering library (`allcsa.js`, `cwasa.css`) that animates 3D avatars (*Luna*, *Siggi*, *Anna*, *Marc*, *Francoise*) directly inside HTML `<div class="CWASAAvatar av0"></div>`.

---

## 4. Backend API Endpoints & Data Contracts

### Server Configuration (`webapp/app.py`)
* **Framework**: Flask (Python 3) on `http://0.0.0.0:5000`

### API Endpoints
#### `GET /`
* Serves the main application SPA (`templates/index.html`).

#### `POST /upload`
* **Accepts**: 
  * `multipart/form-data` with `video` file upload (`.mp4`, `.mov`, `.avi`), OR
  * `form-data` with `sample_name` key (e.g. `Prompt_1.mp4`, `sample_output.mp4`, `dont_worry.mp4`).
* **Response JSON Schema**:
```json
{
  "hamnosys_tags": "hamflathand hamextfingeru hampalmd hamchest hamtouch hammoveo",
  "hamnosys_unicode": "",
  "symbol_chips": [
    { "tag": "hamflathand", "label": "Flat Handshape" },
    { "tag": "hamextfingeru", "label": "Finger Upward" },
    { "tag": "hampalmd", "label": "Palm Down / Away" },
    { "tag": "hamchest", "label": "Chest Location" },
    { "tag": "hamtouch", "label": "Touch Contact" },
    { "tag": "hammoveo", "label": "Outward Movement" }
  ],
  "sigml": "<sigml><hns_sign gloss=\"...\">...</hns_sign></sigml>",
  "sigml_valid": true,
  "warnings": [],
  "video_url": "/samples/Prompt_1.mp4",
  "filename": "Prompt_1.mp4",
  "gloss": "HELLO / HI",
  "meaning": "3D posture and motion extracted frame-by-frame from raw video landmarks.",
  "confidence": "88.5%",
  "precision": "85.7%"
}
```

---

## 5. Comprehensive Catalog of UI Features to Display

Here is the complete master list of all features, controls, metrics, and visual components in the system that are worth displaying on the user interface:

### 1. Video Input & Ingestion Controls
* **Drag & Drop File Upload Zone**: Supports video files with visual drag-over state and browse button.
* **Live Webcam Capture Button**: Toggles camera stream for live signing.
* **Pre-Loaded Sample Quick-Select Buttons**: One-click test clips (`Prompt_1.mp4`, `sample_output.mp4`, `dont_worry.mp4`).
* **Synchronized Video Player**: Video preview player with playback bar, timecode, and fullscreen option.
* **Landmark Overlay Toggle**: Switch to view 2D/3D skeleton keypoint mesh overlay directly on the video.
* **Pipeline Progress Loading Indicator**: Multi-stage processing spinner displaying real-time pipeline status (Extracting Landmarks -> Running 10 Classifier Models -> Generating SiGML XML -> Initializing Avatar).

### 2. 3D WebGL Avatar Viewport & Animation Controls
* **3D Canvas Viewport**: Embedded CWASA WebGL canvas rendering the active avatar model.
* **Avatar Character Selector**: Dropdown or character card switcher between 5 3D models (*Luna*, *Siggi*, *Anna*, *Marc*, *Francoise*).
* **Primary Playback Buttons**: Play, Replay Animation, Pause, Stop.
* **Frame-by-Frame Scrubbing Controls**: Step forward (`+1 Frame`) and step backward (`-1 Frame`) buttons for precise gesture analysis, plus current frame count (`0/0`).
* **Playback Speed Selector**: Speed toggle pills (`0.5x`, `1.0x`, `1.5x`, `2.0x`).
* **Camera / Viewport Angle Controls**: Orbit, Rotate, Zoom In/Out, and Reset Camera View.
* **Live FPS & Speed Counter**: Real-time diagnostic badge (e.g. `60.00 FPS | +1.0x`).

### 3. Phonetic Notation & Symbol Breakdown Displays
* **Phonetic Token String Box**: Displays raw HamNoSys tag tokens with one-click "Copy Tokens" button.
* **HamNoSys Native Unicode Glyph Display**: Large typography display showing native HamNoSys font symbols.
* **HamKeyboard Visual Symbol Chips**: Color-coded breakdown chips converting complex tags into clear visual badges with icons and plain English descriptions.
* **SiGML XML Code Inspector**: Collapsible code drawer showing generated SiGML XML with syntax highlighting and "Copy XML" button.

### 4. Sign Meaning, Translation & Accuracy Metrics Card
* **Recognized Sign Gloss Badge**: Prominent badge displaying translated gloss (e.g. `HELLO / HI`, `WELCOME`, `DONT WORRY`).
* **Semantic English Meaning Description**: Text explanation explaining what the sign represents.
* **Dynamic Confidence & Precision Meters**: Visual progress meters/donuts displaying overall gesture recognition confidence (e.g. `88.5%`) and precision (`85.7%`).
* **Validation & Warning Status Badge**: Visual indicator showing SiGML validation success or warnings.

### 5. Advanced Technical Diagnostics & Model Analytics (Expandable Panel)
* **10-Module Classification Matrix**: Detailed grid showing individual predictions from all 10 underlying AI modules (Handshape, Orientation, Location, Contact, Movement, etc.).
* **System Latency Monitor**: Performance Breakdown (Landmark extraction time, ML classification time, SiGML compile time).
* **Hardware Acceleration Indicator**: Hardware badge displaying active compute engine (`Intel Iris Xe iGPU DirectML` / `OpenVINO` vs `CPU`).

### 6. Future Development Roadmap Section
* **Universal Any-to-Any Sign Language Translation**: Vision card describing cross-lingual translation between regional sign languages (ISL ↔ ASL ↔ BSL).
* **Text-to-Sign & Speech-to-Sign Engine**: Direct audio/text input -> 3D Avatar.
* **Model Retraining Hub**: Interface for dataset uploads and benchmark model training.

---

## 6. Design System & Aesthetic Blueprint

To ensure the UI looks premium, state-of-the-art, and wows users:

* **Design Theme**: Modern **Dark Mode Glassmorphism** (semi-transparent glass cards `backdrop-filter: blur(16px)`, subtle gradient borders `1px solid rgba(255,255,255,0.1)`, smooth box-shadows).
* **Typography**:
  * Headings: **Outfit** (Bold, modern geometric sans-serif).
  * Body & Controls: **Inter** (High legibility UI font).
  * Code & Glyphs: Monospace & Custom **HamNoSys** font.
* **Color Palette Tokens**:
  * **Background**: Deep Slate (`#0B0F17`) to Charcoal (`#121824`).
  * **Card Surface**: Glass Dark (`rgba(18, 24, 38, 0.75)`).
  * **Primary Accents**: Neon Cyan (`#00F2FE`) & Electric Blue (`#4FACFE`).
  * **Secondary Accents**: Purple Glow (`#7F00FF`) & Pink Violet (`#E100FF`).
  * **Status Colors**: Emerald Green (`#00E676` for Success/High Confidence), Amber (`#FFD600` for Warnings), Coral (`#FF1744` for Errors).
* **Layout Grid Structure**:
  * **Header**: App title, system status badge, theme toggle.
  * **Top Main Grid (Side-by-Side 50/50)**:
    * Left Column: Input Video Selection, Upload Dropzone & Synchronized Player.
    * Right Column: 3D WebGL Avatar Viewport & Player Controls.
  * **Middle Grid**: Sign Meaning & English Translation Card + HamNoSys Symbol Chips.
  * **Bottom Accordion / Drawers**: Technical 10-Module Diagnostics, SiGML Code Inspector, and Future Roadmap.

---

## 7. Instructions for AI UI Models / Designers

If you are an AI model (e.g. Claude, ChatGPT, v0.dev, Bolt) receiving this document:
1. **Analyze the Pipeline & Features**: Use the complete context above to understand the relationship between input video processing, 10 AI modules, HamNoSys notation, and 3D WebGL avatar rendering.
2. **Propose UI Wireframes & Layouts**: Generate structured, modern UI layout suggestions (desktop & mobile responsive).
3. **Component Breakdown**: Suggest a modular component tree (e.g., `VideoInputPanel`, `AvatarViewport`, `NotationChipGrid`, `TranslationCard`, `DiagnosticsDrawer`).
4. **Code Suggestions**: Provide HTML5, CSS3 (Glassmorphism), or React/Vue code snippets implementing these visual features.
