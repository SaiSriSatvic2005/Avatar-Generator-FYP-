# V8 Hybrid Pipeline Accuracy & Evaluation Report

**Total Videos Evaluated:** 102  
**Average Processing Time:** 60.60 seconds per video  
**XML Syntactic Validity Rate:** 100.0%

---

## 1. Modular & Overall Accuracy Summary

| Module Component | Architecture / Method | Accuracy | Status |
| :--- | :--- | :--- | :--- |
| **Handshape** (Model A) | Neural (128-D Skeletal Hand-ResNet) | **57.84%** | Verified |
| **Movement** (Model B) | Neural (18-D Spatiotemporal Attention-BiGRU) | **18.63%** | Verified |
| **Palm Orientation** | Heuristic (3D Metric Cross-Product) | **41.18%** | Verified |
| **Pointing Direction** | Heuristic (Longitudinal Vector Cosine) | **30.39%** | Verified |
| **Anatomical Location** | Heuristic (3D Bounding Spatial Intervals) | **26.47%** | Verified |
| **Contact / Torso Clearance** | Heuristic (Depth Offset & Mesh Collision Gating) | **53.92%** | Verified |
| **Two-Hand Symmetry** | Heuristic (Bimanual Trajectory Correlation) | **76.47%** | Verified |
| **SiGML XML Syntax** | Grammar Linter (Ham4Parser ANTLR Compliance) | **100.00%** | 100% Avatar Valid |
| **Compound Full Sign Match** | Complete End-to-End Joint Match | **3.92%** | Full Sequence |

---

## 2. Dataset Performance Breakdown

| Dataset   |   Count | Handshape Acc   | Movement Acc   | Location Acc   | Orientation Acc   | Compound Acc   | XML Valid   |
|:----------|--------:|:----------------|:---------------|:---------------|:------------------|:---------------|:------------|
| DGS       |      52 | 76.9%           | 28.8%          | 26.9%          | 50.0%             | 5.8%           | 100.0%      |
| ISL       |      50 | 38.0%           | 8.0%           | 26.0%          | 32.0%             | 2.0%           | 100.0%      |


---

## 3. Detailed Per-Video Predictions vs. Ground Truth

| # | Video | Dataset | GT Handshape | Pred Handshape | GT Movement | Pred Movement | GT Location | Pred Location | GT Palm Ori | Pred Palm Ori | GT Ext Finger | Pred Ext Finger | GT Contact | Pred Contact | GT 2-Handed | Pred Symmetry | XML Valid | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: |
| 1 | `10_ANPASSEN1.mp4` | DGS | `hamfinger23` | `hamfinger23` | `hammoveo` | `hammoveo` | `hamshoulders` | `hamchest` | `hampalml` | `hampalml` | `hamextfingerul` | `hamextfingerul` | `True` | `False` | `True` | `hamsymmlr` | **VALID** | 46.77s |
| 2 | `11_ADRESSE1A.mp4` | DGS | `hamflathand` | `hamflathand` | `hamnomotion` | `hamnomotion` | `hamneutralspace` | `hamchest` | `hampalmd` | `hampalmd` | `hamextfingero` | `hamextfingerol` | `False` | `False` | `False` | `single` | **VALID** | 46.24s |
| 3 | `12_BEWUNDERN1.mp4` | DGS | `hampinch12` | `hamcee12` | `hamnomotion` | `hammoveo` | `hameyes` | `hamchin` | `hampalml` | `hampalml` | `hamextfingeru` | `hamextfingeru` | `True` | `False` | `True` | `hamsymmlr` | **VALID** | 44.33s |
| 4 | `13_ZUGEBEN1.mp4` | DGS | `hamfist` | `hamfist` | `hammoveo` | `hammoveo` | `hamneutralspace` | `hamneutralspace` | `hampalmd` | `hampalmd` | `hamextfingerol` | `hamextfingerol` | `False` | `False` | `True` | `hamsymmlr` | **VALID** | 46.55s |
| 5 | `14_WERBUNG1.mp4` | DGS | `hamflathand` | `hamceeall` | `hamnomotion` | `hammoveu` | `hamneutralspace` | `hamneutralspace` | `hampalml` | `hampalmul` | `hamextfingerul` | `hamextfingeruo` | `False` | `False` | `True` | `hamsymmlr` | **VALID** | 45.83s |
| 6 | `15_EMPFEHLEN1A.mp4` | DGS | `hamflathand` | `hamceeall` | `hammovedo` | `hammoveo` | `hamchin` | `hamchest` | `hampalml` | `hampalml` | `hamextfingerul` | `hamextfingerul` | `True` | `False` | `False` | `single` | **VALID** | 46.72s |
| 7 | `16_NACH1.mp4` | DGS | `hamflathand` | `hamflathand` | `hammoveo` | `hammoveo` | `hamneutralspace` | `hamchest` | `hampalmd` | `hampalmul` | `hamextfingeru` | `hamextfingeruo` | `False` | `False` | `True` | `single` | **VALID** | 45.98s |
| 8 | `17_NACHMITTAG4.mp4` | DGS | `hamfinger23spread` | `hamfinger23spread` | `hammoveo` | `hammoveo` | `hamchin` | `hamchest` | `hampalml` | `hampalml` | `hamextfingeru` | `hamextfingeruo` | `True` | `False` | `True` | `single` | **VALID** | 42.16s |
| 9 | `18_NOCHMAL1.mp4` | DGS | `hamfinger2` | `hamfinger2` | `hammovedl` | `hammoveo` | `hamneutralspace` | `hamneutralspace` | `hampalmd` | `hampalmul` | `hamextfingeruo` | `hamextfingerol` | `False` | `False` | `True` | `single` | **VALID** | 41.84s |
| 10 | `19_GEGEN1.mp4` | DGS | `hamfinger2345` | `hamfinger2345` | `hammoveo` | `hammoveo` | `hamneutralspace` | `hamchest` | `hampalmdl` | `hampalmul` | `hamextfingeru` | `hamextfingeru` | `False` | `False` | `False` | `single` | **VALID** | 52.55s |
| 11 | `1_BISSCHEN1.mp4` | DGS | `hamcee12` | `hamcee12` | `hamnomotion` | `hamnomotion` | `hamneutralspace` | `hamchest` | `hampalmdl` | `hampalml` | `hamextfingeru` | `hamextfingerul` | `False` | `False` | `False` | `single` | **VALID** | 48.56s |
| 12 | `20_ALT1A.mp4` | DGS | `hamfinger23spread` | `hamfinger23spread` | `hammoved` | `hamnomotion` | `hamcheek` | `hamchest` | `hampalml` | `hampalml` | `hamextfingeru` | `hamextfingeru` | `True` | `False` | `False` | `single` | **VALID** | 672.13s |
| 13 | `21_EINVERSTANDEN1.mp4` | DGS | `hamfinger23spread` | `hamfinger23spread` | `hammoved` | `hammoveo` | `hamforehead` | `hamchin` | `hampalmul` | `hampalml` | `hamextfingeru` | `hamextfingerul` | `True` | `False` | `True` | `hamsymmlr` | **VALID** | 39.57s |
| 14 | `22_LUFT1.mp4` | DGS | `hamflathand` | `hamflathand` | `hamnomotion` | `hammoveo` | `hamneck` | `hamchest` | `hampalml` | `hampalmul` | `hamextfingerul` | `hamextfingerul` | `False` | `False` | `True` | `hamsymmlr` | **VALID** | 36.16s |
| 15 | `23_FLUGZEUG1A.mp4` | DGS | `hamflathand` | `hamflathand` | `hamnomotion` | `hamnomotion` | `hamshoulders` | `hamchest` | `hampalmdl` | `hampalmul` | `hamextfingeruo` | `hamextfingero` | `False` | `False` | `True` | `hamsymmlr` | **VALID** | 37.31s |
| 16 | `24_ALKOHOL2.mp4` | DGS | `hamceeall` | `hamceeall` | `hamnomotion` | `hammoveo` | `hamchin` | `hamchest` | `hampalml` | `hampalml` | `hamextfingerol` | `hamextfingerul` | `True` | `False` | `False` | `single` | **VALID** | 41.93s |
| 17 | `25_LEBEN1A.mp4` | DGS | `hamfist` | `hamfist` | `hamnomotion` | `hamnomotion` | `hamneutralspace` | `hamneutralspace` | `hampalmdl` | `hampalml` | `hamextfingeruo` | `hamextfingeruo` | `False` | `False` | `True` | `hamsymmlr` | **VALID** | 42.25s |
| 18 | `26_FAST2.mp4` | DGS | `hamfinger2` | `hamfist` | `hamnomotion` | `hamnomotion` | `hamneutralspace` | `hamchest` | `hampalmdl` | `hampalml` | `hamextfingeru` | `hamextfingeruo` | `False` | `False` | `False` | `single` | **VALID** | 42.48s |
| 19 | `27_ALLEIN1A.mp4` | DGS | `hamfinger2` | `hamfinger2` | `hamnomotion` | `hamnomotion` | `hamneutralspace` | `hamneutralspace` | `hampalmu` | `hampalmu` | `hamextfingeru` | `hamextfingeru` | `False` | `False` | `False` | `single` | **VALID** | 318.73s |
| 20 | `28_SCHON1B.mp4` | DGS | `hamflathand` | `hamceeall` | `hammoved` | `hammoveo` | `hamthumbside` | `hamneutralspace` | `hampalml` | `hampalml` | `hamextfingerol` | `hamextfingerol` | `True` | `False` | `True` | `hamsymmlr` | **VALID** | 37.58s |
| 21 | `29_AUCH1A.mp4` | DGS | `hamfinger2` | `hamfinger2` | `hammovedl` | `hamnomotion` | `hamneutralspace` | `hamchest` | `hampalmd` | `hampalml` | `hamextfingeruo` | `hamextfingeru` | `False` | `False` | `True` | `single` | **VALID** | 33.57s |
| 22 | `2_AUSLAND3.mp4` | DGS | `hamflathand` | `hampinchall` | `hammoveo` | `hamnomotion` | `hamneutralspace` | `hamchin` | `hampalmd` | `hampalmul` | `hamextfingeruo` | `hamextfingeru` | `False` | `False` | `False` | `single` | **VALID** | 33.98s |
| 23 | `30_IMMER1.mp4` | DGS | `hamfinger2` | `hamfinger2` | `hamnomotion` | `hamnomotion` | `hamneutralspace` | `hamchest` | `hampalml` | `hampalmdl` | `hamextfingerl` | `hamextfingerl` | `True` | `False` | `True` | `hamsymmlr` | **VALID** | 44.24s |
| 24 | `31__RGERN1.mp4` | DGS | `hamfinger2345` | `hamfinger2345` | `hamcircleol` | `hammoveo` | `hamchest` | `hamchest` | `hampalml` | `hampalmul` | `hamextfingerl` | `hamextfingerul` | `True` | `False` | `False` | `single` | **VALID** | 66.05s |
| 25 | `32_TIER1A.mp4` | DGS | `hamfinger2345` | `hamfinger2345` | `hammoveo` | `hamnomotion` | `hamneutralspace` | `hamneutralspace` | `hampalmd` | `hampalmd` | `hamextfingero` | `hamextfingerol` | `False` | `False` | `True` | `hamsymmlr` | **VALID** | 66.26s |
| 26 | `33__RGERN1.mp4` | DGS | `hamfinger2345` | `hamfinger2345` | `hamcircleol` | `hammoveo` | `hamchest` | `hamchest` | `hampalml` | `hampalmul` | `hamextfingerl` | `hamextfingerul` | `True` | `False` | `False` | `single` | **VALID** | 44.4s |
| 27 | `34_ANTWORT1.mp4` | DGS | `hamfinger23` | `hamfinger23` | `hammoveo` | `hamnomotion` | `hamchin` | `hamchest` | `hampalml` | `hampalmdl` | `hamextfingerul` | `hamextfingerul` | `True` | `False` | `True` | `single` | **VALID** | 45.15s |
| 28 | `35_WOHNUNG1B.mp4` | DGS | `hamflathand` | `hamflathand` | `hamnomotion` | `hamnomotion` | `hamcheek` | `hamchest` | `hampalmdl` | `hampalml` | `hamextfingeru` | `hamextfingeru` | `True` | `False` | `False` | `single` | **VALID** | 45.27s |
| 29 | `36_APPETIT1.mp4` | DGS | `hampinchall` | `hamceeall` | `hammovei` | `hammoveo` | `hamlips` | `hamneutralspace` | `hampalmu` | `hampalmu` | `hamextfingeruo` | `hamextfingerdl` | `True` | `False` | `False` | `single` | **VALID** | 44.32s |
| 30 | `37_APFEL2.mp4` | DGS | `hamfinger2345` | `hamfinger2345` | `hamnomotion` | `hammoveo` | `hamchin` | `hamchest` | `hampalml` | `hampalmul` | `hamextfingerl` | `hamextfingerul` | `True` | `False` | `False` | `single` | **VALID** | 42.54s |
| 31 | `38_ANERKENNEN1.mp4` | DGS | `hamfinger2345` | `hamfinger2345` | `hammovedo` | `hammoveo` | `hamforehead` | `hamchin` | `hampalmul` | `hampalmul` | `hamextfingeru` | `hamextfingeruo` | `True` | `False` | `True` | `hamsymmlr` | **VALID** | 45.09s |
| 32 | `39_APRIL2.mp4` | DGS | `hamfinger2345` | `hamfinger2345` | `hamnomotion` | `hammoveo` | `hamnose` | `hamchest` | `hampalml` | `hampalml` | `hamextfingeru` | `hamextfingerul` | `True` | `False` | `False` | `single` | **VALID** | 37.09s |
| 33 | `3_AKZEPTIEREN1.mp4` | DGS | `hamceeall` | `hamceeall` | `hammovei` | `hammoveo` | `hamneutralspace` | `hamchest` | `hampalmd` | `hampalmd` | `hamextfingero` | `hamextfingerdl` | `False` | `False` | `True` | `hamsymmlr` | **VALID** | 42.51s |
| 34 | `40_GEGEND1.mp4` | DGS | `hamfinger2345` | `hamflathand` | `hamnomotion` | `hammoveu` | `hamneutralspace` | `hamchest` | `hampalmd` | `hampalmd` | `hamextfingerol` | `hamextfingerol` | `False` | `False` | `False` | `single` | **VALID** | 43.19s |
| 35 | `41_SESSEL2.mp4` | DGS | `hamflathand` | `hamfinger2345` | `hammoved` | `hamnomotion` | `hamneutralspace` | `hamneutralspace` | `hampalmd` | `hampalmd` | `hamextfingero` | `hamextfingerdo` | `False` | `False` | `True` | `hamsymmlr` | **VALID** | 42.03s |
| 36 | `42_ARMEE1.mp4` | DGS | `hamfist` | `hamfist` | `hammover` | `hamnomotion` | `hamchest` | `hamchest` | `hampalml` | `hampalmul` | `hamextfingerl` | `hamextfingerul` | `True` | `False` | `False` | `single` | **VALID** | 46.98s |
| 37 | `43_ABMACHEN1.mp4` | DGS | `hamflathand` | `hamflathand` | `hammoved` | `hammoveo` | `hampalm` | `hamneutralspace` | `hampalmd` | `hampalmd` | `hamextfingeruo` | `hamextfingerol` | `True` | `False` | `True` | `hamsymmlr` | **VALID** | 44.83s |
| 38 | `44_ANKOMMEN1.mp4` | DGS | `hamfinger2345` | `hamfinger2345` | `hammovedo` | `hammoveo` | `hamneutralspace` | `hamneutralspace` | `hampalmd` | `hampalmd` | `hamextfingerdo` | `hamextfingerol` | `False` | `False` | `True` | `hamsymmlr` | **VALID** | 43.84s |
| 39 | `45_KUNST4__PERSONEN-ILLUSTRATION1.mp4` | DGS | `hampinchall` | `hampinchall` | `hammoved` | `hammoveo` | `hamneutralspace` | `hamneutralspace` | `hampalmul` | `hampalmul` | `hamextfingeruo` | `hamextfingerdl` | `False` | `False` | `False` | `hamsymmlr` | **VALID** | 44.78s |
| 40 | `46_ASCHE1.mp4` | DGS | `hamfinger2345` | `hampinchall` | `hamnomotion` | `hammoveo` | `hamneutralspace` | `hamneutralspace` | `hampalmu` | `hampalmu` | `hamextfingerol` | `hamextfingeruo` | `False` | `False` | `False` | `single` | **VALID** | 45.05s |
| 41 | `47_FRAGE1.mp4` | DGS | `hampinch12open` | `hampinch12open` | `hammoveo` | `hammoveo` | `hamchin` | `hamchest` | `hampalml` | `hampalmul` | `hamextfingerul` | `hamextfingerul` | `True` | `False` | `False` | `single` | **VALID** | 44.66s |
| 42 | `48_BITTE2.mp4` | DGS | `hamflathand` | `hamflathand` | `hammovel` | `hammoveo` | `hampalm` | `hamneutralspace` | `hampalml` | `hampalml` | `hamextfingeruo` | `hamextfingeru` | `True` | `False` | `True` | `hamsymmlr` | **VALID** | 43.46s |
| 43 | `49__SPEZIAL-ANSTREBEN1.mp4` | DGS | `hamflathand` | `hamflathand` | `hamnomotion` | `hammoveo` | `hamchin` | `hamchest` | `hampalml` | `hampalmdl` | `hamextfingerul` | `hamextfingerul` | `True` | `False` | `False` | `single` | **VALID** | 47.6s |
| 44 | `4_UNFALL1B.mp4` | DGS | `hamfist` | `hamfist` | `hammovel` | `hamnomotion` | `hamneutralspace` | `hamchest` | `hampalml` | `hampalml` | `hamextfingerl` | `hamextfingerul` | `True` | `False` | `True` | `hamsymmlr` | **VALID** | 47.25s |
| 45 | `50_ENDE1A.mp4` | DGS | `hamflathand` | `hamflathand` | `hammoved` | `hamnomotion` | `hamneutralspace` | `hamchest` | `hampalml` | `hampalml` | `hamextfingerol` | `hamextfingerul` | `True` | `False` | `True` | `hamsymmlr` | **VALID** | 40.88s |
| 46 | `51_ATHEN1.mp4` | DGS | `hamfist` | `hamfist` | `hammovedl` | `hammoveo` | `hampalm` | `hamchest` | `hampalmr` | `hampalmd` | `hamextfingerl` | `hamextfingerl` | `True` | `False` | `True` | `hamsymmlr` | **VALID** | 45.22s |
| 47 | `52_AUGUST2.mp4` | DGS | `hamfist` | `hampinchall` | `hamnomotion` | `hamnomotion` | `hamneutralspace` | `hamchest` | `hampalmd` | `hampalmul` | `hamextfingeru` | `hamextfingeru` | `False` | `False` | `False` | `single` | **VALID** | 51.45s |
| 48 | `5_NEHMEN1.mp4` | DGS | `hamceeall` | `hamceeall` | `hammovei` | `hamnomotion` | `hamneutralspace` | `hamchest` | `hampalmd` | `hampalmd` | `hamextfingero` | `hamextfingerdl` | `False` | `False` | `True` | `hamsymmlr` | **VALID** | 47.98s |
| 49 | `6_SCHMERZ1.mp4` | DGS | `hamfist` | `hamfist` | `hammoved` | `hamnomotion` | `hamshoulders` | `hamchest` | `hampalmdl` | `hampalml` | `hamextfingeruo` | `hamextfingeru` | `True` | `False` | `True` | `hamsymmlr` | **VALID** | 45.96s |
| 50 | `7_SCHAFFEN1A.mp4` | DGS | `hamfinger2345` | `hamfist` | `hammovel` | `hamnomotion` | `hamthumbside` | `hamneutralspace` | `hampalml` | `hampalml` | `hamextfingerol` | `hamextfingerul` | `True` | `False` | `True` | `hamsymmlr` | **VALID** | 43.12s |
| 51 | `8_HANDLUNG1B.mp4` | DGS | `hamfist` | `hamfist` | `hammovei` | `hammoveo` | `hamneutralspace` | `hamneutralspace` | `hampalmul` | `hampalml` | `hamextfingerol` | `hamextfingerol` | `False` | `False` | `True` | `hamsymmlr` | **VALID** | 44.88s |
| 52 | `98_FLASCHE1A.mp4` | DGS | `hamceeall` | `hamceeall` | `hammoveu` | `hammoveo` | `hamthumbside` | `hamchest` | `hampalml` | `hampalmdl` | `hamextfingerol` | `hamextfingerul` | `True` | `False` | `True` | `hamsymmlr` | **VALID** | 44.82s |
| 53 | `A.mp4` | ISL | `hamfinger2` | `hamfist` | `hamnomotion` | `hammoveo` | `hamchest` | `hamchest` | `hampalmdl` | `hampalmdl` | `hamextfingerul` | `hamextfingerol` | `True` | `False` | `True` | `hamsymmlr` | **VALID** | 37.14s |
| 54 | `About.mp4` | ISL | `hamfinger2` | `hamfinger2` | `hamcircleo` | `hammoveo` | `hamchest` | `hamneutralspace` | `hampalml` | `hampalmd` | `hamextfingerl` | `hamextfingerdl` | `True` | `False` | `True` | `hamsymmlr` | **VALID** | 39.29s |
| 55 | `Above.mp4` | ISL | `hamflathand` | `hampinchall` | `hammoveu` | `hammoveo` | `hamchest` | `hamcheek` | `hampalml` | `hampalml` | `hamextfingerl` | `hamextfingerul` | `True` | `False` | `True` | `hamsymmlr` | **VALID** | 45.53s |
| 56 | `Absorb.mp4` | ISL | `hamfinger2345` | `hampinchall` | `hamnomotion` | `hammoveo` | `hamchest` | `hamchest` | `hampalmd` | `hampalmul` | `hamextfingerir` | `hamextfingeruo` | `True` | `False` | `True` | `hamsymmlr` | **VALID** | 45.12s |
| 57 | `Accept.mp4` | ISL | `hamfinger2345` | `hampinchall` | `hamnomotion` | `hammoved` | `hamchest` | `hamchest` | `hampalml` | `hampalmd` | `hamextfingerl` | `hamextfingerl` | `True` | `False` | `True` | `hamsymmlr` | **VALID** | 46.31s |
| 58 | `Accident.mp4` | ISL | `hamfist` | `hamfist` | `hamnomotion` | `hammoveo` | `hamchest` | `hamchest` | `hampalml` | `hampalml` | `hamextfingerol` | `hamextfingerol` | `False` | `False` | `True` | `hamsymmlr` | **VALID** | 41.17s |
| 59 | `Accuse.mp4` | ISL | `hamflathand` | `hamfinger2` | `hamnomotion` | `hammoveo` | `hamcheek` | `hamcheek` | `hampalml` | `hampalml` | `hamextfingeru` | `hamextfingeruo` | `False` | `False` | `True` | `single` | **VALID** | 46.79s |
| 60 | `Across.mp4` | ISL | `hamflathand` | `hamfinger2345` | `hamnomotion` | `hammoveo` | `hamwristback` | `hamchest` | `hampalml` | `hampalml` | `hamextfingerol` | `hamextfingerol` | `False` | `False` | `True` | `hamsymmlr` | **VALID** | 45.01s |
| 61 | `Active.mp4` | ISL | `hamfist` | `hamfist` | `hamcirclei` | `hammoved` | `hamshoulders` | `hamchin` | `hampalmd` | `hampalml` | `hamextfingeru` | `hamextfingerul` | `False` | `False` | `True` | `hamsymmlr` | **VALID** | 64.89s |
| 62 | `Actor.mp4` | ISL | `hamfinger2` | `hampinchall` | `hammovei` | `hammoveo` | `hamnose` | `hamchin` | `hampalmdl` | `hampalml` | `hamextfingerul` | `hamextfingerul` | `True` | `False` | `True` | `hamsymmlr` | **VALID** | 70.07s |
| 63 | `Actress.mp4` | ISL | `hamfinger2` | `hamfinger2` | `hammovei` | `hammoved` | `hamnose` | `hamchin` | `hampalmdl` | `hampalmdl` | `hamextfingerul` | `hamextfingerul` | `True` | `False` | `True` | `hamsymmlr` | **VALID** | 50.58s |
| 64 | `Advise.mp4` | ISL | `hamflathand` | `hamfinger2345` | `hammoveu` | `hammoveo` | `hamchest` | `hamcheek` | `hampalmul` | `hampalmul` | `hamextfingerl` | `hamextfingerol` | `False` | `False` | `True` | `single` | **VALID** | 46.2s |
| 65 | `Aeroplane.mp4` | ISL | `hamfinger2345` | `hamflathand` | `hammoveu` | `hammoveo` | `hamstomach` | `hamnose` | `hampalmd` | `hampalmdl` | `hamextfingero` | `hamextfingerul` | `False` | `False` | `True` | `single` | **VALID** | 48.01s |
| 66 | `Africa.mp4` | ISL | `hamflathand` | `hamceeall` | `hammoved` | `hammoveo` | `hamcheek` | `hamshoulders` | `hampalmd` | `hampalmu` | `hamextfingeru` | `hamextfingeru` | `False` | `False` | `True` | `single` | **VALID** | 47.8s |
| 67 | `Afternoon.mp4` | ISL | `hamfinger2345` | `hamfinger23` | `hamnomotion` | `hammoveo` | `hamheadtop` | `hamchin` | `hampalmd` | `hampalmul` | `hamextfingerul` | `hamextfingeru` | `False` | `False` | `True` | `single` | **VALID** | 47.17s |
| 68 | `Age.mp4` | ISL | `hamfist` | `hamceeall` | `hamnomotion` | `hammoveo` | `hamneutralspace` | `hamcheek` | `hampalml` | `hampalmdl` | `hamextfingerl` | `hamextfingerul` | `False` | `False` | `True` | `single` | **VALID** | 45.05s |
| 69 | `Agree.mp4` | ISL | `hamfist` | `hamfinger2` | `hammovel` | `hammovel` | `hamneutralspace` | `hamchest` | `hampalml` | `hampalmd` | `hamextfingero` | `hamextfingero` | `False` | `False` | `True` | `hamsymmlr` | **VALID** | 45.72s |
| 70 | `Alive.mp4` | ISL | `hamflathand` | `hamfinger2345` | `hamnomotion` | `hammoveo` | `hamchin` | `hamchest` | `hampalmdl` | `hampalml` | `hamextfingerul` | `hamextfingeruo` | `False` | `False` | `True` | `hamsymmlr` | **VALID** | 43.89s |
| 71 | `All.mp4` | ISL | `hamfinger2` | `hamfinger2` | `hamcircleol` | `hammoveo` | `hamchest` | `hamchest` | `hampalmdl` | `hampalmd` | `hamextfingeru` | `hamextfingerol` | `False` | `False` | `True` | `single` | **VALID** | 410.81s |
| 72 | `Allah.mp4` | ISL | `hamflathand` | `hamflathand` | `hamnomotion` | `hammoved` | `hamshoulders` | `hamcheek` | `hampalmdl` | `hampalmr` | `hamextfingeru` | `hamextfingerui` | `True` | `False` | `True` | `hamsymmlr` | **VALID** | 40.84s |
| 73 | `Allow.mp4` | ISL | `hamfinger2` | `hamfinger2` | `hamnomotion` | `hammoveo` | `hamlips` | `hamchest` | `hampalmdl` | `hampalmr` | `hamextfingerul` | `hamextfingerul` | `True` | `False` | `True` | `single` | **VALID** | 34.58s |
| 74 | `Alone.mp4` | ISL | `hamfinger2` | `hamfinger2` | `hammoveo` | `hammoveo` | `hamshouldertop` | `hamchest` | `hampalmu` | `hampalmu` | `hamextfingeru` | `hamextfingeruo` | `False` | `False` | `True` | `single` | **VALID** | 37.38s |
| 75 | `Always.mp4` | ISL | `hamfinger2` | `hamfinger2` | `hamcircleo` | `hammoveo` | `hamchest` | `hamchest` | `hampalml` | `hampalmdl` | `hamextfingerl` | `hamextfingerl` | `False` | `False` | `True` | `single` | **VALID** | 39.15s |
| 76 | `Ambulance.mp4` | ISL | `hamflathand` | `hamfinger2345` | `hammoveo` | `hammoveuo` | `hamwristback` | `hamchest` | `hampalmd` | `hampalml` | `hamextfingerol` | `hamextfingerul` | `True` | `False` | `True` | `hamsymmlr` | **VALID** | 46.76s |
| 77 | `Among.mp4` | ISL | `hamfinger2` | `hamfinger23spread` | `hamcircleo` | `hammoveo` | `hamneutralspace` | `hamchest` | `hampalmd` | `hampalml` | `hamextfingerd` | `hamextfingeruo` | `False` | `False` | `True` | `hamsymmlr` | **VALID** | 67.28s |
| 78 | `Andhra Pradesh.mp4` | ISL | `hamflathand` | `hamcee12` | `hamnomotion` | `hammoveo` | `hamneutralspace` | `hamchest` | `hampalmd` | `hampalmdl` | `hamextfingeru` | `hamextfingerul` | `False` | `False` | `True` | `hamsymmlr` | **VALID** | 46.98s |
| 79 | `Angel.mp4` | ISL | `hamflathand` | `hampinchall` | `hammover` | `hammoveo` | `hamshouldertop` | `hamcheek` | `hampalml` | `hampalmdr` | `hamextfingeru` | `hamextfingeru` | `False` | `False` | `True` | `hamsymmlr` | **VALID** | 43.29s |
| 80 | `Angry.mp4` | ISL | `hamfinger2345` | `hampinch12open` | `hammover` | `hammoveo` | `hamear` | `hamcheek` | `hampalml` | `hampalml` | `hamextfingeru` | `hamextfingerui` | `True` | `False` | `True` | `single` | **VALID** | 42.9s |
| 81 | `Announce.mp4` | ISL | `hampinchall` | `hamfinger2345` | `hamnomotion` | `hammoveo` | `hamchin` | `hamchin` | `hampalml` | `hampalmdr` | `hamextfingeru` | `hamextfingeruo` | `True` | `False` | `True` | `hamsymmlr` | **VALID** | 43.26s |
| 82 | `Any.mp4` | ISL | `hamflathand` | `hamfinger23spread` | `hammoveo` | `hammovel` | `hamshoulders` | `hamchest` | `hampalmu` | `hampalmu` | `hamextfingerl` | `hamextfingeru` | `False` | `True` | `True` | `single` | **VALID** | 43.5s |
| 83 | `Appear.mp4` | ISL | `hamfinger2` | `hamfinger2` | `hammovel` | `hammoveo` | `hamshouldertop` | `hamchest` | `hampalmu` | `hampalmu` | `hamextfingeru` | `hamextfingeruo` | `False` | `False` | `True` | `single` | **VALID** | 44.63s |
| 84 | `Apple.mp4` | ISL | `hamfinger2` | `hamfinger23spread` | `hamswinging` | `hammoveo` | `hamchin` | `hamchin` | `hampalml` | `hampalmul` | `hamextfingerl` | `hamextfingerul` | `False` | `False` | `True` | `single` | **VALID** | 43.35s |
| 85 | `April.mp4` | ISL | `hamfinger2` | `hampinchall` | `hamnomotion` | `hammoved` | `hamstomach` | `hamcheek` | `hampalmdl` | `hampalmd` | `hamextfingerul` | `hamextfingerl` | `False` | `False` | `True` | `hamsymmlr` | **VALID** | 43.63s |
| 86 | `Area.mp4` | ISL | `hamfinger2345` | `hamfinger2` | `hamnomotion` | `hamnomotion` | `hamchest` | `hamchest` | `hampalmd` | `hampalmdl` | `hamextfingerol` | `hamextfingeruo` | `False` | `False` | `True` | `hamsymmlr` | **VALID** | 43.52s |
| 87 | `Around.mp4` | ISL | `hamfinger2345` | `hamfinger2` | `hamcirclei` | `hamnomotion` | `hamchest` | `hamchest` | `hampalmd` | `hampalmd` | `hamextfingeru` | `hamextfingerol` | `False` | `False` | `True` | `hamsymmlr` | **VALID** | 162.04s |
| 88 | `Arrange.mp4` | ISL | `hamflathand` | `hamcee12` | `hamnomotion` | `hammoveo` | `hamneutralspace` | `hamchest` | `hampalmd` | `hampalmd` | `hamextfingeru` | `hamextfingero` | `False` | `False` | `True` | `hamsymmlr` | **VALID** | 49.28s |
| 89 | `Arrest.mp4` | ISL | `hamfist` | `hamfist` | `hammoveo` | `hamnomotion` | `hamwristback` | `hamcheek` | `hampalmu` | `hampalmul` | `hamextfingerol` | `hamextfingeru` | `True` | `False` | `True` | `hamsymmlr` | **VALID** | 43.9s |
| 90 | `Arrive.mp4` | ISL | `hamfinger2` | `hamfist` | `hamnomotion` | `hammoveo` | `hamshouldertop` | `hamchest` | `hampalmur` | `hampalml` | `hamextfingerui` | `hamextfingerol` | `False` | `False` | `True` | `hamsymmlr` | **VALID** | 43.26s |
| 91 | `Art.mp4` | ISL | `hamfinger23` | `hamfinger23` | `hammoved` | `hammoveo` | `hamforehead` | `hamcheek` | `hampalmdl` | `hampalmdl` | `hamextfingerul` | `hamextfingerul` | `False` | `False` | `True` | `single` | **VALID** | 43.05s |
| 92 | `Asia.mp4` | ISL | `hamflathand` | `hamfist` | `hamcirclei` | `hammoved` | `hamshouldertop` | `hamchest` | `hampalml` | `hampalmdl` | `hamextfingerul` | `hamextfingerol` | `True` | `False` | `True` | `hamsymmlr` | **VALID** | 46.8s |
| 93 | `Assam.mp4` | ISL | `hamfinger2345` | `hamfist` | `hamnomotion` | `hammoved` | `hamshouldertop` | `hamchest` | `hampalmd` | `hampalmdl` | `hamextfingeru` | `hamextfingeruo` | `False` | `False` | `True` | `hamsymmlr` | **VALID** | 261.66s |
| 94 | `At.mp4` | ISL | `hamfinger2` | `hamfinger2` | `hamnomotion` | `hammoveo` | `hamneutralspace` | `hamchest` | `hampalml` | `hampalmdl` | `hamextfingerl` | `hamextfingerl` | `True` | `False` | `True` | `hamsymmlr` | **VALID** | 35.58s |
| 95 | `Audiologist.mp4` | ISL | `hamflathand` | `hamceeall` | `hamcircleo` | `hammoved` | `hamear` | `hamchin` | `hampalml` | `hampalml` | `hamextfingerl` | `hamextfingerui` | `True` | `False` | `True` | `hamsymmlr` | **VALID** | 39.43s |
| 96 | `Australia.mp4` | ISL | `hamflathand` | `hampinchall` | `hamnomotion` | `hammoveo` | `hamforehead` | `hamchest` | `hampalmul` | `hampalmdl` | `hamextfingeru` | `hamextfingeruo` | `False` | `False` | `True` | `hamsymmlr` | **VALID** | 38.14s |
| 97 | `Autorickshaw.mp4` | ISL | `hamfist` | `hampinchall` | `hamnomotion` | `hammoved` | `hambelowstomach` | `hamchest` | `hampalmul` | `hampalmdl` | `hamextfingerdl` | `hamextfingerol` | `False` | `False` | `True` | `hamsymmlr` | **VALID** | 40.37s |
| 98 | `Axe.mp4` | ISL | `hamflathand` | `hamfinger2345` | `hammoved` | `hammoved` | `hamshoulders` | `hamneutralspace` | `hampalmul` | `hampalmu` | `hamextfingeruo` | `hamextfingerul` | `True` | `False` | `True` | `hamsymmlr` | **VALID** | 43.98s |
| 99 | `B.mp4` | ISL | `hamindexfinger` | `hampinchall` | `hamnomotion` | `hammoveo` | `hamchest` | `hamchest` | `hampalml` | `hampalml` | `hamextfingeru` | `hamextfingerul` | `True` | `False` | `True` | `hamsymmlr` | **VALID** | 44.57s |
| 100 | `Bad.mp4` | ISL | `hamfinger2` | `hampinchall` | `hamnomotion` | `hammoveo` | `hamshouldertop` | `hamchest` | `hampalmdl` | `hampalmd` | `hamextfingeru` | `hamextfingero` | `False` | `False` | `True` | `single` | **VALID** | 45.07s |
| 101 | `Badminton.mp4` | ISL | `hamfinger2` | `hampinch12` | `hammoved` | `hamnomotion` | `hamshouldertop` | `hamforehead` | `hampalml` | `hampalmul` | `hamextfingeru` | `hamextfingerul` | `False` | `False` | `True` | `single` | **VALID** | 46.58s |
| 102 | `Bag.mp4` | ISL | `hamfist` | `hamfist` | `hammoveu` | `hammoveo` | `hamshoulders` | `hamshoulders` | `hampalmd` | `hampalmdl` | `hamextfingerol` | `hamextfingerdo` | `False` | `False` | `True` | `single` | **VALID** | 45.82s |