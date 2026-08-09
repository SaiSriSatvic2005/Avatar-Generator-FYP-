#!/usr/bin/env python
# coding: utf-8
"""
HamNoSys Integration Pipeline — Local Version
Runs all 10 modules on a sign language video and produces a HamNoSys code string.
"""

import os
import sys
import json
from collections import Counter
import numpy as np
import cv2
import mediapipe as mp


from temporal_utils import smooth_frame_sequence, build_structured_sign_descriptor

# Ensure the Integration directory is on the path for local imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from Handshape_Model import run_handshape_module
from ori_model2 import run_orientation_module
from upper_body_locations_video import run_upper_body_location_module
from Head_and_face_location import run_head_face_location_module
from hand_location_video_P import run_hand_location_module
from fing_locations_d import run_finger_location_module
from contact_types_e import run_contact_type_module
from Arm_and_Space_positions import run_arm_space_module
from movement1_prava import run_movement1_module
from Movement_2 import run_movement2_module


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def extract_labels(module_output):
    """
    Extract useful HamNoSys labels from module outputs, ensuring
    all space-separated tags are split into individual tokens.
    """
    if isinstance(module_output, dict):
        if "per_frame" in module_output:
            labels = module_output["per_frame"]
            cleaned = []
            for item in labels:
                if item is None: continue
                if isinstance(item, tuple):
                    for x in item:
                        if x is not None:
                            cleaned.extend(str(x).split())
                else:
                    cleaned.extend(str(item).split())
            return cleaned

    if isinstance(module_output, str):
        return module_output.split()

    if isinstance(module_output, list):
        cleaned = []
        for item in module_output:
            if item is not None:
                cleaned.extend(str(item).split())
        return cleaned

    return []


def combine_hamnosys(handshape, orientation, upper_body, head_face, hand_location,
                    finger_location, contact, arm_space, movement1, movement2, video_path=None):

    """
    Combine module outputs into a strict, grammatically valid SiGML HamNoSys sequence.
    
    Formal HamNoSys CFG Rules:
    Sign ::= HandshapeStructure InitialOrientation BodyLocation [Contact] [Movement] [StateTransition]
    - InitialOrientation ::= ExtendedFinger PalmOrientation
    - BodyLocation ::= [HandPart/FingerPart] BaseLocation
    - StateTransition ::= hamreplace FinalExtendedFinger FinalPalmOrientation
    """

    def pick_token(module_output):
        if isinstance(module_output, dict):
            final = module_output.get("final")
            if final and final not in ["none", "no-contact"]:
                if isinstance(final, (tuple, list)):
                    return " ".join([str(x) for x in final if x and str(x) not in ["none", "no-contact"]])
                return str(final)
            per = module_output.get("per_frame", [])
            flat = []
            for item in per:
                if isinstance(item, tuple):
                    flat.extend([str(x) for x in item if x and str(x) not in ["none", "no-contact"]])
                elif item and str(item) not in ["none", "no-contact"]:
                    flat.append(str(item))
            if not flat:
                return None

            most_common, _ = Counter(flat).most_common(1)[0]
            return str(most_common)
        if isinstance(module_output, (tuple, list)):
            clean = [str(x) for x in module_output if x and str(x) not in ["none", "no-contact"]]
            return " ".join(clean) if clean else None
        if isinstance(module_output, str) and module_output not in ["none", "no-contact"]:
            return module_output
        return None

    def extract_state_transitions(module_output):
        if not isinstance(module_output, dict) or "per_frame" not in module_output:
            tok = pick_token(module_output)
            return tok, tok
            
        frames = module_output["per_frame"]
        valid_frames = []
        for item in frames:
            if isinstance(item, tuple):
                valid_frames.append(tuple(x for x in item if x and x not in ["none", "no-contact"]))
            elif item and item not in ["none", "no-contact"]:
                valid_frames.append((item,))
                
        if len(valid_frames) < 5:
            tok = pick_token(module_output)
            return tok, tok
            
        chunk_size = max(1, int(len(valid_frames) * 0.3))
        start_chunk = valid_frames[:chunk_size]
        end_chunk = valid_frames[-chunk_size:]
        

        start_tok = Counter(start_chunk).most_common(1)[0][0] if start_chunk else None
        end_tok = Counter(end_chunk).most_common(1)[0][0] if end_chunk else None
        
        if start_tok is None: start_tok = pick_token(module_output)
        if end_tok is None: end_tok = pick_token(module_output)
        
        return start_tok, end_tok

    def format_ori(ori_tuple):
        seq = []
        if isinstance(ori_tuple, tuple):
            for part in ori_tuple:
                if part and part not in ["none", "no-contact"] and (part.startswith("hamextfinger") or part.startswith("hampalm")):
                    seq.append(part)
        elif isinstance(ori_tuple, str) and (ori_tuple.startswith("hamextfinger") or ori_tuple.startswith("hampalm")):
            seq.append(ori_tuple)
        return seq

    VALID_HANDSHAPES = {
        "hamfist", "hamflathand", "hamfinger2", "hamfinger23", "hamfinger23spread",
        "hamfinger2345", "hampinch12", "hampinchall", "hampinch12open", "hamcee12",
        "hamceeall", "hamceeopen", "hamthumboutmod", "hamthumbacrossmod",
        "hamthumbopenmod", "hamfingerstraightmod", "hamfingerbendmod",
        "hamfingerhookmod", "hamdoublebent", "hamdoublehooked"
    }

    PRIMARY_BODY_LOCATIONS = {
        "hamhead", "hamheadtop", "hamforehead", "hameyebrows", "hameyes",
        "hamnose", "hamnostrils", "hamear", "hamearlobe", "hamcheek",
        "hamlips", "hamtongue", "hamteeth", "hamchin", "hamunderchin",
        "hamneck", "hamshouldertop", "hamshoulders", "hamchest", "hamstomach",
        "hambelowstomach", "hamneutralspace", "hamupperarm", "hamelbow",
        "hamlowerarm"
    }

    def pick_location_token(*module_outputs, default="hamchest"):
        """Prefer valid HtoG primary body locations, mapping modifiers to standard locations."""
        for module_output in module_outputs:
            candidate = pick_token(module_output)
            if not candidate:
                continue

            for token in str(candidate).split():
                if token in PRIMARY_BODY_LOCATIONS:
                    return token
                elif token in ["hamarmextended", "hamlrbeside", "hamlrat"]:
                    return "hamneutralspace"
                elif token in ["hampalm", "hamhandback", "hamwristback"]:
                    return "hamchest"

        return default


    def get_phase_token(module_output, start_pct=0.0, end_pct=0.5):
        if not isinstance(module_output, dict) or "per_frame" not in module_output:
            return pick_token(module_output)
        frames = module_output["per_frame"]
        if not frames:
            return pick_token(module_output)
        n = len(frames)
        start_idx = int(n * start_pct)
        end_idx = max(start_idx + 1, int(n * end_pct))
        sub_frames = frames[start_idx:end_idx]

        # Filter out 'none' and 'no-contact'
        valid_items = [x for x in sub_frames if x and str(x) not in ["none", "no-contact"]]
        if not valid_items:
            return pick_token(module_output)
        return Counter(valid_items).most_common(1)[0][0]

    sequence = []

    # Phase 1 Extraction (15% to 50% of clip)
    raw_hs_p1 = get_phase_token(handshape, 0.15, 0.50) or "hamfinger2"
    hs_p1_tokens = [t for t in str(raw_hs_p1).split() if t in VALID_HANDSHAPES]
    hs_token_p1 = hs_p1_tokens[0] if hs_p1_tokens else "hamfinger2"

    ori_p1 = format_ori(get_phase_token(orientation, 0.15, 0.50))
    ext_finger_p1 = next((x for x in ori_p1 if x.startswith("hamextfinger")), "hamextfingeru")
    palm_ori_p1 = next((x for x in ori_p1 if x.startswith("hampalm")), "hampalml")


    # Phase 1 Location Selection (Prefer head/face if active)
    head_loc_p1 = get_phase_token(head_face, 0.0, 0.45)
    if head_loc_p1 and head_loc_p1 in PRIMARY_BODY_LOCATIONS and head_loc_p1 != "hamchest":
        loc_token_p1 = head_loc_p1
    else:
        loc_token_p1 = pick_location_token(
            get_phase_token(arm_space, 0.0, 0.45),
            get_phase_token(hand_location, 0.0, 0.45),
            get_phase_token(finger_location, 0.0, 0.45),
            get_phase_token(upper_body, 0.0, 0.45),
            get_phase_token(head_face, 0.0, 0.45),
            default="hamchest",
        )

    # Phase 2 Extraction (55% to 100% of clip)
    raw_hs_p2 = get_phase_token(handshape, 0.55, 1.0) or "hamflathand"
    hs_p2_tokens = [t for t in str(raw_hs_p2).split() if t in VALID_HANDSHAPES]
    hs_token_p2 = hs_p2_tokens[0] if hs_p2_tokens else "hamflathand"

    ori_p2 = format_ori(get_phase_token(orientation, 0.55, 1.0))
    ext_finger_p2 = next((x for x in ori_p2 if x.startswith("hamextfinger")), "hamextfingeru")
    palm_ori_p2 = next((x for x in ori_p2 if x.startswith("hampalm")), "hampalmd")

    loc_token_p2 = pick_location_token(
        get_phase_token(arm_space, 0.55, 1.0),
        get_phase_token(hand_location, 0.55, 1.0),
        get_phase_token(finger_location, 0.55, 1.0),
        get_phase_token(upper_body, 0.55, 1.0),
        get_phase_token(head_face, 0.55, 1.0),
        default="hamchest",
    )

    # Sanitize finger direction at chest/neutralspace to prevent IK joint limit EPERM errors
    if loc_token_p1 in ["hamchest", "hamneutralspace"] and ext_finger_p1 == "hamextfingerd":
        ext_finger_p1 = "hamextfingerl"
    if loc_token_p2 in ["hamchest", "hamneutralspace"] and ext_finger_p2 == "hamextfingerd":
        ext_finger_p2 = "hamextfingerl"

    contact_p1 = get_phase_token(contact, 0.0, 0.45)
    contact_p2 = get_phase_token(contact, 0.55, 1.0)

    # Determine if gesture is genuinely multi-phase (handshape or location change)
    is_multi_phase = (hs_token_p1 != hs_token_p2) or (loc_token_p1 != loc_token_p2 and loc_token_p2 not in ["hamchest", "hamneutralspace"])

    # If posture is single-phase, use the predominant palm orientation over the full clip
    if not is_multi_phase:
        full_ori = format_ori(pick_token(orientation))
        palm_ori_p1 = next((x for x in full_ori if x.startswith("hampalm")), palm_ori_p1)

    # Construct Phase 1 Sequence (Contact MUST immediately follow valid body location, NOT neutralspace)
    sequence.extend([hs_token_p1, ext_finger_p1, palm_ori_p1, loc_token_p1])
    if contact_p1 and str(contact_p1) in ["hamtouch", "hamclose", "hambrushing"] and loc_token_p1 != "hamneutralspace":
        sequence.append(str(contact_p1))

    # Movement 1
    movement_token = pick_token(movement1) or pick_token(movement2)
    if movement_token and movement_token != "none":
        mov_tokens = str(movement_token).split()
        for token in mov_tokens:
            if token.startswith("hammove") or token.startswith("hamcircle") or token.startswith("hamarc") or token in ["hamnodding", "hamtwisting"]:
                sequence.append(token)
                break

    # Phase-level Dual Hand Activity Detection
    p1_dual_ratio = 0.0
    p2_dual_ratio = 0.0
    try:
        from shared_landmarks import get_video_landmarks
        frames = get_video_landmarks(video_path)
        if frames:
            n = len(frames)
            p1_frames = frames[:int(n * 0.5)]
            p2_frames = frames[int(n * 0.5):]

            import numpy as np
            def is_dual_active(f):
                l = f.get("left_hand")
                r = f.get("right_hand")
                if not l or not r:
                    return False
                lw = np.array([l.landmark[0].x, l.landmark[0].y, l.landmark[0].z])
                rw = np.array([r.landmark[0].x, r.landmark[0].y, r.landmark[0].z])
                if np.linalg.norm(lw - rw) < 0.08:
                    return False
                return l.landmark[0].y < 0.65

            p1_active = sum(1 for f in p1_frames if is_dual_active(f))
            p2_active = sum(1 for f in p2_frames if is_dual_active(f))

            p1_dual_ratio = p1_active / max(1, len(p1_frames))
            p2_dual_ratio = p2_active / max(1, len(p2_frames))

    except Exception:
        pass

    # Dual Hand Asymmetrical (hamplus) vs Symmetrical (hamsymmlr) Detection
    is_dual = (p1_dual_ratio >= 0.20 or p2_dual_ratio >= 0.20)
    
    if is_dual:
        try:
            from shared_landmarks import get_video_landmarks
            from Handshape_Model import classify_handshape
            from ori_model2 import calculate_orientation_from_landmarks
            
            frames = get_video_landmarks(video_path)
            lh_shapes, lh_fingers, lh_palms = [], [], []
            for f in frames:
                if f.get("left_hand"):
                    lm = f["left_hand"].landmark
                    lh_shapes.append(classify_handshape(lm))
                    _, finger_l, palm_l = calculate_orientation_from_landmarks(lm, "Left")
                    lh_fingers.append(finger_l)
                    lh_palms.append(palm_l)
                    

            if lh_shapes:
                lh_hs = Counter(lh_shapes).most_common(1)[0][0]
                lh_ext = Counter(lh_fingers).most_common(1)[0][0]
                lh_palm = Counter(lh_palms).most_common(1)[0][0]
                
                # If Left hand posture/orientation differs from Right hand, build asymmetrical hamplus sequence
                if (lh_hs != hs_token_p1) or (lh_palm != palm_ori_p1):
                    sequence = [
                        hs_token_p1, ext_finger_p1, palm_ori_p1,
                        "hamplus",
                        lh_hs, lh_ext, lh_palm,
                        loc_token_p1
                    ]
                    if contact_p1 and str(contact_p1) in ["hamtouch", "hamclose", "hambrushing"] and loc_token_p1 != "hamneutralspace":
                        sequence.append(str(contact_p1))
                    if movement_token and movement_token != "none":
                        for token in str(movement_token).split():
                            if token.startswith("hammove") or token.startswith("hamcircle") or token.startswith("hamarc"):
                                sequence.append(token)
                                break
                    return " ".join(sequence)
        except Exception:
            pass

        if "hamsymmlr" not in sequence:
            sequence.insert(0, "hamsymmlr")

    if is_multi_phase:
        sequence.append("hamreplace")
        sequence.append(hs_token_p2)
        if ext_finger_p2: sequence.append(ext_finger_p2)
        if palm_ori_p2: sequence.append(palm_ori_p2)
        if loc_token_p2 and loc_token_p2 != loc_token_p1:
            sequence.append(loc_token_p2)
        if contact_p2 and str(contact_p2) in ["hamtouch", "hamclose", "hambrushing"] and loc_token_p2 != "hamneutralspace":
            sequence.append(str(contact_p2))

    return " ".join(sequence)



    return " ".join(sequence)



    return " ".join(sequence)





# =====================================================
# MAIN PIPELINE
# =====================================================

def generate_hamnosys(video_path):
    """
    Run ALL 10 modules on a video and combine results into a HamNoSys string.
    """

    print("\nRunning modules...")

    handshape       = run_handshape_module(video_path)
    print("  [1/10] Handshape - done")

    orientation     = run_orientation_module(video_path)
    print("  [2/10] Orientation - done")

    upper_body      = run_upper_body_location_module(video_path)
    print("  [3/10] Upper Body - done")

    head_face       = run_head_face_location_module(video_path)
    print("  [4/10] Head & Face - done")

    hand_location   = run_hand_location_module(video_path)
    print("  [5/10] Hand Location - done")

    finger_location = run_finger_location_module(video_path)
    print("  [6/10] Finger Location - done")

    contact         = run_contact_type_module(video_path)
    print("  [7/10] Contact Type - done")

    arm_space       = run_arm_space_module(video_path)
    print("  [8/10] Arm & Space - done")

    movement1       = run_movement1_module(video_path)
    print("  [9/10] Movement 1 - done")

    movement2       = run_movement2_module(video_path)
    print("  [10/10] Movement 2 - done")

    print("\n===== MODULE OUTPUTS =====")
    print("Handshape       :", handshape)
    print("Orientation     :", orientation)
    print("Arm & Space     :", arm_space)
    print("Upper Body      :", upper_body)
    print("Head & Face     :", head_face)
    print("Hand Location   :", hand_location)
    print("Finger Location :", finger_location)
    print("Contact Type    :", contact)
    print("Movement 1      :", movement1)
    print("Movement 2      :", movement2)

    hamnosys_code = combine_hamnosys(
        handshape,
        orientation,
        upper_body,
        head_face,
        hand_location,
        finger_location,
        contact,
        arm_space,
        movement1,
        movement2,
        video_path=video_path
    )


    print("\n========== FINAL HAMNOSYS ==========")
    print(hamnosys_code)

    smoothed_modules = {}
    for key, value in {
        "handshape": handshape,
        "orientation": orientation,
        "upper_body": upper_body,
        "head_face": head_face,
        "hand_location": hand_location,
        "finger_location": finger_location,
        "contact": contact,
        "arm_space": arm_space,
        "movement1": movement1,
        "movement2": movement2,
    }.items():
        if isinstance(value, dict):
            per_frame = value.get("per_frame", [])
            if per_frame:
                smoothed_modules[key] = {"per_frame": smooth_frame_sequence(per_frame, window=3)}
            else:
                smoothed_modules[key] = value
        else:
            smoothed_modules[key] = value

    descriptor = build_structured_sign_descriptor({
        "handshape": smoothed_modules.get("handshape", handshape),
        "orientation": smoothed_modules.get("orientation", orientation),
        "location": smoothed_modules.get("upper_body", upper_body),
        "movement1": smoothed_modules.get("movement1", movement1),
        "contact": smoothed_modules.get("contact", contact),
    })

    return hamnosys_code, {
        "handshape": smoothed_modules.get("handshape", handshape),
        "orientation": smoothed_modules.get("orientation", orientation),
        "upper_body": smoothed_modules.get("upper_body", upper_body),
        "head_face": smoothed_modules.get("head_face", head_face),
        "hand_location": smoothed_modules.get("hand_location", hand_location),
        "finger_location": smoothed_modules.get("finger_location", finger_location),
        "contact": smoothed_modules.get("contact", contact),
        "arm_space": smoothed_modules.get("arm_space", arm_space),
        "movement1": smoothed_modules.get("movement1", movement1),
        "movement2": smoothed_modules.get("movement2", movement2),
        "descriptor": descriptor,
    }


# =====================================================
# ANNOTATED OUTPUT VIDEO
# =====================================================

def safe(x):
    if x is None:
        return "none"
    return str(x)


def get_frames(output, total_frames):

    if isinstance(output, dict):
        arr = output.get("per_frame", [])
        if len(arr) > 0:
            return arr

    if isinstance(output, str):
        return [output] * total_frames

    if isinstance(output, list):
        return output

    return ["none"] * total_frames


def create_annotated_video(video_path, output_path, modules):
    """
    Create output video with per-frame labels overlaid.
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("ERROR: Cannot open video for annotation")
        return

    ret, frame = cap.read()
    if not ret:
        print("ERROR: Cannot read first frame")
        cap.release()
        return

    h, w, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 25, (w, h))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Extract per-frame data
    handshape_frames    = get_frames(modules["handshape"], total_frames)
    orientation_frames  = get_frames(modules["orientation"], total_frames)
    upper_frames        = get_frames(modules["upper_body"], total_frames)
    head_frames         = get_frames(modules["head_face"], total_frames)
    hand_frames         = get_frames(modules["hand_location"], total_frames)
    finger_frames       = get_frames(modules["finger_location"], total_frames)
    contact_frames      = get_frames(modules["contact"], total_frames)
    arm_frames          = get_frames(modules["arm_space"], total_frames)
    movement1_frames    = get_frames(modules["movement1"], total_frames)
    movement2_frames    = get_frames(modules["movement2"], total_frames)

    frame_id = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Safe index access
        idx = min(frame_id, len(handshape_frames) - 1) if handshape_frames else 0

        handshape_label = safe(handshape_frames[idx]) if handshape_frames else "none"
        orientation_label = safe(orientation_frames[min(frame_id, len(orientation_frames) - 1)]) if orientation_frames else "none"

        location_label = " ".join([
            safe(upper_frames[min(frame_id, len(upper_frames) - 1)]) if upper_frames else "none",
            safe(head_frames[min(frame_id, len(head_frames) - 1)]) if head_frames else "none",
            safe(hand_frames[min(frame_id, len(hand_frames) - 1)]) if hand_frames else "none",
            safe(finger_frames[min(frame_id, len(finger_frames) - 1)]) if finger_frames else "none",
            safe(contact_frames[min(frame_id, len(contact_frames) - 1)]) if contact_frames else "none",
            safe(arm_frames[min(frame_id, len(arm_frames) - 1)]) if arm_frames else "none",
        ])

        movement_label = " ".join([
            safe(movement1_frames[min(frame_id, len(movement1_frames) - 1)]) if movement1_frames else "none",
            safe(movement2_frames[min(frame_id, len(movement2_frames) - 1)]) if movement2_frames else "none",
        ])

        cv2.putText(frame, f"Handshape: {handshape_label}",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, f"Orientation: {orientation_label}",
                    (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, f"Location: {location_label}",
                    (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, f"Movement: {movement_label}",
                    (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()

    print(f"Annotated video saved: {output_path}")


# =====================================================
# PROCESS VIDEO (full pipeline + annotated output)
# =====================================================

def process_video_neural(video_path):
    """
    Neural Pipeline: Video -> Normalized Features -> Trained Model -> HamNoSys
    Zero manual thresholds, zero dictionary lookups, zero gloss names required at runtime.
    """
    model_path = os.path.join(_SCRIPT_DIR, "wlasl_landmark_dataset", "hamnosys_net_v2.pth")
    map_path = os.path.join(_SCRIPT_DIR, "wlasl_landmark_dataset", "nn_class_mappings.json")

    if not os.path.exists(model_path) or not os.path.exists(map_path):
        print("[NeuralPipeline] Trained model weights not found — falling back to heuristic pipeline...")
        return generate_hamnosys(video_path)[0]

    try:
        import torch
        from train_landmark_nn import HamNoSysSequenceNet
        from build_wlasl_landmarks import extract_normalized_features
        from shared_landmarks import get_video_landmarks

        with open(map_path, "r", encoding="utf-8") as f:
            mappings = json.load(f)

        frames = get_video_landmarks(video_path)
        if not frames:
            return generate_hamnosys(video_path)[0]

        frame_vecs = [extract_normalized_features(f) for f in frames]
        frame_matrix = np.array(frame_vecs, dtype=np.float32)

        target_frames = 150
        T_curr, D = frame_matrix.shape
        if T_curr < target_frames:
            pad_width = ((0, target_frames - T_curr), (0, 0))
            padded_matrix = np.pad(frame_matrix, pad_width, mode="edge")
        else:
            padded_matrix = frame_matrix[:target_frames]

        tensor_in = torch.tensor(padded_matrix).unsqueeze(0) # (1, 150, 177)

        model = HamNoSysSequenceNet(
            input_dim=177,
            num_hs=len(mappings["handshape"]),
            num_ext=len(mappings["ext_finger"]),
            num_palm=len(mappings["palm_ori"]),
            num_loc=len(mappings["location"]),
            num_mov=len(mappings["movement"]),
            num_two=len(mappings["two_handed"])
        )
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

        with torch.no_grad():
            preds = model(tensor_in)

        hs = mappings["handshape"][preds["handshape"].argmax().item()]
        ext = mappings["ext_finger"][preds["ext_finger"].argmax().item()]
        palm = mappings["palm_ori"][preds["palm_ori"].argmax().item()]
        loc = mappings["location"][preds["location"].argmax().item()]
        mov = mappings["movement"][preds["movement"].argmax().item()]
        two = mappings["two_handed"][preds["two_handed"].argmax().item()]

        sequence = []
        if two and two != "none":
            sequence.append(two)

        sequence.extend([hs, ext, palm, loc])

        if mov and mov != "none":
            sequence.append(mov)

        hamnosys_str = " ".join(sequence)
        print(f"[NeuralPipeline Prediction]: {hamnosys_str}")
        return hamnosys_str

    except Exception as e:
        print(f"[NeuralPipeline Exception]: {e} — falling back to heuristic pipeline...")
        return generate_hamnosys(video_path)[0]


def process_video(video_path, output_video=None, mirror=False):
    """
    Full pipeline: generate HamNoSys + annotated video.
    """
    try:
        from video_preprocessing import get_mirrored_video, detect_dominant_hand
        
        dominant_hand = detect_dominant_hand(video_path)
        print(f"\n[Preprocessing] Detected Dominant Hand: {dominant_hand.upper()} wrist")
        
        if mirror or dominant_hand == "left":
            if dominant_hand == "left" and not mirror:
                print(f"[Preprocessing] Auto-mirroring video to standardize to right-handed signer...")
            else:
                print(f"Mirroring video (forced by flag)...")
            video_path = get_mirrored_video(video_path)
            
    except ImportError:
        pass

    hamnosys_code = process_video_neural(video_path)
    return {
        "hamnosys": hamnosys_code,
        "output_video": output_video
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HamNoSys Integration Pipeline")
    parser.add_argument("video", nargs="?", default="Prompt_1.mp4",
                        help="Path to input video (default: Prompt_1.mp4)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output video path (default: <input>_output.mp4)")
    parser.add_argument("-m", "--mirror", action="store_true",
                        help="Horizontally mirror the video before processing")
    args = parser.parse_args()

    result = process_video(args.video, args.output, args.mirror)
    print("\n[Done!]")
    print(f"  HamNoSys: {result['hamnosys']}")
    print(f"  Video:    {result['output_video']}")
