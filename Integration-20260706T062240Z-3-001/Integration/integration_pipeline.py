#!/usr/bin/env python
# coding: utf-8
"""
HamNoSys Integration Pipeline — Local Version
Runs all 10 modules on a sign language video and produces a HamNoSys code string.
"""

import os
import sys
import numpy as np
import cv2
import mediapipe as mp

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
                    finger_location, contact, arm_space, movement1, movement2):
    """
    Combine module outputs into a strict, grammatically valid SiGML HamNoSys sequence (V3).
    
    Formal HamNoSys CFG Rules (11 Ordered Slots):
    Sign ::= [Symmetry] Handshape [ThumbMod] [BendMod] [HandPart]
             ExtendedFinger PalmOrientation [ArmPosition] BodyLocation 
             [Contact] Movement [MovementModifiers] [Repetition]
             [hamreplace FinalHandshape FinalOrientation FinalLocation]
    """
    from collections import Counter

    def pick_token(module_output, exclude=None):
        if exclude is None:
            exclude = ["none", "no-contact", "hambelowstomach"]
        if isinstance(module_output, dict):
            final = module_output.get("final")
            if final and final not in exclude:
                return final
            per = module_output.get("per_frame", [])
            flat = []
            for item in per:
                if item is None: continue
                if isinstance(item, tuple):
                    flat.extend([str(x) for x in item if x and str(x) not in exclude])
                elif str(item) not in exclude:
                    flat.append(str(item))
            if flat:
                return Counter(flat).most_common(1)[0][0]
            # Fallback without exclusion if all were filtered
            fallback_flat = []
            for item in per:
                if item is None: continue
                if isinstance(item, tuple):
                    fallback_flat.extend([str(x) for x in item if x and str(x) not in ["none", "no-contact"]])
                elif str(item) not in ["none", "no-contact"]:
                    fallback_flat.append(str(item))
            return Counter(fallback_flat).most_common(1)[0][0] if fallback_flat else None
        if isinstance(module_output, str) and module_output not in exclude:
            return module_output
        if isinstance(module_output, list):
            clean = [str(x) for x in module_output if x and str(x) not in exclude]
            return Counter(clean).most_common(1)[0][0] if clean else None
        return None

    sequence = []

    # ─── SLOT 0: Symmetry Operator (hamsymmlr) ───
    is_symm = False
    if isinstance(arm_space, dict):
        asp_per = arm_space.get("per_frame", [])
        symm_cnt = sum(1 for p in asp_per if p in ("hamdoublebent", "hamsymmlr"))
        hs_base_cand = handshape.get("base") if isinstance(handshape, dict) else pick_token(handshape)
        if symm_cnt > 15 and hs_base_cand in ("hamfist", "hamfinger2", "hamfinger23"):
            is_symm = True
    if is_symm:
        sequence.append("hamsymmlr")

    # ─── SLOT 1: Base Handshape (NODE_A) ───
    base_hs = handshape.get("base") if isinstance(handshape, dict) else pick_token(handshape)
    if not base_hs or base_hs in ("none", "no-contact"):
        base_hs = "hamflathand"
    sequence.append(base_hs)

    # ─── SLOT 2: Thumb Modifier (NODE_C) ───
    thumb_mod = handshape.get("thumb_mod") if isinstance(handshape, dict) else None
    if thumb_mod and thumb_mod in ("hamthumboutmod", "hamthumbacrossmod", "hamthumbopenmod"):
        sequence.append(thumb_mod)

    # ─── SLOT 3: Finger Bend Modifier (NODE_D) ───
    bend_mod = handshape.get("bend_mod") if isinstance(handshape, dict) else None
    if bend_mod and base_hs == "hamflathand" and bend_mod in ("hamfingerstraightmod", "hamdoublebent"):
        pass  # omit redundant bend modifier for basic flat hand

    # ─── SLOT 4: Hand Part / Finger Part (NODE_E) ───
    part = handshape.get("hand_part") if isinstance(handshape, dict) else None
    if not part:
        part = pick_token(finger_location) or pick_token(hand_location)
    if part and part in ("hambetween", "hamthumbside", "hamfingertip", "hamfingerpad") and part != base_hs:
        sequence.append(part)

    # ─── SLOT 5 & 6: Initial Orientation (Extended Finger + Palm) ───
    ori_finger = "hamextfingeru"
    ori_palm = "hampalmd"
    if isinstance(orientation, dict):
        fin_ori = orientation.get("final")
        if isinstance(fin_ori, tuple):
            for tok in fin_ori:
                if str(tok).startswith("hamextfinger"): ori_finger = str(tok)
                elif str(tok).startswith("hampalm"): ori_palm = str(tok)
        else:
            per_ori = orientation.get("per_frame", [])
            for f in per_ori:
                if isinstance(f, tuple):
                    for tok in f:
                        if str(tok).startswith("hamextfinger") and tok != "hamextfingerd": ori_finger = str(tok)
                        elif str(tok).startswith("hampalm"): ori_palm = str(tok)
    sequence.append(ori_finger)
    sequence.append(ori_palm)

    # ─── SLOT 7: Body / Face / Arm Location ───
    hf_frames = head_face.get("per_frame", []) if isinstance(head_face, dict) else []
    hf_final = head_face.get("final") if isinstance(head_face, dict) else None
    
    if hf_final and len(hf_frames) >= 15:
        sequence.append(hf_final)
    else:
        ub_loc = pick_token(upper_body, exclude=["none", "no-contact", "hambelowstomach"])
        if not ub_loc or ub_loc == "hambelowstomach":
            ub_loc = "hamshoulders"
        sequence.append(ub_loc)

    # ─── SLOT 8: Contact OR Spatial Distance (Mutually exclusive in simple location) ───
    ct_per = contact.get("per_frame", []) if isinstance(contact, dict) else []
    ct_touch_cnt = sum(1 for c in ct_per if c == "hamtouch")
    asp_per = arm_space.get("per_frame", []) if isinstance(arm_space, dict) else []
    asp_ext_cnt = sum(1 for p in asp_per if p == "hamarmextended")
    asp_beside_cnt = sum(1 for p in asp_per if p == "hamlrbeside")

    if ct_touch_cnt >= 2:
        sequence.append("hamtouch")
    elif asp_ext_cnt > 10:
        sequence.append("hamarmextended")
    elif asp_beside_cnt > 15:
        sequence.append("hamlrat")

    # ─── SLOT 9: Movement (Base + Modifiers + Repetition) ───
    m1_str = str(movement1) if movement1 else ""
    m2_fin = movement2.get("final") if isinstance(movement2, dict) else str(movement2)
    
    mov_base = None
    if "hamswinging" in m1_str:
        mov_base = "hamswinging"
    elif "hammover" in m1_str:
        mov_base = "hammover"
    elif "hammoveu" in m1_str:
        mov_base = "hammoveu"
    elif "hammoved" in m1_str:
        mov_base = "hammoved"
    elif "hammoveo" in m1_str:
        mov_base = "hammoveo"
    elif m2_fin and m2_fin.startswith("ham"):
        mov_base = m2_fin
    
    if mov_base and mov_base != "hamnomotion":
        sequence.append(mov_base)

    if "hamrepeatfromstart" in m1_str:
        sequence.append("hamrepeatfromstart")
    elif "hamrepeatreverse" in m1_str and "hamrepeat" not in sequence:
        sequence.append("hamrepeatreverse")

    # ─── SLOT 10: Dynamic State Transition (hamreplace) ───
    # If starting flat hand closes to fist at stomach (e.g. Come sign)
    ub_tokens = []
    if isinstance(upper_body, dict):
        for f in upper_body.get("per_frame", []):
            if isinstance(f, str): ub_tokens.append(f)
            elif isinstance(f, tuple): ub_tokens.extend([str(x) for x in f])
    elif isinstance(upper_body, list):
        ub_tokens = [str(x) for x in upper_body]
    elif isinstance(upper_body, str):
        ub_tokens = upper_body.split()

    if base_hs == "hamflathand" and "hamstomach" in ub_tokens and "hambelowstomach" != pick_token(upper_body):
        sequence.extend(["hamreplace", "hamfist", "hamstomach"])

    # Pass through generalized ANTLR Ham4Parser / CWASA Grammar Linter
    try:
        from sigml_grammar_linter import sanitize_hamnosys_sequence
        canonical_tokens = sanitize_hamnosys_sequence(sequence)
        return " ".join(canonical_tokens)
    except Exception as e:
        print(f"Linter fallback warning: {e}")
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
        movement2
    )

    print("\n========== FINAL HAMNOSYS ==========")
    print(hamnosys_code)

    return hamnosys_code, {
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

def process_video(video_path, output_video=None):
    """
    Full pipeline: generate HamNoSys + annotated video.
    """

    if output_video is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_video = os.path.join(_SCRIPT_DIR, f"{base}_output.mp4")

    hamnosys_code, modules = generate_hamnosys(video_path)

    create_annotated_video(video_path, output_video, modules)

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
    args = parser.parse_args()

    result = process_video(args.video, args.output)
    print("\n✓ Done!")
    print(f"  HamNoSys: {result['hamnosys']}")
    print(f"  Video:    {result['output_video']}")
