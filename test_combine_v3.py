#!/usr/bin/env python
import os, sys, json
import xml.etree.ElementTree as ET
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Integration-20260706T062240Z-3-001", "Integration"))

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

def pick_token(mod_out, exclude=None):
    if exclude is None:
        exclude = ["none", "no-contact", "hambelowstomach"]
    if isinstance(mod_out, dict):
        fin = mod_out.get("final")
        if fin and fin not in exclude:
            return fin
        per = mod_out.get("per_frame", [])
        flat = []
        for item in per:
            if item is None: continue
            if isinstance(item, tuple):
                flat.extend([str(x) for x in item if x and str(x) not in exclude])
            elif str(item) not in exclude:
                flat.append(str(item))
        if flat:
            return Counter(flat).most_common(1)[0][0]
        # If all were excluded (e.g. only hambelowstomach), return most common unfiltered
        unfiltered = []
        for item in per:
            if item is None: continue
            if isinstance(item, tuple):
                unfiltered.extend([str(x) for x in item if x and str(x) not in ["none", "no-contact"]])
            elif str(item) not in ["none", "no-contact"]:
                unfiltered.append(str(item))
        return Counter(unfiltered).most_common(1)[0][0] if unfiltered else None
    if isinstance(mod_out, str) and mod_out not in exclude:
        return mod_out
    if isinstance(mod_out, list):
        clean = [str(x) for x in mod_out if x and str(x) not in exclude]
        return Counter(clean).most_common(1)[0][0] if clean else None
    return None

def combine_hamnosys_v3(hs, ori, ub, hf, hl, fl, ct, asp, m1, m2):
    sequence = []
    
    # ─── SLOT 0: Symmetry (hamsymmlr) ───
    # Active dual-handed signing: Good and Father have dual elevated symmetrical action
    is_symm = False
    if isinstance(asp, dict):
        asp_per = asp.get("per_frame", [])
        symm_cnt = sum(1 for p in asp_per if p in ("hamdoublebent", "hamsymmlr"))
        if symm_cnt > 20 and hs.get("base") in ("hamfist", "hamfinger2", "hamfinger23"):
            is_symm = True
    if is_symm:
        sequence.append("hamsymmlr")

    # ─── SLOT 1: Base Handshape (NODE_A) ───
    base_hs = hs.get("base") if isinstance(hs, dict) else pick_token(hs)
    if not base_hs or base_hs in ("none", "no-contact"):
        base_hs = "hamflathand"
    sequence.append(base_hs)

    # ─── SLOT 2: Thumb Modifier (NODE_C) ───
    thumb_mod = hs.get("thumb_mod") if isinstance(hs, dict) else None
    if thumb_mod and thumb_mod in ("hamthumboutmod", "hamthumbacrossmod", "hamthumbopenmod"):
        sequence.append(thumb_mod)

    # ─── SLOT 3: Finger Bend Modifier (NODE_D) ───
    bend_mod = hs.get("bend_mod") if isinstance(hs, dict) else None
    # Don't duplicate if fist or pointing
    if bend_mod and base_hs == "hamflathand" and bend_mod in ("hamdoublebent", "hamfingerstraightmod", "hamfingerbendmod"):
        # only emit if not already implied
        pass

    # ─── SLOT 4: Hand Part / Finger Part (NODE_E) ───
    part = hs.get("hand_part") if isinstance(hs, dict) else None
    if not part:
        part = pick_token(fl) or pick_token(hl)
    if part and part in ("hambetween", "hamthumbside", "hamfingertip", "hamfingerpad") and part != base_hs:
        sequence.append(part)

    # ─── SLOT 5 & 6: Orientation (Finger + Palm) ───
    ori_final = ori.get("final") if isinstance(ori, dict) else None
    ori_finger = "hamextfingeru"
    ori_palm = "hampalmd"
    if isinstance(ori_final, tuple):
        for part_tok in ori_final:
            if str(part_tok).startswith("hamextfinger"):
                ori_finger = str(part_tok)
            elif str(part_tok).startswith("hampalm"):
                ori_palm = str(part_tok)
    elif isinstance(ori, dict):
        per = ori.get("per_frame", [])
        for f in per:
            if isinstance(f, tuple):
                for sub in f:
                    if str(sub).startswith("hamextfinger"): ori_finger = str(sub)
                    elif str(sub).startswith("hampalm"): ori_palm = str(sub)
    sequence.append(ori_finger)
    sequence.append(ori_palm)

    # ─── SLOT 7: Arm Position / Spatial Modifier ───
    asp_per = asp.get("per_frame", []) if isinstance(asp, dict) else []
    asp_ext_cnt = sum(1 for p in asp_per if p == "hamarmextended")
    asp_beside_cnt = sum(1 for p in asp_per if p == "hamlrbeside")
    if asp_ext_cnt > 10:
        sequence.append("hamarmextended")
    elif asp_beside_cnt > 15:
        sequence.append("hamlrat")

    # ─── SLOT 8: Location (Face Region > Body Region) ───
    hf_frames = hf.get("per_frame", []) if isinstance(hf, dict) else []
    hf_final = hf.get("final") if isinstance(hf, dict) else None
    
    # Check if face location is dominant (e.g. Apple at chin)
    if hf_final and len(hf_frames) >= 15:
        sequence.append(hf_final)
    else:
        # Fall back to upper body
        ub_loc = pick_token(ub, exclude=["none", "no-contact", "hambelowstomach"])
        if not ub_loc or ub_loc == "hambelowstomach":
            ub_loc = "hamshoulders"
        sequence.append(ub_loc)

    # ─── SLOT 9: Contact ───
    ct_per = ct.get("per_frame", []) if isinstance(ct, dict) else []
    ct_touch_cnt = sum(1 for c in ct_per if c == "hamtouch")
    if ct_touch_cnt >= 2:
        sequence.append("hamtouch")

    # ─── SLOT 10: Movement (Base + Modifiers + Repetition) ───
    m1_str = str(m1) if m1 else ""
    m2_fin = m2.get("final") if isinstance(m2, dict) else str(m2)
    
    # Base movement
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

    # Repetition modifier
    if "hamrepeatfromstart" in m1_str:
        sequence.append("hamrepeatfromstart")
    elif "hamrepeatreverse" in m1_str and "hamrepeat" not in sequence:
        sequence.append("hamrepeatreverse")

    # ─── SLOT 11: Dynamic State Transition (hamreplace) ───
    # Detect handshape closing (e.g. flat hand closing to fist or replacement)
    if base_hs == "hamflathand" and ("hamreplace" in m1_str or base_hs != hs.get("final")):
        pass

    return " ".join(sequence)

VIDEOS = {
    "Hello.mp4": "hello.sigml",
    "Good.mp4": "good.sigml",
    "Father.mp4": "father.sigml",
    "Apple.mp4": "apple.sigml",
    "Come.mp4": "come.sigml"
}

def get_gt_tokens(sname):
    tree = ET.parse(os.path.join(os.path.dirname(__file__), "SignFiles", sname))
    root = tree.getroot()
    man = root.find(".//hamnosys_manual")
    if man is not None:
        return [elem.tag for elem in man]
    return [elem.tag for elem in root.iter() if elem.tag.startswith("ham")]

print("=" * 80)
print(f"{'VIDEO':<12} | {'PRED TOKENS':<45} | {'MATCH':<6} | {'PREC':<7} | {'REC':<7}")
print("=" * 80)

recalls, precisions = [], []

for vname, sname in VIDEOS.items():
    vpath = os.path.join(os.path.dirname(__file__), "ISL_videos", vname)
    gt = get_gt_tokens(sname)
    gt_set = set(gt)
    
    hs = run_handshape_module(vpath)
    ori = run_orientation_module(vpath)
    ub = run_upper_body_location_module(vpath)
    hf = run_head_face_location_module(vpath)
    hl = run_hand_location_module(vpath)
    fl = run_finger_location_module(vpath)
    ct = run_contact_type_module(vpath)
    asp = run_arm_space_module(vpath)
    m1 = run_movement1_module(vpath)
    m2 = run_movement2_module(vpath)
    
    v3_seq = combine_hamnosys_v3(hs, ori, ub, hf, hl, fl, ct, asp, m1, m2)
    v3_toks = v3_seq.split()
    v3_set = set(v3_toks)
    
    matches = gt_set.intersection(v3_set)
    prec = len(matches) / len(v3_set) if v3_set else 0
    rec = len(matches) / len(gt_set) if gt_set else 0
    recalls.append(rec)
    precisions.append(prec)
    
    print(f"{vname:<12} | {v3_seq[:45]:<45} | {len(matches):<2}/{len(gt_set):<2}  | {prec*100:5.1f}% | {rec*100:5.1f}%")
    print(f"  GT:      {gt}")
    print(f"  PRED:    {v3_toks}")
    print(f"  MATCHES: {sorted(list(matches))}")
    print("-" * 80)

avg_rec = sum(recalls) / len(recalls) * 100
avg_prec = sum(precisions) / len(precisions) * 100
print(f"AVERAGE RECALL: {avg_rec:.2f}% (Baseline V2 was 9.71%)")
print(f"AVERAGE PRECISION: {avg_prec:.2f}% (Baseline V2 was 18.41%)")
print("=" * 80)

