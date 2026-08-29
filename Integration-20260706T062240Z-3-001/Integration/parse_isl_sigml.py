#!/usr/bin/env python3
"""
Phase 1: Automated Ground Truth Parser for ISL SiGML Files
Parses .sigml ground truth XML files from SignFiles/ and matches them
with corresponding ISL video files to produce isl_ground_truth_labels.csv

Reference: HYBRID_NN_ARCHITECTURE_AND_TRAINING_SPECIFICATION.md - Step 1
"""

import os
import sys
import csv
import xml.etree.ElementTree as ET

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))

SIGML_DIR = os.path.join(_REPO_ROOT, "SignFiles")
VIDEOS_DIR = os.path.join(_REPO_ROOT, "ISL_videos")
OUTPUT_CSV = os.path.join(_SCRIPT_DIR, "isl_ground_truth_labels.csv")

# =====================================================
# HamNoSys Token Classification Sets
# =====================================================

HANDSHAPE_BASE_SET = {
    "hamfist", "hamflathand", "hamfinger2", "hamfinger23", "hamfinger23spread",
    "hamfinger2345", "hampinch12", "hampinch12open", "hampinchall",
    "hamcee12", "hamceeall", "hamindexfinger", "hammiddlefinger",
    "hamfinger2spread", "hamfinger23spreadthin",
}

THUMB_MOD_SET = {
    "hamthumboutmod", "hamthumbacrossmod", "hamthumbopenmod",
}

BEND_MOD_SET = {
    "hamdoublebent", "hamdoublehooked",
    "hamfingerstraightmod", "hamfingerbendmod", "hamfingerhookmod",
}

EXTENDED_FINGER_SET = {
    "hamextfingeru", "hamextfingerur", "hamextfingerr", "hamextfingerdr",
    "hamextfingerd", "hamextfingerul", "hamextfingerl", "hamextfingerdl",
    "hamextfingero", "hamextfingeror", "hamextfingerir",
    "hamextfingeri", "hamextfingerol", "hamextfingeril",
    "hamextfingeruo", "hamextfingerdo", "hamextfingerui", "hamextfingerdi",
}

PALM_ORI_SET = {
    "hampalmu", "hampalmur", "hampalmr", "hampalmdr",
    "hampalmd", "hampalmul", "hampalml", "hampalmdl",
    "hampalmo", "hampalmi",
}

LOCATION_SET = {
    "hamhead", "hamforehead", "hamheadtop", "hameyes", "hamnose",
    "hamlips", "hamchin", "hamear", "hamcheek", "hamneck",
    "hamshoulders", "hamshouldertop", "hamchest", "hamstomach",
    "hambelowstomach", "hamupperarm", "hamelbowinside",
    "hamwristback", "hamwristfront",
    "hamneutralspace",
}

CONTACT_SET = {
    "hamtouch", "hamclose", "haminterlock", "hamcross", "hamhook",
    "hambehind", "hamarmextended", "hamlrat", "hamlrbeside",
}

MOVEMENT_BASE_SET = {
    "hammoveu", "hammoved", "hammover", "hammovel",
    "hammoveo", "hammovei", "hammoveuo", "hammovedi",
    "hamswinging", "hamwavy", "hamcircleo", "hamcirclei",
    "hamcircleol", "hamcircleor", "hamcircleul", "hamcircledr",
    "hamstircw", "hamstirccw",
    "hamnomotion", "hamellipticcw", "hamellipticccw",
    "hamclockfull", "hamclockhalf", "hamclockquarter",
}

MOVEMENT_MOD_SET = {
    "hamlargemod", "hamsmallmod", "hamslow", "hamfast",
    "hamrest", "hamhalt", "hamincreasing",
}

REPETITION_SET = {
    "hamrepeatfromstart", "hamrepeatreverse",
    "hamrepeatfromstartseveral", "hamrepeatseveral",
}

SYMMETRY_SET = {
    "hamsymmlr", "hamplus",
}

HAND_PART_SET = {
    "hamthumb", "hamindexfinger", "hammiddlefinger", "hamringfinger", "hampinky",
    "hambetween", "hamfingernail", "hamfingerpad", "hamfingerside",
    "hamfingermidjoint", "hamfingertip",
    "hampalm", "hamthumbside", "hampinkyside", "hamhandback",
}


def parse_sigml_ground_truth(sigml_path):
    """
    Extract HamNoSys linguistic components from a SiGML XML file.
    Returns a dictionary with classified token slots.
    """
    try:
        tree = ET.parse(sigml_path)
        root = tree.getroot()
    except ET.ParseError:
        return None

    # Find hamnosys_manual element
    manual = root.find(".//hamnosys_manual")
    if manual is None:
        return None

    tags = [elem.tag for elem in manual]
    all_tags_str = " ".join(tags)

    # 1. Base Handshape
    base_hs = next((t for t in tags if t in HANDSHAPE_BASE_SET), "hamflathand")

    # 2. Thumb Modifier
    thumb_mod = next((t for t in tags if t in THUMB_MOD_SET), "none")

    # 3. Bend Modifier
    bend_mod = next((t for t in tags if t in BEND_MOD_SET), "none")

    # 4. Extended Finger Orientation
    ext_finger = next((t for t in tags if t in EXTENDED_FINGER_SET), "hamextfingeru")

    # 5. Palm Orientation
    palm_ori = next((t for t in tags if t in PALM_ORI_SET), "hampalmd")

    # 6. Body Location
    location = next((t for t in tags if t in LOCATION_SET), "hamneutralspace")

    # 7. Contact
    contact = next((t for t in tags if t in CONTACT_SET), "none")

    # 8. Movement Base
    movement = next((t for t in tags if t in MOVEMENT_BASE_SET), "hamnomotion")

    # 9. Movement Modifier
    move_mod = next((t for t in tags if t in MOVEMENT_MOD_SET), "none")

    # 10. Repetition
    repetition = next((t for t in tags if t in REPETITION_SET), "none")

    # 11. Two-Handed Structure
    two_handed = "none"
    if "hamsymmlr" in tags:
        two_handed = "hamsymmlr"
    elif "hamplus" in tags:
        two_handed = "hamplus"

    # 12. Hand Part
    hand_part = next((t for t in tags if t in HAND_PART_SET), "none")

    # Gloss
    sign_elem = root.find(".//hns_sign")
    gloss = sign_elem.get("gloss", "") if sign_elem is not None else ""

    return {
        "gloss": gloss,
        "handshape": base_hs,
        "thumb_mod": thumb_mod,
        "bend_mod": bend_mod,
        "ext_finger": ext_finger,
        "palm_ori": palm_ori,
        "location": location,
        "contact": contact,
        "movement": movement,
        "move_mod": move_mod,
        "repetition": repetition,
        "two_handed": two_handed,
        "hand_part": hand_part,
        "all_tags": all_tags_str,
    }


def match_videos_to_sigml():
    """
    Match ISL video files to corresponding SiGML ground truth files
    using case-insensitive name matching.
    """
    # Build lookup: normalized_name -> filename
    video_lookup = {}
    for f in os.listdir(VIDEOS_DIR):
        if f.lower().endswith('.mp4'):
            # Normalize: "All.mp4" -> "all", "Act [VEB].mp4" -> "act[veb]"
            base = os.path.splitext(f)[0].lower().replace(' ', '')
            video_lookup[base] = f

    sigml_lookup = {}
    for f in os.listdir(SIGML_DIR):
        if f.lower().endswith('.sigml'):
            base = os.path.splitext(f)[0].lower().replace(' ', '')
            sigml_lookup[base] = f

    # Find matches
    matched = []
    for name in sorted(set(video_lookup.keys()) & set(sigml_lookup.keys())):
        matched.append({
            "video_file": video_lookup[name],
            "sigml_file": sigml_lookup[name],
            "name": name,
        })

    return matched


def build_ground_truth_csv():
    """
    Main function: Parse all matched video-SiGML pairs and write training labels.
    """
    print("=" * 60)
    print("  Phase 1: ISL Ground Truth Label Extraction")
    print("=" * 60)

    matches = match_videos_to_sigml()
    print(f"  Found {len(matches)} matched video-SiGML pairs")

    rows = []
    parse_errors = 0

    for m in matches:
        sigml_path = os.path.join(SIGML_DIR, m["sigml_file"])
        labels = parse_sigml_ground_truth(sigml_path)

        if labels is None:
            parse_errors += 1
            continue

        video_path = os.path.join(VIDEOS_DIR, m["video_file"])
        row = {
            "video_file": m["video_file"],
            "video_path": video_path,
            "sigml_file": m["sigml_file"],
            **labels,
        }
        rows.append(row)

    # Write CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"  Successfully parsed: {len(rows)} files")
    print(f"  Parse errors: {parse_errors}")
    print(f"  Output: {OUTPUT_CSV}")

    # Print class distribution summary
    print("\n  --- Class Distribution ---")
    from collections import Counter
    for field in ["handshape", "movement", "two_handed", "ext_finger", "palm_ori", "location"]:
        counts = Counter(r[field] for r in rows)
        print(f"\n  {field}: {len(counts)} classes")
        for cls, cnt in counts.most_common(8):
            print(f"    {cls}: {cnt}")
        if len(counts) > 8:
            print(f"    ... and {len(counts) - 8} more")

    return rows


if __name__ == "__main__":
    rows = build_ground_truth_csv()
    print(f"\n[DONE] Generated {len(rows)} training labels -> {OUTPUT_CSV}")
