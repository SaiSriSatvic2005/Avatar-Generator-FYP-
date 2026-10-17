#!/usr/bin/env python3
"""
Full SiGML XML Semantic Parser & Clean Ground Truth Dataset Builder
Extracts complete phonology:
- Base Handshapes + Finger/Thumb Modifiers
- Dual-Hand Structures (hamsymmlr, hamparbegin, hamplus)
- Full 18-Direction Palm & Extended Finger Orientations
- Anatomical Locations (including hamneutralspace)
- Compound Movement Sequences
"""

import os
import glob
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SIGML_DIR = os.path.join(ROOT_DIR, "downloaded_dgs_dataset", "sigml")
OUTPUT_CSV = os.path.join(ROOT_DIR, "downloaded_dgs_dataset", "dgs_clean_ground_truth.csv")

BASE_HANDSHAPES = {
    "hamflathand", "hamfist", "hamfinger2", "hamfinger23", "hamfinger23spread",
    "hamfinger2345", "hampinch12", "hampinchall", "hampinch12open", "hamcee12",
    "hamceeall", "hamceeopen"
}

HAND_MODIFIERS = {
    "hamfingerstraightmod", "hamfingerbendmod", "hamfingerhookmod",
    "hamthumboutmod", "hamthumbacrossmod", "hamthumbopenmod"
}

EXT_DIRECTIONS = {
    "hamextfingeru", "hamextfingerd", "hamextfingerl", "hamextfingerr",
    "hamextfingerul", "hamextfingerur", "hamextfingerdl", "hamextfingerdr",
    "hamextfingero", "hamextfingeri", "hamextfingerol", "hamextfingeror",
    "hamextfingeruo", "hamextfingerdo", "hamextfingerui", "hamextfingerdi"
}

PALM_DIRECTIONS = {
    "hampalmu", "hampalmd", "hampalml", "hampalmr",
    "hampalmul", "hampalmur", "hampalmdl", "hampalmdr",
    "hampalmo", "hampalmi"
}

LOCATIONS = {
    "hamhead", "hamforehead", "hameyes", "hamnose", "hamcheek", "hamlips",
    "hamteeth", "hamtongue", "hamchin", "hamunderchin", "hamneck",
    "hamshouldertop", "hamshoulders", "hamchest", "hamstomach", "hambelly",
    "hamneutralspace", "hampalm", "hamthumbball", "hamhandback", "hamthumbside"
}

MOVEMENTS = {
    "hammoveu", "hammoved", "hammovel", "hammover", "hammoveo", "hammovei",
    "hammoveui", "hammoveuo", "hammovedi", "hammovedo", "hammoveul", "hammoveur",
    "hammovedl", "hammovedr", "hamcircleol", "hamcircleor", "hamcircleil",
    "hamcircleir", "hamwavy", "hamnodding", "hamnomotion"
}

def parse_sigml_file(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except Exception as e:
        return None

    tokens = []
    for elem in root.iter():
        tag = elem.tag.strip()
        if tag not in ["sigml", "hns_sign", "hamnosys_nonmanual", "hamnosys_manual"]:
            tokens.append(tag)

    if not tokens:
        return None

    is_two_handed = "hamsymmlr" in tokens or "hamsymmpar" in tokens or "hamparbegin" in tokens
    symmetry_type = "single"
    if "hamsymmlr" in tokens: symmetry_type = "hamsymmlr"
    elif "hamsymmpar" in tokens: symmetry_type = "hamsymmpar"
    elif "hamparbegin" in tokens: symmetry_type = "hamparbegin"

    base_hs = next((t for t in tokens if t in BASE_HANDSHAPES), "hamflathand")
    hs_mods = [t for t in tokens if t in HAND_MODIFIERS]
    ext_dir = next((t for t in tokens if t in EXT_DIRECTIONS), "hamextfingero")
    palm_dir = next((t for t in tokens if t in PALM_DIRECTIONS), "hampalmd")
    location = next((t for t in tokens if t in LOCATIONS), "hamneutralspace")
    movement = next((t for t in tokens if t in MOVEMENTS), "hamnomotion")

    has_touch = "hamtouch" in tokens or "hamclose" in tokens

    stem = Path(file_path).stem
    video_file = f"{stem}.mp4"

    return {
        "stem": stem,
        "video_file": video_file,
        "sigml_file": os.path.basename(file_path),
        "is_two_handed": is_two_handed,
        "symmetry_type": symmetry_type,
        "base_handshape": base_hs,
        "handshape_modifiers": " ".join(hs_mods),
        "extended_finger": ext_dir,
        "palm_orientation": palm_dir,
        "location": location,
        "movement": movement,
        "has_touch": has_touch,
        "raw_tokens": " ".join(tokens)
    }

def main():
    sigml_files = sorted(glob.glob(os.path.join(SIGML_DIR, "*.sigml")))
    print(f"[*] Found {len(sigml_files)} SiGML XML files in {SIGML_DIR}")

    records = []
    for f in sigml_files:
        rec = parse_sigml_file(f)
        if rec:
            records.append(rec)

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[+] Successfully parsed {len(df)} signs into: {OUTPUT_CSV}")
    print(f"[*] Summary Statistics:")
    print(f"    - Two-handed signs: {df['is_two_handed'].sum()} / {len(df)} ({df['is_two_handed'].mean()*100:.1f}%)")
    print(f"    - Neutral space locations: {(df['location'] == 'hamneutralspace').sum()} / {len(df)} ({(df['location'] == 'hamneutralspace').mean()*100:.1f}%)")
    print(f"    - Handshapes with modifiers: {(df['handshape_modifiers'] != '').sum()} / {len(df)} ({(df['handshape_modifiers'] != '').mean()*100:.1f}%)")

if __name__ == "__main__":
    main()
