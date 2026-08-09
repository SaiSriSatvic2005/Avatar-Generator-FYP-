#!/usr/bin/env python3
"""
Dictionary HamNoSys & SiGML Verifier
Tests all 40 gloss definitions in gloss_to_hamnosys_dict.json against:
1. Symbol mapping table (conversionSpreadSheet.txt)
2. SiGML Compiler (HamNoSys2SiGML.py)
"""

import os
import sys
import json
import subprocess

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(_SCRIPT_DIR))

DICT_PATH = os.path.join(_SCRIPT_DIR, "gloss_to_hamnosys_dict.json")
SPREADSHEET_PATH = os.path.join(BASE_DIR, "Senior Code", "HamNoSys2SiGML-master", "HamNoSys2SiGML-master", "Original", "conversionSpreadSheet.txt")
HAM2SIGML_SCRIPT = os.path.join(BASE_DIR, "Senior Code", "HamNoSys2SiGML-master", "HamNoSys2SiGML-master", "Original", "HamNoSys2SiGML.py")

def load_reverse_mapping(path):
    mapping = {}
    if not os.path.exists(path):
        print(f"❌ Error: Spreadsheet path not found: {path}")
        return mapping
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if "," in line:
                parts = line.strip().split(",")
                tag = parts[0].strip()
                code = parts[1].strip().split()[0].strip()
                mapping[tag] = code
    return mapping

def verify_all():
    print("="*75)
    print("  DICTIONARY HAMNOSYS & SIGML VERIFICATION REPORT")
    print("="*75)

    if not os.path.exists(DICT_PATH):
        print(f"❌ Error: Dictionary file not found: {DICT_PATH}")
        return

    with open(DICT_PATH, "r", encoding="utf-8") as f:
        gloss_dict = json.load(f)

    mapping = load_reverse_mapping(SPREADSHEET_PATH)
    print(f"[OK] Loaded {len(mapping)} HamNoSys-to-Unicode symbol mappings.")
    print(f"[OK] Testing {len(gloss_dict)} Gloss Dictionary Definitions...\n")

    print(f"{'#':<3} | {'Gloss':<14} | {'Two-Handed':<10} | {'Unmapped Tags':<15} | {'SiGML Status':<12}")
    print("-" * 75)

    passed_count = 0
    failed_count = 0
    results = []

    for idx, (gloss, entry) in enumerate(gloss_dict.items(), 1):
        # Assemble HamNoSys tag sequence
        tags = []
        th = entry.get("two_handed", "none")
        if th != "none":
            tags.append(th)

        tags.extend([
            entry.get("handshape", ""),
            entry.get("ext_finger", ""),
            entry.get("palm_ori", ""),
            entry.get("location", "")
        ])

        mov = entry.get("movement", "none")
        if mov != "none":
            tags.append(mov)

        # Map tags to Unicode
        unmapped = []
        unicode_chars = []
        for tag in tags:
            if not tag: continue
            if tag in mapping:
                try:
                    unicode_chars.append(chr(int(mapping[tag], 16)))
                except ValueError:
                    unmapped.append(tag)
            else:
                unmapped.append(tag)

        unicode_str = "".join(unicode_chars)
        sigml_status = "UNKNOWN"
        xml_len = 0

        if unicode_str and os.path.exists(HAM2SIGML_SCRIPT):
            cmd = [sys.executable, HAM2SIGML_SCRIPT, unicode_str]
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(HAM2SIGML_SCRIPT))
            sigml_out = proc.stdout.strip()

            if "<sigml" in sigml_out.lower() and "<hns_sign" in sigml_out.lower():
                sigml_status = "[OK] PASS"
                passed_count += 1
                xml_len = len(sigml_out)
            else:
                sigml_status = "[FAIL] INVALID"
                failed_count += 1
        else:
            sigml_status = "[FAIL] NO SIGML"
            failed_count += 1

        unmapped_str = ", ".join(unmapped) if unmapped else "None"
        print(f"{idx:<3} | {gloss:<14} | {th:<10} | {unmapped_str:<15} | {sigml_status:<12}")

        results.append({
            "gloss": gloss,
            "two_handed": th,
            "tags": " ".join(tags),
            "unmapped": unmapped,
            "sigml_valid": sigml_status == "✅ PASS",
            "xml_length": xml_len
        })

    print("-" * 75)
    print(f"\nSUMMARY RESULTS:")
    print(f"   Total Signs Tested:  {len(gloss_dict)}")
    print(f"   Passed SiGML Validation:  {passed_count} / {len(gloss_dict)} ({passed_count/len(gloss_dict)*100:.1f}%)")
    print(f"   Failed SiGML Validation:  {failed_count} / {len(gloss_dict)}")
    print("="*75)

if __name__ == "__main__":
    verify_all()
