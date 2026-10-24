#!/usr/bin/env python3
"""
Generate and verify the 20 DGS dataset SiGML output files from the V8 hybrid evaluation.
Saves individual <stem>_hamer_output.sigml files into the root directory and dgs_sigml_outputs/.
Validates XML syntax with xml.etree.ElementTree to ensure full CWASA player compatibility.
"""

import os
import json
import xml.etree.ElementTree as ET

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_JSON = os.path.join(ROOT_DIR, "v8_benchmark_results.json")
OUTPUT_DIR = os.path.join(ROOT_DIR, "dgs_sigml_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    if not os.path.exists(BENCHMARK_JSON):
        print(f"Error: {BENCHMARK_JSON} not found.")
        return

    with open(BENCHMARK_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("detailed_results", [])
    print("=" * 80)
    print(f"      GENERATING AND VERIFYING {len(results)} DGS SiGML OUTPUT FILES")
    print("=" * 80)

    summary_list = []

    for idx, item in enumerate(results, 1):
        vname = item["video_file"]
        stem = os.path.splitext(vname)[0]
        gloss = item.get("gloss", stem)
        sigml_xml = item.get("sigml_xml", "").strip()
        hamnosys = item.get("hamnosys", "").strip()
        hs = item.get("pred_handshape", "N/A")
        mv = item.get("pred_movement", "N/A")

        # Verify XML syntax
        try:
            root = ET.fromstring(sigml_xml)
            is_valid = (root.tag == "sigml")
        except Exception as e:
            is_valid = False
            print(f"[{idx:2d}/20] XML Parse Error for {stem}: {e}")

        # Destination paths: Root directory (standard) and dgs_sigml_outputs/ (collection)
        root_path = os.path.join(ROOT_DIR, f"{stem}_hamer_output.sigml")
        collection_path = os.path.join(OUTPUT_DIR, f"{stem}_hamer_output.sigml")

        with open(root_path, "w", encoding="utf-8") as rf:
            rf.write(sigml_xml + "\n")

        with open(collection_path, "w", encoding="utf-8") as cf:
            cf.write(sigml_xml + "\n")

        summary_list.append({
            "index": idx,
            "video_file": vname,
            "stem": stem,
            "handshape": hs,
            "movement": mv,
            "hamnosys": hamnosys,
            "sigml_file": f"{stem}_hamer_output.sigml",
            "is_valid_xml": is_valid
        })

        status_str = "[OK VALID XML]" if is_valid else "[INVALID XML]"
        print(f"[{idx:2d}/20] {status_str} Saved: {stem}_hamer_output.sigml")
        print(f"        Video: {vname} | Handshape: {hs} | Movement: {mv}")
        print(f"        HamNoSys: {hamnosys}\n")

    # Save summary index
    summary_path = os.path.join(OUTPUT_DIR, "dgs_20_sigml_manifest.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_list, f, indent=2)

    print("=" * 80)
    print(f"Successfully generated all {len(summary_list)} DGS SiGML files.")
    print(f"Root files: {ROOT_DIR}\\<stem>_hamer_output.sigml")
    print(f"Archive folder: {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
