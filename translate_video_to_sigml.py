#!/usr/bin/env python3
"""
End-to-End ISL Video to HamNoSys & SiGML XML Translator
======================================================
Usage:
    python translate_video_to_sigml.py "ISL_videos/testing/Police.mp4"
    python translate_video_to_sigml.py "path/to/my_video.mp4" --output "output.sigml"
"""

import os
import sys
import argparse
from pathlib import Path

# Setup paths to import from Integration directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
INTEGRATION_DIR = os.path.join(ROOT_DIR, "Integration-20260706T062240Z-3-001", "Integration")
if INTEGRATION_DIR not in sys.path:
    sys.path.insert(0, INTEGRATION_DIR)

from pipeline_v3_hybrid import get_v3_pipeline

def main():
    parser = argparse.ArgumentParser(
        description="Translate an Indian Sign Language (ISL) video to HamNoSys and CWASA SiGML XML."
    )
    parser.add_argument(
        "video",
        type=str,
        help="Path to the input ISL MP4 video file."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to save the generated SiGML XML file (default: <video_name>.sigml)."
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="Open CWASA avatar viewer in default browser after generation."
    )

    args = parser.parse_args()
    video_path = os.path.abspath(args.video)

    if not os.path.exists(video_path):
        print(f"\n[Error] Video file not found: {video_path}")
        sys.exit(1)

    stem = Path(video_path).stem
    out_sigml_path = os.path.abspath(args.output) if args.output else os.path.join(ROOT_DIR, f"{stem}_output.sigml")
    out_hns_path = os.path.join(ROOT_DIR, f"{stem}_hamnosys.txt")

    print("\n" + "=" * 65)
    print("  ISL-to-SiGML Hybrid Neural Translation Pipeline")
    print(f"  Input Video : {video_path}")
    print(f"  Target SiGML: {out_sigml_path}")
    print("=" * 65 + "\n")

    print("[1/3] Initializing DWPose GPU Tracker & V4 Attention-BiGRU...")
    engine = get_v3_pipeline()

    print(f"[2/3] Processing frames and extracting kinematics...")
    hns_tokens, sigml_xml = engine.process_video(video_path)

    print(f"[3/3] Translation complete!")
    print("\n" + "-" * 65)
    print("PREDICTED HAMNOSYS SEQUENCE:")
    print("-" * 65)
    print(hns_tokens)
    print("-" * 65 + "\n")

    # Save SiGML and HamNoSys files
    with open(out_sigml_path, "w", encoding="utf-8") as f:
        f.write(sigml_xml)

    with open(out_hns_path, "w", encoding="utf-8") as f:
        f.write(hns_tokens + "\n")

    print(f"[OK] Saved SiGML XML  -> {out_sigml_path}")
    print(f"[OK] Saved HamNoSys   -> {out_hns_path}")

    print("\n" + "=" * 65)
    print("  SiGML XML Preview (First 8 lines)")
    print("=" * 65)
    lines = sigml_xml.strip().split("\n")
    for line in lines[:8]:
        print(f"  {line}")
    if len(lines) > 8:
        print(f"  ... ({len(lines) - 8} more lines)")
    print("=" * 65)

    if args.view:
        import webbrowser
        viewer_html = os.path.join(ROOT_DIR, "UI", "index.html")
        if os.path.exists(viewer_html):
            print(f"\nOpening CWASA Avatar Viewer: {viewer_html}")
            webbrowser.open(f"file:///{viewer_html}")

if __name__ == "__main__":
    main()
