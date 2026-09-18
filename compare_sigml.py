#!/usr/bin/env python3
"""
Side-by-Side SiGML Comparison Tool (Ground Truth vs V6 Hybrid Prediction).
Usage:
  python compare_sigml.py 1007_RAD1B.mp4
  python compare_sigml.py 1001_GEWICHT1
  python compare_sigml.py --list
  python compare_sigml.py --benchmark-summary
"""

import os
import sys
import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DGS_DIR = os.path.join(ROOT_DIR, "downloaded_dgs_dataset")
GT_SIGML_DIR = os.path.join(DGS_DIR, "sigml")
VIDEOS_DIR = os.path.join(DGS_DIR, "videos")
BENCHMARK_FILE = os.path.join(ROOT_DIR, "v6_benchmark_results.json")

def parse_sigml_tokens(sigml_content_or_path):
    """Extracts raw HamNoSys manual token tags from SiGML XML."""
    try:
        if os.path.isfile(sigml_content_or_path):
            tree = ET.parse(sigml_content_or_path)
            root = tree.getroot()
        else:
            root = ET.fromstring(sigml_content_or_path)
        
        manual = root.find(".//hamnosys_manual")
        if manual is not None:
            return [elem.tag for elem in manual]
        return []
    except Exception as e:
        return [f"Error: {e}"]

def format_side_by_side(lines_left, lines_right, title_left="GROUND TRUTH", title_right="V6 PREDICTED", col_width=42):
    """Renders two blocks of text side-by-side with ASCII border separators."""
    output = []
    sep = f"+{'-' * (col_width + 2)}+{'-' * (col_width + 2)}+"
    output.append(sep)
    output.append(f"| {title_left.center(col_width)} | {title_right.center(col_width)} |")
    output.append(sep)

    max_len = max(len(lines_left), len(lines_right))
    for i in range(max_len):
        l_text = lines_left[i] if i < len(lines_left) else ""
        r_text = lines_right[i] if i < len(lines_right) else ""
        # Truncate if line exceeds col_width
        l_text = (l_text[:col_width-3] + "...") if len(l_text) > col_width else l_text
        r_text = (r_text[:col_width-3] + "...") if len(r_text) > col_width else r_text
        output.append(f"| {l_text:<{col_width}} | {r_text:<{col_width}} |")

    output.append(sep)
    return "\n".join(output)

def get_stem(identifier):
    name = Path(identifier).stem
    if name.endswith("_hamer_output"):
        name = name.replace("_hamer_output", "")
    return name

def compare_single(stem):
    gt_file = os.path.join(GT_SIGML_DIR, f"{stem}.sigml")
    pred_file = os.path.join(ROOT_DIR, f"{stem}_hamer_output.sigml")

    # Check if prediction file exists; if not, check benchmark results
    pred_content = None
    if os.path.exists(pred_file):
        with open(pred_file, "r", encoding="utf-8") as f:
            pred_content = f.read()
    elif os.path.exists(BENCHMARK_FILE):
        with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
            bench = json.load(f)
        for item in bench.get("detailed_results", []):
            if item.get("video_file", "").startswith(stem) or item.get("gloss", "") == stem:
                # Reconstruct XML from hamnosys tokens
                tokens = item.get("hamnosys", "").split()
                xml_lines = [
                    '<?xml version="1.0" encoding="utf-8"?>',
                    '<sigml>',
                    f'  <hns_sign gloss="{stem}">',
                    '    <hamnosys_nonmanual/>',
                    '    <hamnosys_manual>'
                ]
                for t in tokens:
                    xml_lines.append(f'      <{t}/>')
                xml_lines.extend(['    </hamnosys_manual>', '  </hns_sign>', '</sigml>'])
                pred_content = "\n".join(xml_lines)
                with open(pred_file, "w", encoding="utf-8") as pf:
                    pf.write(pred_content)
                break

    # If still not found, run local pipeline
    if pred_content is None:
        video_path = os.path.join(VIDEOS_DIR, f"{stem}.mp4")
        if not os.path.exists(video_path):
            print(f"[-] Video file not found: {video_path}")
            return
        print(f"[*] Running V6 inference on {stem} to generate prediction...")
        from local_hamer_pipeline import LocalHaMeRPipeline
        pipeline = LocalHaMeRPipeline()
        res = pipeline.process_video(video_path)
        pred_content = res["sigml_xml"]
        with open(pred_file, "w", encoding="utf-8") as pf:
            pf.write(pred_content)

    gt_content = ""
    if os.path.exists(gt_file):
        with open(gt_file, "r", encoding="utf-8") as f:
            gt_content = f.read()
    else:
        gt_content = "<!-- Ground truth file not found -->"

    gt_tokens = parse_sigml_tokens(gt_content)
    pred_tokens = parse_sigml_tokens(pred_content)

    # Phonetic overlap analysis
    set_gt = set(gt_tokens)
    set_pred = set(pred_tokens)
    overlap = set_gt.intersection(set_pred)
    jaccard = (len(overlap) / len(set_gt.union(set_pred)) * 100) if set_gt.union(set_pred) else 0.0

    print("\n" + "=" * 90)
    print(f"   COMPARISON FOR SIGN: {stem}")
    print("=" * 90)
    print(f"  Ground Truth SiGML: [ {gt_file} ]")
    print(f"  V6 Predicted SiGML: [ {pred_file} ]")
    print("-" * 90)

    # 1. Phonetic Sequence Table
    print("\n[1] HAMNOSYS PHONETIC TOKENS COMPARISON:")
    table_tokens = format_side_by_side(
        gt_tokens, pred_tokens,
        title_left="GROUND TRUTH TOKENS",
        title_right="V6 PREDICTED TOKENS",
        col_width=40
    )
    print(table_tokens)
    print(f"  Phonetic Overlap: {len(overlap)} matching tokens: {sorted(list(overlap))}")
    print(f"  Jaccard Similarity: {jaccard:.1f}%\n")

    # 2. Raw SiGML XML Comparison
    print("[2] RAW SIGML XML CONTENT COMPARISON:")
    gt_lines = [l.strip() for l in gt_content.strip().splitlines() if l.strip()]
    pred_lines = [l.strip() for l in pred_content.strip().splitlines() if l.strip()]
    table_xml = format_side_by_side(
        gt_lines, pred_lines,
        title_left="GROUND TRUTH SIGML XML",
        title_right="V6 PREDICTED SIGML XML",
        col_width=40
    )
    print(table_xml)

    print("\n" + "=" * 90)
    print("  HOW TO PLAY BOTH IN CWASA WEBGL AVATAR PLAYER:")
    print("  1. Open http://localhost:8000/gui_clean.html in your browser.")
    print(f"  2. Paste the Ground Truth XML from: {gt_file}")
    print(f"  3. Paste the V6 Predicted XML from: {pred_file}")
    print("  4. Click 'Play' to watch the avatar perform the side-by-side sign!")
    print("=" * 90 + "\n")

def list_available():
    print("\n" + "=" * 80)
    print("  AVAILABLE SIGNS FOR GROUND TRUTH VS PREDICTION COMPARISON")
    print("=" * 80)
    
    # Check benchmark results if available
    bench_videos = []
    if os.path.exists(BENCHMARK_FILE):
        with open(BENCHMARK_FILE, "r") as f:
            b = json.load(f)
            bench_videos = [get_stem(x["video_file"]) for x in b.get("detailed_results", [])]

    print(f"Tested in V6 Benchmark ({len(bench_videos)} videos ready for instant comparison):")
    for i, v in enumerate(bench_videos):
        print(f"  {i+1:2d}. python compare_sigml.py {v}")

    print("\nOther available DGS ground truth signs (sample):")
    other_gt = [Path(f).stem for f in os.listdir(GT_SIGML_DIR) if f.endswith(".sigml") and Path(f).stem not in bench_videos][:10]
    for g in other_gt:
        print(f"   - python compare_sigml.py {g}")
    print("=" * 80 + "\n")

def show_benchmark_summary():
    if not os.path.exists(BENCHMARK_FILE):
        print(f"Benchmark file not found: {BENCHMARK_FILE}")
        return

    with open(BENCHMARK_FILE, "r") as f:
        data = json.load(f)

    print("\n" + "=" * 85)
    print("         V6 HYBRID BENCHMARK ACCURACY SUMMARY (15 DGS TEST VIDEOS)")
    print("=" * 85)
    print(f"  Handshape Accuracy (Top-1): {data.get('handshape_accuracy_pct')}%")
    print(f"  Movement Accuracy (Top-1):  {data.get('movement_accuracy_pct')}%")
    print(f"  Compound Accuracy:          {data.get('compound_accuracy_pct')}%")
    print(f"  SiGML XML Valid Rate:       {data.get('xml_validity_rate_pct')}%")
    print("-" * 85)
    print(f"{'#':<3} {'Video Stem':<25} {'GT Handshape':<16} {'Pred Handshape':<16} {'HS Match':<9} {'MV Match':<9}")
    print("-" * 85)

    for i, row in enumerate(data.get("detailed_results", [])):
        stem = get_stem(row["video_file"])
        gt_h = row.get("gt_handshape", "")
        pred_h = row.get("pred_handshape", "")
        hs_m = "YES" if row.get("hs_match") else "NO"
        mv_m = "YES" if row.get("mv_match") else "NO"
        print(f"{i+1:<3} {stem:<25} {gt_h:<16} {pred_h:<16} {hs_m:<9} {mv_m:<9}")

    print("=" * 85 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Side-by-side comparison of Ground Truth and Predicted SiGML.")
    parser.add_argument("target", nargs="?", help="Video filename, path, or sign name (e.g., 1007_RAD1B)")
    parser.add_argument("--list", "-l", action="store_true", help="List available signs for comparison")
    parser.add_argument("--summary", "-s", action="store_true", help="Show full benchmark summary table")

    args = parser.parse_args()

    if args.list:
        list_available()
    elif args.summary:
        show_benchmark_summary()
    elif args.target:
        stem = get_stem(args.target)
        compare_single(stem)
    else:
        parser.print_help()
        print("\nQuick example:")
        print("  python compare_sigml.py 1007_RAD1B")
        print("  python compare_sigml.py 1001_GEWICHT1")
        print("  python compare_sigml.py --summary\n")
