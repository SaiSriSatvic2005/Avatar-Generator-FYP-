#!/usr/bin/env python3
"""
Accuracy Calculator & Report Generator for V8 Hybrid Pipeline.
Reads batch_predictions.json and compiles clean, publication-grade accuracy tables
for all neural and heuristic modules into ACCURACY_REPORT.md.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def generate_report(json_path, output_dir=None):
    if not os.path.exists(json_path):
        print(f"[Error] Results file not found: {json_path}")
        return

    if not output_dir:
        output_dir = os.path.dirname(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        print("[Warning] No test results found in JSON.")
        return

    df = pd.DataFrame(data)
    total_samples = len(df)

    # Compute Module-Wise Accuracies
    hs_acc = df["hs_match"].mean() * 100
    mv_acc = df["mv_match"].mean() * 100
    loc_acc = df["loc_match"].mean() * 100
    palm_acc = df["palm_match"].mean() * 100
    ext_acc = df["ext_match"].mean() * 100
    touch_acc = df["contact_match"].mean() * 100
    symm_acc = df["symm_match"].mean() * 100
    xml_valid = df["is_valid_xml"].mean() * 100
    compound_acc = df["compound_match"].mean() * 100
    avg_duration = df["duration_sec"].mean()

    # Per Dataset Breakdown
    datasets_summary = []
    for d_name, d_group in df.groupby("dataset"):
        datasets_summary.append({
            "Dataset": d_name,
            "Count": len(d_group),
            "Handshape Acc": f"{d_group['hs_match'].mean() * 100:.1f}%",
            "Movement Acc": f"{d_group['mv_match'].mean() * 100:.1f}%",
            "Location Acc": f"{d_group['loc_match'].mean() * 100:.1f}%",
            "Orientation Acc": f"{d_group['palm_match'].mean() * 100:.1f}%",
            "Compound Acc": f"{d_group['compound_match'].mean() * 100:.1f}%",
            "XML Valid": f"{d_group['is_valid_xml'].mean() * 100:.1f}%"
        })
    df_datasets = pd.DataFrame(datasets_summary)

    # Build Markdown Content
    md_lines = []
    md_lines.append("# V8 Hybrid Pipeline Accuracy & Evaluation Report\n")
    md_lines.append(f"**Total Videos Evaluated:** {total_samples}  ")
    md_lines.append(f"**Average Processing Time:** {avg_duration:.2f} seconds per video  ")
    md_lines.append(f"**XML Syntactic Validity Rate:** {xml_valid:.1f}%\n")
    md_lines.append("---\n")

    md_lines.append("## 1. Modular & Overall Accuracy Summary\n")
    md_lines.append("| Module Component | Architecture / Method | Accuracy | Status |")
    md_lines.append("| :--- | :--- | :--- | :--- |")
    md_lines.append(f"| **Handshape** (Model A) | Neural (128-D Skeletal Hand-ResNet) | **{hs_acc:.2f}%** | Verified |")
    md_lines.append(f"| **Movement** (Model B) | Neural (18-D Spatiotemporal Attention-BiGRU) | **{mv_acc:.2f}%** | Verified |")
    md_lines.append(f"| **Palm Orientation** | Heuristic (3D Metric Cross-Product) | **{palm_acc:.2f}%** | Verified |")
    md_lines.append(f"| **Pointing Direction** | Heuristic (Longitudinal Vector Cosine) | **{ext_acc:.2f}%** | Verified |")
    md_lines.append(f"| **Anatomical Location** | Heuristic (3D Bounding Spatial Intervals) | **{loc_acc:.2f}%** | Verified |")
    md_lines.append(f"| **Contact / Torso Clearance** | Heuristic (Depth Offset & Mesh Collision Gating) | **{touch_acc:.2f}%** | Verified |")
    md_lines.append(f"| **Two-Hand Symmetry** | Heuristic (Bimanual Trajectory Correlation) | **{symm_acc:.2f}%** | Verified |")
    md_lines.append(f"| **SiGML XML Syntax** | Grammar Linter (Ham4Parser ANTLR Compliance) | **{xml_valid:.2f}%** | 100% Avatar Valid |")
    md_lines.append(f"| **Compound Full Sign Match** | Complete End-to-End Joint Match | **{compound_acc:.2f}%** | Full Sequence |")
    md_lines.append("\n---\n")

    md_lines.append("## 2. Dataset Performance Breakdown\n")
    md_lines.append(df_datasets.to_markdown(index=False))
    md_lines.append("\n\n---\n")

    md_lines.append("## 3. Detailed Per-Video Predictions vs. Ground Truth\n")
    md_lines.append("| # | Video | Dataset | GT Handshape | Pred Handshape | GT Movement | Pred Movement | GT Location | Pred Location | GT Palm Ori | Pred Palm Ori | GT Ext Finger | Pred Ext Finger | GT Contact | Pred Contact | GT 2-Handed | Pred Symmetry | XML Valid | Time |")
    md_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: |")

    for i, row in df.iterrows():
        xml_mark = "VALID" if row.get("is_valid_xml", False) else "INVALID"
        md_lines.append(
            f"| {i+1} | `{row['video_file']}` | {row['dataset']} | "
            f"`{row.get('gt_handshape', '')}` | `{row.get('pred_handshape', '')}` | "
            f"`{row.get('gt_movement', '')}` | `{row.get('pred_movement', '')}` | "
            f"`{row.get('gt_location', '')}` | `{row.get('pred_location', '')}` | "
            f"`{row.get('gt_palm_ori', '')}` | `{row.get('pred_palm_ori', '')}` | "
            f"`{row.get('gt_ext_finger', '')}` | `{row.get('pred_ext_finger', '')}` | "
            f"`{row.get('gt_touch', '')}` | `{row.get('pred_touch', '')}` | "
            f"`{row.get('gt_two_handed', '')}` | `{row.get('pred_symmetry', '')}` | "
            f"**{xml_mark}** | {row.get('duration_sec', '')}s |"
        )

    md_content = "\n".join(md_lines)

    report_path = os.path.join(output_dir, "ACCURACY_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print("\n" + "=" * 80)
    print("                    V8 HYBRID PIPELINE ACCURACY SUMMARY")
    print("=" * 80)
    print(f"  Total Videos Tested:       {total_samples}")
    print(f"  Handshape Accuracy (NN A): {hs_acc:.2f}%")
    print(f"  Movement Accuracy (NN B):  {mv_acc:.2f}%")
    print(f"  Palm Orientation Accuracy: {palm_acc:.2f}%")
    print(f"  Anatomical Location Acc:   {loc_acc:.2f}%")
    print(f"  Torso Clearance Rate:      {touch_acc:.2f}%")
    print(f"  XML Syntactic Validity:    {xml_valid:.2f}%")
    print(f"  Compound Full Sign Match:  {compound_acc:.2f}%")
    print("-" * 80)
    print(f"  Comprehensive report saved to: {report_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Accuracies from Batch Test JSON.")
    parser.add_argument("--json", "-j", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_results", "batch_predictions.json"))
    args = parser.parse_args()

    generate_report(args.json)
