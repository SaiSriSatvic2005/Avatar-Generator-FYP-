# --- Fix Windows DLL conflict: MediaPipe must be imported first before numpy/pandas/torch ---
import mediapipe as mp
# ----------------------------------------------------------------------------------------
#!/usr/bin/env python3
"""
Bulk Testing Runner for V8 Hybrid Pipeline (DGS & ISL Datasets)
Executes video test sets, caches intermediate frame traces, saves predicted SiGML files,
and records ground-truth comparisons in an organized directory structure.

Usage:
  python run_bulk_testing.py --dataset dgs --limit 5
  python run_bulk_testing.py --dataset isl --limit 5
  python run_bulk_testing.py --dataset all
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Setup paths relative to Active_Hybrid_Pipeline
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_ENGINE_DIR = os.path.join(BASE_DIR, "pipeline_engine")
HEURISTICS_DIR = os.path.join(BASE_DIR, "heuristics")
HAMER_ENGINE_DIR = os.path.join(BASE_DIR, "hamer_3d_engine")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

for p in [BASE_DIR, PIPELINE_ENGINE_DIR, HEURISTICS_DIR, HAMER_ENGINE_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Fix NumPy / Python 3.12 compatibility
import inspect
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

for _attr, _type in [
    ('bool', bool), ('int', int), ('float', float), ('complex', complex),
    ('object', object), ('unicode', str), ('str', str)
]:
    if _attr not in np.__dict__:
        setattr(np, _attr, _type)

from local_hamer_pipeline import LocalHaMeRPipeline
from run_pipeline import print_frame_table, print_decision_trace


def load_dgs_test_manifest():
    """Loads held-out DGS test videos with matching ground truth."""
    workspace_root = os.path.dirname(BASE_DIR)
    csv_candidates = [
        os.path.join(BASE_DIR, "FYP-DataSet", "DGS", "dgs_clean_ground_truth.csv"),
        os.path.join(workspace_root, "FYP-DataSet", "DGS", "dgs_clean_ground_truth.csv"),
        os.path.join(BASE_DIR, "evaluation", "dgs_clean_ground_truth.csv"),
        os.path.join(workspace_root, "downloaded_dgs_dataset", "dgs_clean_ground_truth.csv"),
    ]
    csv_path = next((p for p in csv_candidates if os.path.exists(p)), None)
    if not csv_path:
        raise FileNotFoundError("DGS clean ground truth CSV not found.")

    df = pd.read_csv(csv_path)

    # Search for video paths across available folders (FYP-DataSet prioritized)
    video_dirs = [
        os.path.join(BASE_DIR, "FYP-DataSet", "DGS", "reference_videos"),
        os.path.join(BASE_DIR, "FYP-DataSet", "DGS", "videos"),
        os.path.join(workspace_root, "FYP-DataSet", "DGS", "reference_videos"),
        os.path.join(workspace_root, "FYP-DataSet", "DGS", "videos"),
        os.path.join(BASE_DIR, "evaluation", "sample_videos"),
        os.path.join(workspace_root, "downloaded_dgs_dataset", "videos"),
    ]

    records = []
    for _, row in df.iterrows():
        vname = str(row["video_file"])
        for vdir in video_dirs:
            full_path = os.path.join(vdir, vname)
            if os.path.exists(full_path):
                records.append({
                    "dataset": "DGS",
                    "stem": str(row["stem"]),
                    "video_file": vname,
                    "video_path": full_path,
                    "sigml_file": str(row.get("sigml_file", f"{row['stem']}.sigml")),
                    "gt_handshape": str(row.get("base_handshape", "")),
                    "gt_movement": str(row.get("movement", "")),
                    "gt_location": str(row.get("location", "")),
                    "gt_palm_ori": str(row.get("palm_orientation", "")),
                    "gt_ext_finger": str(row.get("extended_finger", "")),
                    "gt_has_touch": bool(row.get("has_touch", False)),
                    "gt_is_two_handed": bool(row.get("is_two_handed", False))
                })
                break
    return records


def load_isl_test_manifest():
    """Loads held-out ISL test videos with matching ground truth."""
    workspace_root = os.path.dirname(BASE_DIR)
    csv_candidates = [
        os.path.join(BASE_DIR, "FYP-DataSet", "ISL", "isl_ground_truth_labels.csv"),
        os.path.join(workspace_root, "FYP-DataSet", "ISL", "isl_ground_truth_labels.csv"),
        os.path.join(workspace_root, "Integration-20260706T062240Z-3-001", "Integration", "isl_ground_truth_labels.csv"),
    ]
    csv_path = next((p for p in csv_candidates if os.path.exists(p)), None)
    if not csv_path:
        raise FileNotFoundError("ISL ground truth CSV not found.")

    df = pd.read_csv(csv_path)

    # Search for videos in ISL testing directory first, then main directory
    testing_dir = os.path.join(BASE_DIR, "FYP-DataSet", "ISL", "videos", "testing")
    if not os.path.exists(testing_dir):
        testing_dir = os.path.join(workspace_root, "FYP-DataSet", "ISL", "videos", "testing")
    main_dir = os.path.join(BASE_DIR, "FYP-DataSet", "ISL", "videos")
    if not os.path.exists(main_dir):
        main_dir = os.path.join(workspace_root, "FYP-DataSet", "ISL", "videos")
    if not os.path.exists(main_dir):
        main_dir = os.path.join(workspace_root, "ISL_videos")

    records = []
    for _, row in df.iterrows():
        vname = str(row["video_file"])
        cand_path = os.path.join(testing_dir, vname)
        if not os.path.exists(cand_path):
            cand_path = os.path.join(main_dir, vname)

        if os.path.exists(cand_path):
            records.append({
                "dataset": "ISL",
                "stem": Path(vname).stem,
                "video_file": vname,
                "video_path": cand_path,
                "sigml_file": str(row.get("sigml_file", "")),
                "gt_handshape": str(row.get("handshape", "")),
                "gt_movement": str(row.get("movement", "")),
                "gt_location": str(row.get("location", "")),
                "gt_palm_ori": str(row.get("palm_ori", "")),
                "gt_ext_finger": str(row.get("ext_finger", "")),
                "gt_has_touch": bool(str(row.get("contact", "")).lower() == "hamtouch"),
                "gt_is_two_handed": bool(row.get("two_handed", False))
            })
    return records


def run_bulk(dataset_type="all", limit=None, output_dir=None, dgs_limit=None, isl_limit=None, append=True):
    if not output_dir:
        output_dir = os.path.join(BASE_DIR, "test_results")

    # Create organized directory structure
    sigml_out_dir = os.path.join(output_dir, "predictions_sigml")
    traces_out_dir = os.path.join(output_dir, "traces")
    os.makedirs(sigml_out_dir, exist_ok=True)
    os.makedirs(traces_out_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "batch_predictions.json")

    results_data = []
    already_processed = set()
    if append and os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                results_data = json.load(f)
            already_processed = {r["video_file"] for r in results_data if "video_file" in r}
            print(f"[Append Mode] Loaded {len(results_data)} existing records. Skipping already processed videos.")
        except Exception as e:
            print(f"[Append Mode Warning] Could not load existing results: {e}")

    test_items = []
    if dataset_type.lower() in ["dgs", "all"]:
        dgs_items = load_dgs_test_manifest()
        if dgs_limit and dgs_limit > 0:
            dgs_items = dgs_items[:dgs_limit]
        print(f"[Manifest] Loaded {len(dgs_items)} DGS testing videos with ground truth.")
        test_items.extend(dgs_items)

    if dataset_type.lower() in ["isl", "all"]:
        isl_items = load_isl_test_manifest()
        if isl_limit and isl_limit > 0:
            isl_items = isl_items[:isl_limit]
        print(f"[Manifest] Loaded {len(isl_items)} ISL testing videos with ground truth.")
        test_items.extend(isl_items)

    if limit and limit > 0:
        test_items = test_items[:limit]

    if already_processed:
        test_items = [item for item in test_items if item["video_file"] not in already_processed]
        print(f"[Append Mode] {len(test_items)} new videos remaining to process.")

    print("\n" + "=" * 90)
    print(f"       STARTING BULK TESTING SUITE ON {len(test_items)} GROUND-TRUTH VIDEOS")
    print("=" * 90)
    print(f"  Target Datasets:  {dataset_type.upper()}")
    print(f"  Results Storage:  {output_dir}")
    print(f"  SiGML Output Dir: {sigml_out_dir}")
    print(f"  Traces Dir:       {traces_out_dir}")
    print("=" * 90 + "\n")

    print("[Pipeline] Initializing Complete V8 Hybrid HaMeR Engine...")
    pipeline = LocalHaMeRPipeline()

    start_time_all = time.time()

    for idx, item in enumerate(test_items, start=1):
        vpath = item["video_path"]
        vname = item["video_file"]
        stem = item["stem"]
        ds = item["dataset"]

        print(f"\n[{idx:03d}/{len(test_items):03d}] Processing {ds}: {vname} ...")
        t0 = time.time()
        try:
            res = pipeline.process_video(vpath)
            duration = round(time.time() - t0, 2)

            # 1. Save Predicted SiGML XML
            sigml_path = os.path.join(sigml_out_dir, f"{ds}_{stem}_prediction.sigml")
            with open(sigml_path, "w", encoding="utf-8") as f:
                f.write(res["sigml_xml"])

            # 2. Save Human-Readable Text Trace
            trace_path = os.path.join(traces_out_dir, f"{ds}_{stem}_trace.txt")
            with open(trace_path, "w", encoding="utf-8") as f:
                # Capture terminal table and decision trace
                import io
                from contextlib import redirect_stdout
                buf = io.StringIO()
                with redirect_stdout(buf):
                    print(f"Video: {vname} ({ds}) | Processing Time: {duration}s")
                    if "frame_diagnostics" in res:
                        print_frame_table(res["frame_diagnostics"])
                    if "explainability" in res:
                        print_decision_trace(res)
                f.write(buf.getvalue())

            # 3. Compute Component Matches against Ground Truth
            gt_hs = item["gt_handshape"].strip().lower()
            pred_hs = str(res["handshape"]).strip().lower()
            hs_match = bool(gt_hs and gt_hs in pred_hs or pred_hs in gt_hs)

            gt_mv = item["gt_movement"].strip().lower()
            pred_mv = str(res["movement"]).strip().lower()
            mv_match = bool(gt_mv and (gt_mv == pred_mv or (gt_mv in pred_mv) or (pred_mv in gt_mv)))

            gt_loc = item["gt_location"].strip().lower()
            pred_loc = str(res["location"]).strip().lower()
            loc_match = bool(gt_loc and (gt_loc == pred_loc or gt_loc in pred_loc))

            gt_palm = item["gt_palm_ori"].strip().lower()
            pred_palm = str(res["palm_orientation"]).strip().lower()
            palm_match = bool(gt_palm and (gt_palm == pred_palm))

            gt_ext = item["gt_ext_finger"].strip().lower()
            pred_ext = str(res["extended_finger"]).strip().lower()
            ext_match = bool(gt_ext and (gt_ext == pred_ext))

            contact_match = (item["gt_has_touch"] == res["has_touch"])
            symm_match = (item["gt_is_two_handed"] == (res["symmetry"] == "hamsymmlr"))

            compound_match = (hs_match and mv_match and loc_match)

            record = {
                "dataset": ds,
                "video_file": vname,
                "stem": stem,
                "duration_sec": duration,
                "is_valid_xml": res["is_valid_xml"],
                # Handshape
                "gt_handshape": gt_hs,
                "pred_handshape": pred_hs,
                "hs_match": hs_match,
                # Movement
                "gt_movement": gt_mv,
                "pred_movement": pred_mv,
                "mv_match": mv_match,
                # Location
                "gt_location": gt_loc,
                "pred_location": pred_loc,
                "loc_match": loc_match,
                # Orientation
                "gt_palm_ori": gt_palm,
                "pred_palm_ori": pred_palm,
                "palm_match": palm_match,
                "gt_ext_finger": gt_ext,
                "pred_ext_finger": pred_ext,
                "ext_match": ext_match,
                # Contact & Symmetry
                "gt_touch": item["gt_has_touch"],
                "pred_touch": res["has_touch"],
                "contact_match": contact_match,
                "gt_two_handed": item["gt_is_two_handed"],
                "pred_symmetry": res["symmetry"],
                "symm_match": symm_match,
                # Overall
                "compound_match": compound_match,
                "hamnosys": res["hamnosys"],
                "sigml_file": sigml_path,
                "trace_file": trace_path
            }
            results_data.append(record)
            print(f"  -> HS Match: {hs_match} | MV Match: {mv_match} | Loc Match: {loc_match} | XML: {res['is_valid_xml']} ({duration}s)")

            # Save progress incrementally to json
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(results_data, f, indent=2)

        except Exception as e:
            print(f"  [Error] Failed to process {vname}: {e}")

    # 4. Save Structured JSON Database
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    total_time = round(time.time() - start_time_all, 2)
    print("\n" + "=" * 90)
    print(f"Bulk testing complete! Processed {len(results_data)} videos in {total_time}s.")
    print(f"Predictions JSON saved to: {json_path}")
    print("=" * 90)

    # 5. Automatically Generate Readable Markdown Accuracy Report
    from calculate_accuracy import generate_report
    generate_report(json_path, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Bulk Testing & Generate Accuracy Reports.")
    parser.add_argument("--dataset", "-d", choices=["dgs", "isl", "all"], default="all", help="Dataset to test")
    parser.add_argument("--limit", "-n", type=int, default=None, help="Limit number of videos to test")
    parser.add_argument("--dgs_limit", type=int, default=None, help="Limit number of DGS videos to test")
    parser.add_argument("--isl_limit", type=int, default=None, help="Limit number of ISL videos to test")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing batch_predictions.json instead of appending")
    parser.add_argument("--output_dir", "-o", default=None, help="Directory to store test results")
    args = parser.parse_args()

    run_bulk(
        dataset_type=args.dataset,
        limit=args.limit,
        dgs_limit=args.dgs_limit,
        isl_limit=args.isl_limit,
        append=not args.overwrite,
        output_dir=args.output_dir
    )
