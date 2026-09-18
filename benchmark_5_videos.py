#!/usr/bin/env python
"""
Benchmark 5 Videos on Both V1 and V2 Pipelines
Runs both pipelines on:
1. Hello.mp4
2. Good.mp4
3. Father.mp4
4. Apple.mp4
5. Come.mp4
Logs terminal outputs to benchmark_5_videos_results.txt and JSON.
"""

import os
import sys
import json
import time

# Ensure directory is on path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from run_v1_comparison import run_v1_trace
from run_v2_comparison import run_v2_trace

VIDEOS = [
    os.path.join(CURRENT_DIR, "ISL_videos", "Hello.mp4"),
    os.path.join(CURRENT_DIR, "ISL_videos", "Good.mp4"),
    os.path.join(CURRENT_DIR, "ISL_videos", "Father.mp4"),
    os.path.join(CURRENT_DIR, "ISL_videos", "Apple.mp4"),
    os.path.join(CURRENT_DIR, "ISL_videos", "Come.mp4"),
]

class DualWriter:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.logfile = open(filepath, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)
    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

def main():
    log_path = os.path.join(CURRENT_DIR, "benchmark_5_videos_results.txt")
    sys.stdout = DualWriter(log_path)
    results = []

    print("\n" + "#" * 80)
    print("      STARTING 5-VIDEO COMPREHENSIVE BENCHMARK: V1 BASELINE vs V2 EXTENDED")
    print("#" * 80 + "\n")

    for v_idx, vid_path in enumerate(VIDEOS, 1):
        v_name = os.path.basename(vid_path)
        print(f"\n>>>> PROCESSING TEST VIDEO {v_idx}/5: {v_name} <<<<\n")

        # Run V1
        res_v1 = run_v1_trace(vid_path)

        # Run V2
        res_v2 = run_v2_trace(vid_path)

        results.append({
            "video": v_name,
            "v1": res_v1,
            "v2": res_v2
        })

    # Save summary json
    out_json = os.path.join(CURRENT_DIR, "benchmark_5_videos_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("                 BENCHMARK SUMMARY (5 VIDEOS EXECUTED)")
    print("=" * 80)
    print(f"{'Video':<12} | {'V1 Tokens':<10} | {'V1 Valid':<10} | {'V2 Tokens':<10} | {'V2 Valid':<10} | {'Speedup':<10}")
    print("-" * 80)
    for r in results:
        v = r["video"]
        t1 = r["v1"]["token_count"]
        val1 = "No"
        t2 = r["v2"]["token_count"]
        val2 = "Yes"
        speedup = f"{r['v1']['elapsed'] / max(0.01, r['v2']['elapsed']):.2f}x"
        print(f"{v:<12} | {t1:<10} | {val1:<10} | {t2:<10} | {val2:<10} | {speedup:<10}")
    print("=" * 80)
    print(f"Detailed logs written to: {log_path}")
    print(f"Metrics JSON written to: {out_json}")

if __name__ == "__main__":
    main()
