#!/usr/bin/env python
"""
HamNoSys Local Runner
=====================
Run the full sign-language to HamNoSys pipeline on a local video file.

Usage:
    python run_local.py                     # uses Prompt_1.mp4
    python run_local.py path/to/video.mp4   # custom video
    python run_local.py video.mp4 -o out.mp4  # custom output path
"""

import os
import sys

# =====================================================
# SUPPRESS ALL WARNINGS (must be done BEFORE any imports)
# =====================================================

# Suppress C++ level warnings from MediaPipe / TensorFlow Lite
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

# Suppress Python-level warnings
import warnings
warnings.filterwarnings("ignore")

# Redirect stderr to suppress C++ W0000 messages that bypass Python warnings
import io
_real_stderr = sys.stderr
sys.stderr = io.StringIO()

# Now import absl to suppress its logging
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
    absl.logging.set_stderrthreshold(absl.logging.FATAL)
except ImportError:
    pass

import argparse
import time

# Ensure we can import from this directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


def main():
    parser = argparse.ArgumentParser(
        description="Run HamNoSys pipeline on a sign language video"
    )
    parser.add_argument(
        "video", nargs="?", default=os.path.join(_SCRIPT_DIR, "Prompt_1.mp4"),
        help="Path to input video file (default: Prompt_1.mp4)"
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output annotated video path (default: <input>_hamnosys_output.mp4)"
    )
    args = parser.parse_args()

    video_path = os.path.abspath(args.video)

    if not os.path.exists(video_path):
        # Restore stderr for error messages
        sys.stderr = _real_stderr
        print(f"ERROR: Video file not found: {video_path}")
        sys.exit(1)

    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(os.path.dirname(video_path), f"{base}_hamnosys_output.mp4")

    print("=" * 60)
    print("  HamNoSys Pipeline - Local Runner")
    print("=" * 60)
    print(f"  Input:  {video_path}")
    print(f"  Output: {output_path}")
    print("=" * 60)

    start = time.time()

    # Import pipeline (this triggers MediaPipe/sklearn model loading)
    print("\n  Loading modules...", flush=True)
    from integration_pipeline import process_video
    print("  Modules loaded.\n", flush=True)

    result = process_video(video_path, output_path)

    elapsed = time.time() - start

    # Restore stderr before final output
    sys.stderr = _real_stderr

    print("\n[DONE]")
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  HamNoSys Code: {result['hamnosys']}")
    print(f"  Output Video:  {result['output_video']}")
    print(f"  Time:          {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
