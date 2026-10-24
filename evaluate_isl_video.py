import os
import sys
import argparse
import xml.etree.ElementTree as ET

# Ensure integration pipeline can be imported
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTEGRATION_DIR = os.path.join(BASE_DIR, "Integration-20260706T062240Z-3-001", "Integration")
if INTEGRATION_DIR not in sys.path:
    sys.path.insert(0, INTEGRATION_DIR)

# Suppress verbose warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
import warnings
warnings.filterwarnings("ignore")

def extract_sigml_tokens(sigml_path):
    """Extract HamNoSys token tags from the <hamnosys_manual> section of a SiGML file."""
    if not os.path.exists(sigml_path):
        return []
    try:
        tree = ET.parse(sigml_path)
        root = tree.getroot()
        manual = root.find(".//hamnosys_manual")
        if manual is not None:
            return [elem.tag for elem in manual]
        # Fallback to any child tags
        return [elem.tag for elem in root.iter() if elem.tag.startswith("ham")]
    except Exception as e:
        print(f"[Warning] Failed to parse {sigml_path}: {e}")
        return []

def find_matching_ground_truth(video_name, sign_dir):
    """Find the corresponding .sigml ground truth file in SignFiles."""
    base = os.path.splitext(os.path.basename(video_name))[0].lower().strip()
    
    # 1. Exact match (e.g., 'hello' -> 'hello.sigml')
    exact_path = os.path.join(sign_dir, f"{base}.sigml")
    if os.path.exists(exact_path):
        return exact_path
    
    # 2. Check parts if separated by ' or ' (e.g., '0 or Zero' -> '0.sigml' or 'zero.sigml')
    if " or " in base:
        parts = [p.strip() for p in base.split(" or ")]
        for p in parts:
            p_clean = p.split()[0] if p else ""
            candidate1 = os.path.join(sign_dir, f"{p}.sigml")
            candidate2 = os.path.join(sign_dir, f"{p_clean}.sigml")
            if os.path.exists(candidate1):
                return candidate1
            if os.path.exists(candidate2):
                return candidate2
                
    # 3. Check for base file ignoring extra words
    first_word = base.split()[0]
    candidate = os.path.join(sign_dir, f"{first_word}.sigml")
    if os.path.exists(candidate):
        return candidate
        
    return None

def compute_metrics(gt_tokens, pred_tokens):
    """Compute precision, recall, and F1 between ground truth and predicted tokens."""
    gt_set = set(gt_tokens)
    pred_set = set(pred_tokens)
    common = gt_set.intersection(pred_set)
    
    precision = len(common) / len(pred_set) if pred_set else 0.0
    recall = len(common) / len(gt_set) if gt_set else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "common": sorted(list(common)),
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def list_available_matches(isl_dir, sign_dir):
    """List all ISL videos that have matching ground truth in SignFiles."""
    matched = []
    for f in sorted(os.listdir(isl_dir)):
        if f.lower().endswith(".mp4"):
            gt_file = find_matching_ground_truth(f, sign_dir)
            if gt_file:
                matched.append((f, os.path.basename(gt_file)))
    return matched

def main():
    parser = argparse.ArgumentParser(
        description="Run HamNoSys pipeline on an ISL video and evaluate against SignFiles ground truth."
    )
    parser.add_argument(
        "video", nargs="?", default="Hello.mp4",
        help="Name or path of video in ISL_videos (default: Hello.mp4)"
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Path for annotated output video"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all ISL videos that have matching ground truth in SignFiles"
    )
    args = parser.parse_args()

    isl_dir = os.path.join(BASE_DIR, "ISL_videos")
    sign_dir = os.path.join(BASE_DIR, "SignFiles")

    if args.list:
        matches = list_available_matches(isl_dir, sign_dir)
        print(f"\nFound {len(matches)} ISL videos with matching ground truth in SignFiles:")
        print("-" * 65)
        print(f"{'ISL Video':<35} | {'Ground Truth SiGML':<25}")
        print("-" * 65)
        for v, s in matches:
            print(f"{v:<35} | {s:<25}")
        print("-" * 65)
        return

    # Resolve video path
    video_input = args.video
    if os.path.isabs(video_input) and os.path.exists(video_input):
        video_path = video_input
    elif os.path.exists(os.path.join(isl_dir, video_input)):
        video_path = os.path.join(isl_dir, video_input)
    elif os.path.exists(video_input):
        video_path = os.path.abspath(video_input)
    else:
        print(f"Error: Video not found at '{video_input}' or in '{isl_dir}'.")
        print("Tip: Use '--list' to view all available ISL videos.")
        sys.exit(1)

    video_name = os.path.basename(video_path)
    gt_file = find_matching_ground_truth(video_name, sign_dir)

    print("\n" + "=" * 65)
    print("      ISL VIDEO HAMNOSYS PIPELINE & GROUND TRUTH EVALUATION")
    print("=" * 65)
    print(f"  Target Video:       {video_name}")
    print(f"  Video Path:         {video_path}")
    if gt_file:
        print(f"  Ground Truth SiGML: {os.path.basename(gt_file)} ({gt_file})")
    else:
        print("  Ground Truth SiGML: None found in SignFiles")
    print("=" * 65)

    # Output video path
    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        base = os.path.splitext(video_name)[0]
        output_path = os.path.join(os.path.dirname(video_path), f"{base}_output.mp4")

    # Load Ground Truth
    gt_tokens = []
    if gt_file:
        gt_tokens = extract_sigml_tokens(gt_file)
        print(f"\n[Ground Truth Tokens] ({len(gt_tokens)} tokens):")
        print("  " + " ".join(gt_tokens))

    # Run Prediction Pipeline
    print("\nRunning feature extraction & HamNoSys pipeline...", flush=True)
    from integration_pipeline import process_video
    
    result = process_video(video_path, output_path)
    pred_str = result.get("hamnosys", "").strip()
    pred_tokens = pred_str.split() if pred_str else []

    print("\n[Predicted HamNoSys Tokens] (" + str(len(pred_tokens)) + " tokens):")
    print("  " + (pred_str if pred_str else "<None>"))
    print(f"\n[Annotated Video Saved] -> {result.get('output_video', output_path)}")

    # Evaluation Metrics
    if gt_tokens and pred_tokens:
        metrics = compute_metrics(gt_tokens, pred_tokens)
        print("\n" + "=" * 65)
        print("                      EVALUATION RESULTS")
        print("=" * 65)
        print(f"  Matching Tokens:  {len(metrics['common'])} / {len(set(gt_tokens))} ground truth types")
        print(f"  Common Tokens:    {', '.join(metrics['common']) if metrics['common'] else 'None'}")
        print(f"  Token Precision:  {metrics['precision'] * 100:.2f}%")
        print(f"  Token Recall:     {metrics['recall'] * 100:.2f}%")
        print(f"  F1 Score:         {metrics['f1'] * 100:.2f}%")
        print("=" * 65)
    elif not gt_tokens:
        print("\n[Notice] No ground truth file found for this sign to compute quantitative accuracy.")
    print("\nDone!\n")

if __name__ == "__main__":
    main()
