#!/usr/bin/env python3
"""
Automated Downloader for RKMVERI Indian Sign Language (ISL) Video Dataset
=========================================================================
Target Source: https://cs.rkmvu.ac.in/~isl/
Cross-references the online 2,002-word dictionary against local ground truth
.sigml files in SignFiles/, and downloads all matching MP4 videos into ISL_videos/.

Features:
- Fast concurrent multithreaded downloads (ThreadPoolExecutor)
- Resumable: Skips videos already downloaded
- Produces a dataset manifest: isl_dataset_manifest.json
"""

import os
import re
import sys
import json
import glob
import time
import argparse
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORDS_JSON_URL = "https://cs.rkmvu.ac.in/~isl/assets/words.json"
VIDEO_BASE_URL = "https://cs.rkmvu.ac.in/~isl/assets/videos/"


def load_local_sigml_files(sign_dir):
    """Index all local ground truth .sigml files by normalized base name."""
    sigml_index = {}
    pattern = os.path.join(sign_dir, "*.sigml")
    files = glob.glob(pattern)
    for p in files:
        base = os.path.splitext(os.path.basename(p))[0].lower().strip()
        sigml_index[base] = p
    return sigml_index


def normalize_title_to_sigml(title, sigml_index):
    """
    Match an online dictionary title to a local .sigml file.
    Handles compound words, verbs [VEB], synonyms ('or'), and punctuation.
    """
    clean_title = title.lower().strip()

    # 1. Exact match
    if clean_title in sigml_index:
        return clean_title, sigml_index[clean_title]

    # 2. Strip bracketed annotations like [VEB]
    no_brackets = re.sub(r'\[.*?\]', '', clean_title).strip()
    if no_brackets in sigml_index:
        return no_brackets, sigml_index[no_brackets]

    # 3. Split synonyms (e.g. '0 or zero', 'eat or food', 'choose/select')
    parts = [p.strip() for p in re.split(r'\bor\b|/|,', no_brackets) if p.strip()]
    for p in parts:
        p_clean = re.sub(r'[^a-z0-9\-]', '', p)
        if p in sigml_index:
            return p, sigml_index[p]
        if p_clean in sigml_index:
            return p_clean, sigml_index[p_clean]

    # 4. Fallback to first word
    if parts:
        first_word = parts[0].split()[0] if parts[0].split() else ""
        first_clean = re.sub(r'[^a-z0-9\-]', '', first_word)
        if first_word in sigml_index:
            return first_word, sigml_index[first_word]
        if first_clean in sigml_index:
            return first_clean, sigml_index[first_clean]

    return None, None


def download_single_video(task):
    """Download a single video file from RKMVERI server with automatic retry."""
    item, sigml_name, sigml_path, out_dir = task
    file_name = item["file"]
    dest_path = os.path.join(out_dir, file_name)

    # Encode video filename for HTTP URL (handles spaces, brackets, special characters)
    quoted_name = urllib.parse.quote(file_name)
    download_url = VIDEO_BASE_URL + quoted_name

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ISL-Research-Pipeline"
    }

    # Retry up to 3 times on transient network drops
    last_err = None
    for attempt in range(1, 4):
        try:
            req = urllib.request.Request(download_url, headers=headers)
            with urllib.request.urlopen(req, timeout=25) as resp:
                if resp.status == 200:
                    with open(dest_path, "wb") as f_out:
                        f_out.write(resp.read())
                    return True, file_name, dest_path, None
                else:
                    last_err = f"HTTP Status {resp.status}"
        except Exception as e:
            last_err = str(e)
            time.sleep(1.0 * attempt)

    return False, file_name, None, last_err


def main():
    parser = argparse.ArgumentParser(
        description="Automated downloader for RKMVERI ISL videos matching SignFiles/ ground truth."
    )
    parser.add_argument(
        "--sign-dir", default=os.path.join(_BASE_DIR, "SignFiles"),
        help="Path to directory containing ground truth .sigml files"
    )
    parser.add_argument(
        "--out-dir", default=os.path.join(_BASE_DIR, "ISL_videos"),
        help="Destination directory for downloaded videos"
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Number of concurrent download threads (default: 8)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of new videos to download (useful for testing)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only display matching videos without downloading"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 70)
    print("      AUTOMATED RKMVERI ISL VIDEO DATASET HARVESTER")
    print("=" * 70)
    print(f"  Target Source:      {WORDS_JSON_URL}")
    print(f"  Ground Truth Dir:   {args.sign_dir}")
    print(f"  Destination Dir:    {args.out_dir}")
    print(f"  Concurrent Threads: {args.workers}")
    print("=" * 70)

    # 1. Load Local Ground Truths
    sigml_index = load_local_sigml_files(args.sign_dir)
    print(f"\n[1/4] Indexed {len(sigml_index)} local .sigml ground truth files.")

    # 2. Fetch Remote Dictionary from RKMVERI
    print(f"[2/4] Fetching remote dictionary index from RKMVERI...")
    try:
        req = urllib.request.Request(WORDS_JSON_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            remote_words = json.loads(resp.read().decode("utf-8"))
        print(f"      Successfully loaded {len(remote_words)} vocabulary entries from server.")
    except Exception as e:
        print(f"Error fetching remote dictionary: {e}")
        sys.exit(1)

    # 3. Match Remote Videos against Local Ground Truths
    existing_videos = set(os.listdir(args.out_dir)) if os.path.exists(args.out_dir) else set()

    already_downloaded = []
    to_download = []
    all_matched_pairs = []

    for item in remote_words:
        title = item.get("title", "")
        file_name = item.get("file", "")
        sigml_name, sigml_path = normalize_title_to_sigml(title, sigml_index)

        if sigml_name:
            pair_record = {
                "word_title": title,
                "video_file": file_name,
                "sigml_file": os.path.basename(sigml_path),
                "sigml_path": sigml_path
            }
            all_matched_pairs.append(pair_record)

            if file_name in existing_videos:
                already_downloaded.append((item, sigml_name, sigml_path))
            else:
                to_download.append((item, sigml_name, sigml_path, args.out_dir))

    print(f"\n[3/4] Dataset Matching Results:")
    print(f"  • Total Matched Dataset Capacity: {len(all_matched_pairs)} video-to-SiGML pairs")
    print(f"  • Already Downloaded in ISL_videos: {len(already_downloaded)} videos")
    print(f"  • New Videos Available to Download: {len(to_download)} videos")

    # Save manifest
    manifest_path = os.path.join(_BASE_DIR, "isl_dataset_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f_m:
        json.dump(all_matched_pairs, f_m, indent=2)
    print(f"  • Dataset Manifest Saved: {manifest_path}")

    if args.dry_run:
        print("\n[Dry Run] Sample 15 matches ready to download:")
        for itm, sname, _, _ in to_download[:15]:
            print(f"    - {itm['title']:<30} -> {sname}.sigml ({itm['file']})")
        print("\nDry-run complete. Run without --dry-run to start downloading.")
        return

    if not to_download:
        print("\nAll matching videos are already downloaded! Your dataset is 100% up to date.")
        return

    # Apply limit if requested
    if args.limit:
        to_download = to_download[:args.limit]
        print(f"\n[Limit Applied] Downloading first {len(to_download)} new videos.")

    # 4. Download Videos in Parallel
    print(f"\n[4/4] Starting concurrent download of {len(to_download)} videos...")
    start_time = time.time()
    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_single_video, task): task for task in to_download}
        total_tasks = len(futures)

        for i, future in enumerate(as_completed(futures), 1):
            success, file_name, dest_path, err = future.result()
            if success:
                success_count += 1
                status = "OK"
            else:
                fail_count += 1
                status = f"FAILED ({err})"

            # Print compact progress on single line
            pct = (i / total_tasks) * 100
            sys.stdout.write(f"\r  Progress: [{i}/{total_tasks}] ({pct:5.1f}%) | Success: {success_count} | Failed: {fail_count} | Last: {file_name[:25]:<25}")
            sys.stdout.flush()

    elapsed = time.time() - start_time
    print(f"\n\n" + "=" * 70)
    print("               DOWNLOAD COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"  • Downloaded:       {success_count} videos")
    print(f"  • Failed:           {fail_count} videos")
    print(f"  • Time Elapsed:     {elapsed:.1f} seconds")
    print(f"  • Total in Folder:  {len(os.listdir(args.out_dir))} videos")
    print(f"  • Total Matched:    {len(already_downloaded) + success_count} Ground Truth Pairs")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
