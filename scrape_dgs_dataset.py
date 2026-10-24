"""
DGS (German Sign Language) Paired Video & SiGML Scraper & Builder
Scrapes isolated sign videos and matching HamNoSys phonetic transcriptions
from the University of Hamburg Dicta-Sign Corpus (1,046 total concepts),
converts the HamNoSys font glyphs to valid CWASA SiGML XML files,
and downloads the corresponding videos as pairs.
"""

import os
import re
import csv
import argparse
import urllib.request
from typing import Dict, List, Tuple, Optional

# Load HamNoSys Unicode hex -> XML tag mapping
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
SPREADSHEET_PATH = os.path.join(
    WORKSPACE_DIR,
    "Senior Code", "HamNoSys2SiGML-master", "HamNoSys2SiGML-master", "Original", "conversionSpreadSheet.txt"
)

def load_hamnosys_map() -> Dict[str, str]:
    mapping = {}
    if os.path.exists(SPREADSHEET_PATH):
        with open(SPREADSHEET_PATH, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split(", ")
                if len(parts) == 2:
                    tag, hex_code = parts[0].strip(), parts[1].strip().upper()
                    mapping[hex_code] = tag
    return mapping

HEX_TO_TAG = load_hamnosys_map()

# Fallback basic movement & location maps if needed
HANDSHAPES = {
    "hamfist", "hamflathand", "hamfinger2", "hamfinger23", "hamfinger23spread",
    "hamfinger2345", "hampinch12", "hampinchall", "hampinch12open", "hamcee12",
    "hamceeall", "hamceeopen", "hamfinger2straight", "hamfinger2bend"
}

MOVEMENTS = {
    "hammoveu", "hammoved", "hammovel", "hammover", "hammoveo", "hammovei",
    "hammoveui", "hammoveuo", "hammovedi", "hammovedo", "hammoveul", "hammoveur",
    "hammovedl", "hammovedr", "hamcircleol", "hamcircleor", "hamcircleil",
    "hamcircleir", "hamwavy", "hamnodding", "hamnomotion"
}

LOCATIONS = {
    "hamhead", "hamforehead", "hamchin", "hamteeth", "hamlips", "hamtongue",
    "hamneck", "hamshoulders", "hamchest", "hamstomach", "hambelly", "hampalm",
    "hamthumbball", "hamhandback", "hamthumbside", "hampinkyside"
}

PALM_DIRS = {
    "hampalmu", "hampalmd", "hampalml", "hampalmr", "hampalmul", "hampalmur",
    "hampalmdl", "hampalmdr"
}

def clean_filename(name: str) -> str:
    cleaned = re.sub(r'[^a-zA-Z0-9_\-]', '_', name)
    return cleaned.strip('_')

def convert_hns_to_tags(hns_text: str) -> List[str]:
    tags = []
    for char in hns_text:
        hex_code = f"{ord(char):04X}"
        if hex_code in HEX_TO_TAG:
            tags.append(HEX_TO_TAG[hex_code])
        else:
            # Check prefix for E0xx
            tags.append(f"hns_{hex_code}")
    return tags

def generate_sigml(gloss: str, tags: List[str]) -> str:
    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<sigml>',
        f'  <hns_sign gloss="{gloss}">',
        '    <hamnosys_nonmanual>',
        '    </hamnosys_nonmanual>',
        '    <hamnosys_manual>'
    ]
    for tag in tags:
        lines.append(f'      <{tag}/>')
    lines.append('    </hamnosys_manual>')
    lines.append('  </hns_sign>')
    lines.append('</sigml>')
    return '\n'.join(lines)

def scrape_pair(concept_id: int, lang: str = "dgs", output_dir: str = "downloaded_dgs_dataset") -> Optional[dict]:
    url = f"https://www.sign-lang.uni-hamburg.de/dicta-sign/portal/concepts/cs/cs_{concept_id}.html"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    req = urllib.request.Request(url, headers=headers)
    try:
        html_bytes = urllib.request.urlopen(req, timeout=10).read()
        html_text = html_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"[-] Concept {concept_id}: HTTP fetch failed ({e})")
        return None

    # Search section for requested language (dgs, bsl, lsf, gsl)
    lang_upper = lang.upper()
    lang_idx = html_text.find(lang_upper)
    if lang_idx == -1:
        print(f"[-] Concept {concept_id}: Language {lang_upper} section not found")
        return None

    section = html_text[lang_idx:lang_idx + 2500]
    
    # Extract video, gloss, and hns glyphs
    video_match = re.search(r"source src=[\"']\./\.\.(/" + lang + r"/\d+\.mp4)[\"']", section)
    gloss_match = re.search(r"class=[\"']gloss[\"']>([^<]+)</div>", section)
    hns_match = re.search(r"class=[\"']hns[\"']>([^<]+)</div>", section)
    
    if not video_match:
        print(f"[-] Concept {concept_id}: Video link not found for {lang_upper}")
        return None

    rel_video_path = video_match.group(1)
    video_url = f"https://www.sign-lang.uni-hamburg.de/dicta-sign/portal/concepts{rel_video_path}"
    
    raw_gloss = gloss_match.group(1).strip() if gloss_match else f"sign_{concept_id}"
    safe_gloss = clean_filename(f"{concept_id}_{raw_gloss}")
    hns_str = hns_match.group(1).strip() if hns_match else ""
    
    tags = convert_hns_to_tags(hns_str)
    sigml_content = generate_sigml(raw_gloss, tags)
    
    # Identify component categories for training
    handshape = next((t for t in tags if t in HANDSHAPES), "hamflathand")
    palm = next((t for t in tags if t in PALM_DIRS), "hampalmd")
    location = next((t for t in tags if t in LOCATIONS), "hamchest")
    movement = next((t for t in tags if t in MOVEMENTS), "hamnomotion")

    # Destination directories
    video_dir = os.path.join(output_dir, "videos")
    sigml_dir = os.path.join(output_dir, "sigml")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(sigml_dir, exist_ok=True)

    dest_video = os.path.join(video_dir, f"{safe_gloss}.mp4")
    dest_sigml = os.path.join(sigml_dir, f"{safe_gloss}.sigml")

    # Download video
    try:
        v_req = urllib.request.Request(video_url, headers=headers)
        with urllib.request.urlopen(v_req, timeout=15) as resp, open(dest_video, "wb") as f_out:
            f_out.write(resp.read())
    except Exception as e:
        print(f"[-] Concept {concept_id}: Failed to download video from {video_url} ({e})")
        return None

    # Write SiGML XML
    with open(dest_sigml, "w", encoding="utf-8") as f_sigml:
        f_sigml.write(sigml_content)

    print(f"[+] Downloaded: {safe_gloss}.mp4 & {safe_gloss}.sigml (tags: {len(tags)})")

    return {
        "concept_id": concept_id,
        "language": lang_upper,
        "gloss": raw_gloss,
        "video_file": f"{safe_gloss}.mp4",
        "sigml_file": f"{safe_gloss}.sigml",
        "handshape": handshape,
        "palm_orientation": palm,
        "location": location,
        "movement": movement,
        "raw_hns_tags": " ".join(tags)
    }

def main():
    parser = argparse.ArgumentParser(description="Download paired video and ground-truth SiGML dataset")
    parser.add_argument("--count", type=int, default=10, help="Number of pairs to download (max 1046)")
    parser.add_argument("--start", type=int, default=1, help="Starting concept ID (1-1046)")
    parser.add_argument("--lang", type=str, default="dgs", choices=["dgs", "bsl", "lsf", "gsl"], help="Sign language")
    parser.add_argument("--output_dir", type=str, default="downloaded_dgs_dataset", help="Output directory")
    args = parser.parse_args()

    print(f"=== Starting Download of {args.count} Paired Signs ({args.lang.upper()}) ===")
    print(f"Source: University of Hamburg Dicta-Sign Corpus")
    print(f"Output Directory: {args.output_dir}\n")

    records = []
    for cid in range(args.start, args.start + args.count):
        rec = scrape_pair(cid, lang=args.lang, output_dir=args.output_dir)
        if rec:
            records.append(rec)

    if records:
        csv_path = os.path.join(args.output_dir, f"{args.lang}_ground_truth_labels.csv")
        fieldnames = ["concept_id", "language", "gloss", "video_file", "sigml_file", "handshape", "palm_orientation", "location", "movement", "raw_hns_tags"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        print(f"\n[DONE] Successfully downloaded {len(records)} paired samples!")
        print(f"Ground truth table saved to: {csv_path}")

if __name__ == "__main__":
    main()
