import os
import json
import xml.etree.ElementTree as ET

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SIGN_DIR = os.path.join(CURRENT_DIR, "SignFiles")
BENCH_JSON = os.path.join(CURRENT_DIR, "benchmark_5_videos_results.json")

with open(BENCH_JSON, "r", encoding="utf-8") as f:
    bench_data = json.load(f)

mapping = {
    "Hello.mp4": "hello.sigml",
    "Good.mp4": "good.sigml",
    "Father.mp4": "father.sigml",
    "Apple.mp4": "apple.sigml",
    "Come.mp4": "come.sigml"
}

def get_sigml_tokens(sname):
    tree = ET.parse(os.path.join(SIGN_DIR, sname))
    root = tree.getroot()
    manual = root.find(".//hamnosys_manual")
    if manual is not None:
        return [elem.tag for elem in manual]
    return [elem.tag for elem in root.iter() if elem.tag.startswith("ham")]

print("=" * 95)
print(f"{'VIDEO':<12} | {'VERSION':<8} | {'PRED':<5} | {'GT':<5} | {'MATCH':<5} | {'PRECISION':<11} | {'RECALL':<10} | {'F1':<10}")
print("=" * 95)

v1_p_list, v1_r_list, v1_f1_list = [], [], []
v2_p_list, v2_r_list, v2_f1_list = [], [], []

for item in bench_data:
    vname = item["video"]
    sname = mapping[vname]
    gt = get_sigml_tokens(sname)
    gt_set = set(gt)
    
    # V1 evaluation
    v1_toks = [t for t in item["v1"]["tokens"].split() if t.startswith("ham")]
    v1_set = set(v1_toks)
    c1 = gt_set.intersection(v1_set)
    p1 = len(c1) / len(v1_set) if v1_set else 0
    r1 = len(c1) / len(gt_set) if gt_set else 0
    f1_1 = 2 * p1 * r1 / (p1 + r1) if (p1 + r1) else 0
    v1_p_list.append(p1)
    v1_r_list.append(r1)
    v1_f1_list.append(f1_1)
    
    # V2 evaluation
    v2_toks = [t for t in item["v2"]["tokens"].split() if t.startswith("ham")]
    v2_set = set(v2_toks)
    c2 = gt_set.intersection(v2_set)
    p2 = len(c2) / len(v2_set) if v2_set else 0
    r2 = len(c2) / len(gt_set) if gt_set else 0
    f1_2 = 2 * p2 * r2 / (p2 + r2) if (p2 + r2) else 0
    v2_p_list.append(p2)
    v2_r_list.append(r2)
    v2_f1_list.append(f1_2)

    print(f"{vname:<12} | V1       | {len(v1_set):<5} | {len(gt_set):<5} | {len(c1):<5} | {p1*100:6.2f}%     | {r1*100:6.2f}%   | {f1_1*100:6.2f}%")
    print(f"{'':<12} | V2       | {len(v2_set):<5} | {len(gt_set):<5} | {len(c2):<5} | {p2*100:6.2f}%     | {r2*100:6.2f}%   | {f1_2*100:6.2f}%")
    print(f"  [Ground Truth Tokens ({len(gt_set)})]: {gt}")
    print(f"  [V1 Matching Tokens]: {sorted(list(c1))}")
    print(f"  [V2 Matching Tokens]: {sorted(list(c2))}")
    print("-" * 95)

avg_p1 = sum(v1_p_list) / len(v1_p_list) * 100
avg_r1 = sum(v1_r_list) / len(v1_r_list) * 100
avg_f1_1 = sum(v1_f1_list) / len(v1_f1_list) * 100

avg_p2 = sum(v2_p_list) / len(v2_p_list) * 100
avg_r2 = sum(v2_r_list) / len(v2_r_list) * 100
avg_f1_2 = sum(v2_f1_list) / len(v2_f1_list) * 100

print(f"{'AVERAGE':<12} | V1       | {'-':<5} | {'-':<5} | {'-':<5} | {avg_p1:6.2f}%     | {avg_r1:6.2f}%   | {avg_f1_1:6.2f}%")
print(f"{'AVERAGE':<12} | V2       | {'-':<5} | {'-':<5} | {'-':<5} | {avg_p2:6.2f}%     | {avg_r2:6.2f}%   | {avg_f1_2:6.2f}%")
print("=" * 95)
