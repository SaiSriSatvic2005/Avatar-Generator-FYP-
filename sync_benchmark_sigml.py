#!/usr/bin/env python3
import json
import os

with open("v6_benchmark_results.json", "r", encoding="utf-8") as f:
    bench = json.load(f)

for item in bench.get("detailed_results", []):
    stem = os.path.splitext(item["video_file"])[0]
    out_sigml = f"{stem}_hamer_output.sigml"
    tokens = item.get("hamnosys", "").split()
    gloss = item.get("gloss", stem)
    xml_lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<sigml>',
        f'  <hns_sign gloss="{gloss}">',
        '    <hamnosys_nonmanual/>',
        '    <hamnosys_manual>'
    ]
    for t in tokens:
        xml_lines.append(f'      <{t}/>')
    xml_lines.extend(['    </hamnosys_manual>', '  </hns_sign>', '</sigml>'])
    with open(out_sigml, "w", encoding="utf-8") as pf:
        pf.write("\n".join(xml_lines))
    print(f"[OK] Generated: {out_sigml}")
