#!/usr/bin/env python3
"""Quick diagnostic: check WLASL video availability & class balance."""
import json, os

dict_path = r'd:\academics\HamNoSys_Group14_V2\Integration-20260706T062240Z-3-001\Integration\gloss_to_hamnosys_dict.json'
wlasl_path = r'd:\academics\HamNoSys_Group14_V2\Integration-20260706T062240Z-3-001\Integration\WLASL_videos\archive\WLASL_v0.3.json'
video_dir = r'd:\academics\HamNoSys_Group14_V2\Integration-20260706T062240Z-3-001\Integration\WLASL_videos\archive\videos'

with open(dict_path) as f:
    d = json.load(f)
with open(wlasl_path) as f:
    wlasl = json.load(f)

videos = set(os.path.splitext(f)[0] for f in os.listdir(video_dir) if f.endswith('.mp4'))
print(f'Total videos on disk: {len(videos)}')
print(f'Dictionary glosses: {len(d)}')

matched = 0
total_vids_matched = 0
two_hand_vids = 0
one_hand_vids = 0

for entry in wlasl:
    g = entry.get('gloss')
    if g in d:
        vids = [inst['video_id'] for inst in entry.get('instances', []) if inst['video_id'] in videos]
        if vids:
            matched += 1
            total_vids_matched += len(vids)
            two = d[g]['two_handed']
            if two != 'none':
                two_hand_vids += len(vids)
            else:
                one_hand_vids += len(vids)
            print(f'  {g}: {len(vids)} videos, two_handed={two}')

print(f'\nMatched glosses: {matched}, Total matching videos: {total_vids_matched}')
print(f'1-handed videos: {one_hand_vids}, 2-handed videos: {two_hand_vids}')

print(f'\n2-handed glosses in dict:')
for k, v in d.items():
    th = v['two_handed']
    if th != 'none':
        print(f'  {k}: {th}')
