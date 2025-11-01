import json
import numpy as np
import os

gt_pth = "./artist20/train.json"
# pred_pth = "./predictions.json"

with open(gt_pth, 'r') as f:
    gt_data = json.load(f)

gts = {}

for audio_path in gt_data:
    gt = audio_path.split('/')[-3]
    title = audio_path.split('/')[-1].split('.')[0]
    gts[title] = gt

out_pth = "./train_gt.json"

with open(out_pth, 'w') as f:
    json.dump(gts, f, indent=4)
    