import sys
import os
import json
from collections import defaultdict

from PIL import Image
import numpy as np
import pycocotools.mask as mask_utils
import tqdm

gt_json_path = sys.argv[1]
input_mask_path = sys.argv[2]
output_json_path = sys.argv[3]

with open(gt_json_path, 'r') as f:
    json_dict = json.load(f)

videos = defaultdict(list)
video_list = os.listdir(input_mask_path)
for video_name in video_list:
    dataset, video = video_name.split('_-_')
    videos[dataset].append(video)

sequences = json_dict['sequences']
for seq in tqdm.tqdm(sequences):
    dataset = seq['dataset']
    seq_name = seq['seq_name']

    assert dataset in videos
    assert seq_name in videos[dataset]

    annotated_image_paths = seq['annotated_image_paths']
    segmentations = []
    for image_path in annotated_image_paths:
        this_segment = {}
        mask_path = os.path.join(input_mask_path, dataset + '_-_' + seq_name,
                                 image_path[:-4] + '.png')

        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path))
            labels = np.unique(mask)
            labels = labels[labels != 0].tolist()

            for label in labels:
                this_mask = (mask == label).astype(np.uint8)
                if this_mask.sum() == 0:
                    continue

                this_mask = mask_utils.encode(np.asfortranarray(this_mask))
                this_mask['counts'] = this_mask['counts'].decode('utf-8')
                this_segment[label] = {'rle': this_mask['counts']}

        segmentations.append(this_segment)

    seq['segmentations'] = segmentations

with open(output_json_path, 'w') as f:
    json.dump(json_dict, f)
