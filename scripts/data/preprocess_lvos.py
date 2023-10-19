# Find the first annotation of every object in the dataset and put them in a separate folder
# such that it matches the test set format.

import os
import sys

from PIL import Image
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

input_dir = sys.argv[1]
output_dir = sys.argv[2]


def process_vid(video_name):
    os.makedirs(os.path.join(output_dir, video_name), exist_ok=True)
    frames = sorted(os.listdir(os.path.join(input_dir, video_name)))

    existing_labels = set()
    for f in frames:
        mask = Image.open(os.path.join(input_dir, video_name, f))
        palette = mask.getpalette()
        mask = np.array(mask).astype(np.uint8)
        labels = np.unique(mask)
        labels = labels[labels != 0].tolist()
        new_labels = [l for l in labels if l not in existing_labels]

        if len(new_labels) > 0:
            existing_labels.update(new_labels)
            output_mask = np.zeros_like(mask)
            for l in new_labels:
                output_mask[mask == l] = l
            output_mask = Image.fromarray(output_mask)
            output_mask.putpalette(palette)
            output_mask.save(os.path.join(output_dir, video_name, f))


if __name__ == '__main__':
    videos = sorted(os.listdir(input_dir))
    with Pool(8) as p:
        list(tqdm(p.imap(process_vid, videos), total=len(videos)))