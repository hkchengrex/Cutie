import sys
import os
from os import path

from PIL import Image
import numpy as np

from multiprocessing import Pool
from tqdm import tqdm
import json

data_path = sys.argv[1]
output_path = sys.argv[2]


def process_vid(v):
    output = []
    frames = sorted(os.listdir(path.join(data_path, v)))
    for f in frames:
        im = Image.open(path.join(data_path, v, f))
        im = np.array(im)
        if im.max() == 0:
            output.append(f[:-4])
    return output


videos = sorted(os.listdir(data_path))
pbar = tqdm(videos)
output = {}
for v in pbar:
    output[v] = process_vid(v)

with open(output_path, 'w') as f:
    json.dump(output, f)
