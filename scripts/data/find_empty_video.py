import sys
import os
from os import path

from PIL import Image
import numpy as np


data_path = sys.argv[1]

videos = os.listdir(data_path)
for v in videos:
    f = sorted(os.listdir(path.join(data_path, v)))[0]
    im = Image.open(path.join(data_path, v, f))
    im = np.array(im)
    if im.max() == 0:
        print(v)
