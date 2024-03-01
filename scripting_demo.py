import os

import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image
import numpy as np

from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model


@torch.inference_mode()
@torch.cuda.amp.autocast()
def main():

    cutie = get_default_model()
    processor = InferenceCore(cutie, cfg=cutie.cfg)

    image_path = './examples/images/bike'
    images = sorted(os.listdir(image_path))  # ordering is important
    mask = Image.open('./examples/masks/bike/00000.png')
    palette = mask.getpalette()
    objects = np.unique(np.array(mask))
    objects = objects[objects != 0].tolist()  # background "0" does not count as an object
    mask = torch.from_numpy(np.array(mask)).cuda()

    for ti, image_name in enumerate(images):
        image = Image.open(os.path.join(image_path, image_name))
        image = to_tensor(image).cuda().float()

        if ti == 0:
            prediction = processor.step(image, mask, objects=objects)
        else:
            prediction = processor.step(image)

        # visualize prediction
        prediction = torch.argmax(prediction, dim=0)
        prediction = Image.fromarray(prediction.cpu().numpy().astype(np.uint8))
        prediction.putpalette(palette)
        prediction.show()  # or use prediction.save(...) to save it somewhere


main()
