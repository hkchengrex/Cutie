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
    # obtain the Cutie model with default parameters -- skipping hydra configuration
    cutie = get_default_model()
    # Typically, use one InferenceCore per video
    processor = InferenceCore(cutie, cfg=cutie.cfg)
    # the processor matches the shorter edge of the input to this size
    # you might want to experiment with different sizes; -1 keeps the original size
    processor.max_internal_size = 480

    image_path = './examples/images/judo'
    mask_path = './examples/masks/judo'
    # ordering is important
    images = sorted(os.listdir(image_path))

    for ti, image_name in enumerate(images):
        # load the image as RGB; normalization is done within the model
        image = Image.open(os.path.join(image_path, image_name))
        image = to_tensor(image).cuda().float()

        # deleting the red mask at time step 10 for no reason -- you can set your own condition
        if ti == 10:
            processor.delete_objects([1])

        mask_name = image_name[:-4] + '.png'

        # we pass the mask in if it exists
        if os.path.exists(os.path.join(mask_path, mask_name)):
            # NOTE: this should be a grayscale mask or a indexed (with/without palette) mask,
            # and definitely NOT a colored RGB image
            # https://pillow.readthedocs.io/en/stable/handbook/concepts.html: mode "L" or "P"
            mask = Image.open(os.path.join(mask_path, mask_name))

            # palette is for visualization
            palette = mask.getpalette()

            # the number of objects is determined by counting the unique values in the mask
            # common mistake: if the mask is resized w/ interpolation, there might be new unique values
            objects = np.unique(np.array(mask))
            # background "0" does not count as an object
            objects = objects[objects != 0].tolist()
            mask = torch.from_numpy(np.array(mask)).cuda()

            # if mask is passed in, it is memorized
            # if not all objects are specified, we propagate the unspecified objects using memory
            output_prob = processor.step(image, mask, objects=objects)
        else:
            # otherwise, we propagate the mask from memory
            output_prob = processor.step(image)

        # convert output probabilities to an object mask
        mask = processor.output_prob_to_mask(output_prob)

        # visualize prediction
        mask = Image.fromarray(mask.cpu().numpy().astype(np.uint8))
        mask.putpalette(palette)
        # mask.show()  # or use prediction.save(...) to save it somewhere
        mask.save(os.path.join('./examples', mask_name))


main()
