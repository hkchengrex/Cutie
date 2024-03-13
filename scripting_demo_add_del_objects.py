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

    image_path = './examples/images/judo'
    mask_path = './examples/masks/judo'
    images = sorted(os.listdir(image_path))  # ordering is important

    for ti, image_name in enumerate(images):
        image = Image.open(os.path.join(image_path, image_name))
        image = to_tensor(image).cuda().float()

        # deleting the red mask at time step 10 for no reason -- you can set your own condition
        if ti == 10:
            processor.delete_objects([1])

        mask_name = image_name[:-4] + '.png'
        if os.path.exists(os.path.join(mask_path, mask_name)):
            # add the objects in the mask
            mask = Image.open(os.path.join(mask_path, mask_name))
            palette = mask.getpalette()
            objects = np.unique(np.array(mask))
            objects = objects[objects != 0].tolist()  # background "0" does not count as an object
            mask = torch.from_numpy(np.array(mask)).cuda()

            prediction = processor.step(image, mask, objects=objects)
        else:
            prediction = processor.step(image)

        # visualize prediction
        mask = torch.argmax(prediction, dim=0)

        # since the objects might shift in the channel dim due to deletion, remap the ids
        new_mask = torch.zeros_like(mask)
        for tmp_id, obj in processor.object_manager.tmp_id_to_obj.items():
            new_mask[mask == tmp_id] = obj.id
        mask = new_mask

        mask = Image.fromarray(mask.cpu().numpy().astype(np.uint8))
        mask.putpalette(palette)
        # mask.show()  # or use prediction.save(...) to save it somewhere
        mask.save(os.path.join('./examples', mask_name))


main()
