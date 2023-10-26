from os import path
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np
import pycocotools.mask as mask_utils
from typing import Dict

from cutie.utils.palette import davis_palette


class BURSTVideoReader(Dataset):
    """
    This class is used to read a video, one frame at a time
    Tailored for the BURST dataset
    """
    def __init__(
        self,
        image_root: str,
        sequence_json: Dict,
        *,
        size: int = -1,
        skip_frames: int = -1,
    ):
        self.sequence_json = sequence_json
        dataset = self.sequence_json['dataset']
        self.vid_name = self.sequence_json['seq_name']
        annotated_frames = self.sequence_json['annotated_image_paths']
        self.annotated_frames = [f[:-4] for f in annotated_frames]  # remove .jpg extensions

        # read all frames in the image directory
        self.image_dir = path.join(image_root, dataset, self.vid_name)
        self.frames = self.sequence_json['all_image_paths']

        # subsample frames
        if skip_frames > 0:
            self.frames = set(self.frames[::skip_frames]).union(set(annotated_frames))
            self.frames = sorted(list(self.frames))

        if size < 0:
            self.im_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            self.mask_transform = transforms.Compose([])
        else:
            self.im_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(size, interpolation=InterpolationMode.BILINEAR, antialias=True),
            ])
            self.mask_transform = transforms.Compose([
                transforms.Resize(size, interpolation=InterpolationMode.NEAREST, antialias=True),
            ])
        self.size = size
        self.use_long_id = False

    def __getitem__(self, idx):
        frame = self.frames[idx]
        info = {}
        data = {}
        info['frame'] = frame
        info['save'] = (frame[:-4] in self.annotated_frames)

        im_path = path.join(self.image_dir, frame)
        img = Image.open(im_path).convert('RGB')
        shape = np.array(img).shape[:2]
        img = self.im_transform(img)

        frame_annotated = frame[:-4] in self.annotated_frames
        if frame_annotated:
            annotation_index = self.annotated_frames.index(frame[:-4])
            segmentations = self.sequence_json['segmentations'][annotation_index]
            if len(segmentations) > 0:
                valid_labels = np.array([int(k) for k in segmentations.keys()])
                mask = np.zeros(shape, dtype=np.uint8)
                for id, segment in segmentations.items():
                    object_mask = mask_utils.decode({'size': shape, 'counts': segment['rle']})
                    mask[object_mask == 1] = int(id)

                    assert int(id) <= 255, 'Too many objects in the frame -- long id needed'

                mask = Image.fromarray(mask)
                mask = self.mask_transform(mask)
                mask = torch.LongTensor(np.array(mask))
                data['mask'] = mask
                data['valid_labels'] = valid_labels

        info['shape'] = shape
        info['resize_needed'] = not (self.size < 0)
        info['time_index'] = self.frames.index(frame)
        info['path_to_image'] = im_path
        data['rgb'] = img
        data['info'] = info

        return data

    def get_palette(self):
        return davis_palette

    def __len__(self):
        return len(self.frames)