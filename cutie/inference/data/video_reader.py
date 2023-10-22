from typing import List, Optional
import os
from os import path
import copy

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image, ImagePalette
import numpy as np


class VideoReader(Dataset):
    """
    This class is used to read a video, one frame at a time
    """
    def __init__(
        self,
        vid_name: str,
        image_dir: str,
        mask_dir: str,
        *,
        size: int = -1,
        to_save: Optional[List[str]] = None,
        use_all_masks: bool = False,
        size_dir: Optional[str] = None,
        start: int = -1,
        end: int = -1,
        reverse: bool = False,
        object_name: str = None,
        enabled_frame_list: Optional[List[str]] = None,
    ):
        """
        image_dir - points to a directory of jpg images
        mask_dir - points to a directory of png masks
        size_dir - points to a directory of jpg images that determine the size of the output
        size - resize min. side to size. Does nothing if <0.
        to_save - optionally contains a list of file names without extensions 
            where the segmentation mask is required
        use_all_mask - when true, read all available mask in mask_dir.
            Default false. Set to true for YouTubeVOS validation.
        object_name - if none, read from all objects. if not none, read that object only. 
                        only valid in soft mask mode
        """
        self.vid_name = vid_name
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.to_save = to_save
        self.use_all_mask = use_all_masks
        self.object_name = object_name
        self.enabled_frame_list = enabled_frame_list
        if size_dir is None:
            self.size_dir = self.image_dir
        else:
            self.size_dir = size_dir

        # read all frames in the image directory
        self.frames = sorted(os.listdir(self.image_dir))

        if enabled_frame_list is not None:
            self.frames = [f for f in self.frames if f[:-4] in enabled_frame_list]

        # enforce start and end frame if needed
        self._all_frames = copy.deepcopy(self.frames)
        if start >= 0:
            if end >= 0:
                self.frames = self.frames[start:end]
            else:
                self.frames = self.frames[start:]
        elif end >= 0:
            self.frames = self.frames[:end]

        if reverse:
            self.frames = list(reversed(self.frames))

        # determine if the mask uses 3-channel long ID or 1-channel (0~255) short ID
        self.first_mask_frame = sorted(os.listdir(self.mask_dir))[0]
        first_mask = Image.open(path.join(self.mask_dir, self.first_mask_frame))
        if first_mask.mode == 'P':
            self.use_long_id = False
            self.palette = first_mask.getpalette()
        elif first_mask.mode == 'RGB':
            self.use_long_id = True
            self.palette = None
        elif first_mask.mode == 'L':
            self.use_long_id = False
            self.palette = None
        else:
            raise NotImplementedError(f'Unknown mode {first_mask.mode} in {self.first_mask_frame}.')

        self.im_transform = transforms.ToTensor()
        self.im_resize = transforms.Resize(size,
                                           interpolation=InterpolationMode.BILINEAR,
                                           antialias=True)
        self.mask_resize = transforms.Resize(size,
                                             interpolation=InterpolationMode.NEAREST,
                                             antialias=True)
        self.size = size

    def __getitem__(self, idx):
        frame = self.frames[idx]
        info = {}
        data = {}
        info['frame'] = frame
        info['save'] = (self.to_save is None) or (frame[:-4] in self.to_save)

        im_path = path.join(self.image_dir, frame)
        img = Image.open(im_path).convert('RGB')

        if self.image_dir == self.size_dir:
            output_shape = np.array(img).shape[:2]
        else:
            size_path = path.join(self.size_dir, frame)
            size_im = Image.open(size_path).convert('RGB')
            output_shape = np.array(size_im).shape[:2]

        # resize if the input image is too large
        if self.image_dir != self.size_dir:
            # might be different from shape if size_dir is used
            input_shape = np.array(img).shape[:2]
        else:
            input_shape = output_shape
        resize_needed = (input_shape != output_shape) or ((self.size > 0) and
                                                          (min(input_shape) > self.size))
        img = self.im_transform(img)
        if resize_needed:
            img = self.im_resize(img)

        load_mask = self.use_all_mask or (frame[:-4] == self.first_mask_frame[:-4])

        if load_mask:
            mask_path = path.join(self.mask_dir, frame[:-4] + '.png')
            if path.exists(mask_path):
                mask = Image.open(mask_path)
                if resize_needed:
                    mask = self.mask_resize(mask)
                mask = torch.LongTensor(np.array(mask))

                if self.use_long_id:
                    assert len(mask.shape) == 3, 'RGB masks should have 3 dimensions'
                    mask = mask[:, :, 0] + mask[:, :, 1] * 256 + mask[:, :, 2] * 256 * 256
                else:
                    assert len(mask.shape) == 2, 'Single channel masks should have 2 dimensions'
                    pass

                valid_labels = torch.unique(mask)
                valid_labels = valid_labels[valid_labels != 0]
                data['mask'] = mask
                data['valid_labels'] = valid_labels

        info['shape'] = output_shape
        info['resize_needed'] = resize_needed
        info['time_index'] = self._all_frames.index(frame)
        info['path_to_image'] = im_path
        data['rgb'] = img
        data['info'] = info

        return data

    def get_palette(self) -> ImagePalette:
        return self.palette

    def __len__(self):
        return len(self.frames)
