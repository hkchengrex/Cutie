import os
from os import path
import shutil
import collections
import logging
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from omegaconf import DictConfig, open_dict
from typing import Dict, Optional, Tuple, Literal, Union
import cv2
from PIL import Image
if not hasattr(Image, 'Resampling'):  # Pillow<9.0
    Image.Resampling = Image
import numpy as np

from cutie.utils.palette import davis_palette
from tqdm import tqdm

log = logging.getLogger()


# https://bugs.python.org/issue28178
# ah python ah why
class LRU:

    def __init__(self, func, maxsize=128):
        self.cache = collections.OrderedDict()
        self.func = func
        self.maxsize = maxsize

    def __call__(self, *args):
        cache = self.cache
        if args in cache:
            cache.move_to_end(args)
            return cache[args]
        result = self.func(*args)
        cache[args] = result
        if len(cache) > self.maxsize:
            cache.popitem(last=False)
        return result

    def invalidate(self, key):
        self.cache.pop(key, None)


@dataclass
class SaveItem:
    type: Literal['mask', 'visualization', 'soft_mask']
    data: Union[Image.Image, np.ndarray]
    name: str = None  # only used for soft_mask


class ResourceManager:

    def __init__(self, cfg: DictConfig):
        # determine inputs
        images = cfg['images']
        video = cfg['video']
        self.workspace = cfg['workspace']
        self.max_size = cfg['max_overall_size']
        self.palette = davis_palette

        # create temporary workspace if not specified
        if self.workspace is None:
            if images is not None:
                basename = path.basename(images)
            elif video is not None:
                basename = path.basename(video)[:-4]
            else:
                raise NotImplementedError('Either images, video, or workspace has to be specified')

            self.workspace = path.join('./workspace', basename)

        print(f'Workspace is in: {self.workspace}')
        with open_dict(cfg):
            cfg['workspace'] = self.workspace

        # determine the location of input images
        need_decoding = False
        need_resizing = False
        if path.exists(path.join(self.workspace, 'images')):
            pass
        elif images is not None:
            need_resizing = True
        elif video is not None:
            # will decode video into frames later
            need_decoding = True

        # create workspace subdirectories
        self.image_dir = path.join(self.workspace, 'images')
        self.mask_dir = path.join(self.workspace, 'masks')
        self.visualization_dir = path.join(self.workspace, 'visualization')
        self.soft_mask_dir = path.join(self.workspace, 'soft_masks')
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.mask_dir, exist_ok=True)
        os.makedirs(self.visualization_dir, exist_ok=True)
        os.makedirs(self.soft_mask_dir, exist_ok=True)

        # create all soft mask sub-directories
        for i in range(1, cfg['num_objects'] + 1):
            os.makedirs(path.join(self.soft_mask_dir, f'{i}'), exist_ok=True)

        # convert read functions to be buffered
        self.get_image = LRU(self._get_image_unbuffered, maxsize=cfg['buffer_size'])
        self.get_mask = LRU(self._get_mask_unbuffered, maxsize=cfg['buffer_size'])

        # extract frames from video
        if need_decoding:
            self._extract_frames(video)

        # copy/resize existing images to the workspace
        if need_resizing:
            self._copy_resize_frames(images)

        # read all frame names
        self.names = sorted(os.listdir(self.image_dir))
        self.names = [f[:-4] for f in self.names]  # remove extensions
        self.length = len(self.names)

        assert self.length > 0, f'No images found! Check {self.workspace}/images. Remove folder if necessary.'

        print(f'{self.length} images found.')

        self.height, self.width = self.get_image(0).shape[:2]

        # create the saver threads for saving the masks/visualizations
        self.save_queue = Queue(maxsize=cfg['save_queue_size'])
        self.num_save_threads = cfg['num_save_threads']
        self.save_threads = [
            Thread(target=self.save_thread, args=(self.save_queue, ))
            for _ in range(self.num_save_threads)
        ]
        for t in self.save_threads:
            t.daemon = True
            t.start()

    def __del__(self):
        for _ in range(self.num_save_threads):
            self.save_queue.put(None)
        self.save_queue.join()
        for t in self.save_threads:
            t.join()

    def save_thread(self, queue: Queue):
        while True:
            args: SaveItem = queue.get()
            if args is None:
                queue.task_done()
                break
            if args.type == 'mask':
                # PIL image
                args.data.save(path.join(self.mask_dir, args.name + '.png'))
            elif args.type.startswith('visualization'):
                # numpy array, save with cv2
                vis_mode = args.type.split('_')[-1]
                os.makedirs(path.join(self.visualization_dir, vis_mode), exist_ok=True)
                if vis_mode == 'rgba':
                    data = cv2.cvtColor(args.data, cv2.COLOR_RGBA2BGRA).copy()
                    cv2.imwrite(path.join(self.visualization_dir, vis_mode, args.name + '.png'),
                                data)
                else:
                    data = cv2.cvtColor(args.data, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(path.join(self.visualization_dir, vis_mode, args.name + '.jpg'),
                                data)
            elif args.type == 'soft_mask':
                # numpy array, save each channel with cv2
                num_channels = args.data.shape[0]
                # first channel is background -- ignore
                for i in range(1, num_channels):
                    data = args.data[i]
                    data = (data * 255).astype(np.uint8)
                    cv2.imwrite(path.join(self.soft_mask_dir, f'{i}', args.name + '.png'), data)
            else:
                raise NotImplementedError
            queue.task_done()

    def _extract_frames(self, video: str):
        cap = cv2.VideoCapture(video)
        frame_index = 0
        print(f'Extracting frames from {video} into {self.image_dir}...')
        with tqdm() as bar:
            while (cap.isOpened()):
                _, frame = cap.read()
                if frame is None:
                    break
                h, w = frame.shape[:2]
                if self.max_size > 0 and min(h, w) > self.max_size:
                    new_w = (w * self.max_size // min(w, h))
                    new_h = (h * self.max_size // min(w, h))
                    frame = cv2.resize(frame, dsize=(new_w, new_h), interpolation=cv2.INTER_AREA)
                cv2.imwrite(path.join(self.image_dir, f'{frame_index:07d}.jpg'), frame)
                frame_index += 1
                bar.update()
        print('Done!')

    def _copy_resize_frames(self, images: str):
        image_list = os.listdir(images)
        print(f'Copying/resizing frames into {self.image_dir}...')
        for image_name in tqdm(image_list):
            if self.max_size < 0:
                # just copy
                shutil.copy2(path.join(images, image_name), self.image_dir)
            else:
                frame = cv2.imread(path.join(images, image_name))
                h, w = frame.shape[:2]
                if self.max_size > 0 and min(h, w) > self.max_size:
                    new_w = (w * self.max_size // min(w, h))
                    new_h = (h * self.max_size // min(w, h))
                    frame = cv2.resize(frame, dsize=(new_w, new_h), interpolation=cv2.INTER_AREA)
                cv2.imwrite(path.join(self.image_dir, image_name), frame)
        print('Done!')

    def add_to_queue_with_warning(self, item: SaveItem):
        if self.save_queue.full():
            print(
                'The save queue is full! You need more threads or faster IO. Program might pause.')
        self.save_queue.put(item)

    def save_mask(self, ti: int, mask: np.ndarray):
        # mask should be uint8 H*W without channels
        assert 0 <= ti < self.length
        assert isinstance(mask, np.ndarray)

        mask = Image.fromarray(mask)
        mask.putpalette(self.palette)
        self.invalidate(ti)
        self.add_to_queue_with_warning(SaveItem('mask', mask, self.names[ti]))

    def save_visualization(self, ti: int, vis_mode: str, image: np.ndarray):
        # image should be uint8 3*H*W
        assert 0 <= ti < self.length
        assert isinstance(image, np.ndarray)

        self.add_to_queue_with_warning(SaveItem(f'visualization_{vis_mode}', image, self.names[ti]))

    def save_soft_mask(self, ti: int, prob: np.ndarray):
        # mask should be float (num_objects+1)*H*W np array
        assert 0 <= ti < self.length
        assert isinstance(prob, np.ndarray)

        self.add_to_queue_with_warning(SaveItem('soft_mask', prob, self.names[ti]))

    def _get_image_unbuffered(self, ti: int):
        # returns H*W*3 uint8 array
        assert 0 <= ti < self.length

        image = Image.open(path.join(self.image_dir, self.names[ti] + '.jpg')).convert('RGB')
        image = np.array(image)
        return image

    def _get_mask_unbuffered(self, ti: int):
        # returns H*W uint8 array
        assert 0 <= ti < self.length

        mask_path = path.join(self.mask_dir, self.names[ti] + '.png')
        if path.exists(mask_path):
            mask = Image.open(mask_path)
            mask = np.array(mask)
            return mask
        else:
            return None

    def import_mask(self, file_name: str, size: Optional[Tuple[int, int]] = None):
        # read an mask file and resize it to exactly match the canvas size
        image = Image.open(file_name)
        if size is not None:
            # PIL uses (width, height)
            image = image.resize((size[1], size[0]), resample=Image.Resampling.NEAREST)
        image = np.array(image)
        return image

    def import_layer(self, file_name: str, size: Tuple[int, int]):
        # read a RGBA/RGB file and resize it such that the entire layer is visible in the canvas
        # and then pad it to the canvas size (h, w)
        image = Image.open(file_name).convert('RGBA')
        im_w, im_h = image.size
        im_ratio = im_w / im_h
        canvas_ratio = size[1] / size[0]
        if im_ratio < canvas_ratio:
            # fit height
            new_h = size[0]
            new_w = int(new_h * im_ratio)
        else:
            # fit width
            new_w = size[1]
            new_h = int(new_w / im_ratio)
        image = image.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
        image = np.array(image)
        # padding
        pad_h = (size[0] - new_h) // 2
        pad_w = (size[1] - new_w) // 2
        image = np.pad(image,
                       ((pad_h, size[0] - new_h - pad_h), (pad_w, size[1] - new_w - pad_w), (0, 0)),
                       mode='constant',
                       constant_values=0)

        return image

    def invalidate(self, ti: int):
        # the image buffer is never invalidated
        self.get_mask.invalidate((ti, ))

    def __len__(self):
        return self.length

    @property
    def T(self) -> int:
        return self.length

    @property
    def h(self) -> int:
        return self.height

    @property
    def w(self) -> int:
        return self.width
