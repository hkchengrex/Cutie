import os
from os import path
import logging
from typing import Dict, List, Tuple

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np
import cv2

from cutie.dataset.utils import im_mean, reseed

log = logging.getLogger()
local_rank = int(os.environ['LOCAL_RANK'])


class VOSMergeTrainDataset(Dataset):
    """
    Note: data normalization happens within the model instead of here
    
    For VOS data training
    data_configs is a Dict indexed by the name of the dataset, each containing:
    - im_root: path to the image directory
    - gt_root: path to the ground-truth directory
    - max_skip: maximum number of allowed separations between consecutive frames
    - subset: a list of video names to use. If None, all videos are used.
    - empty_masks: a Dict[video_name, list of frames as string without extensions] 
                    that contain no objects. 
                    Can be None. (used to speed up data selection -- not mandatory)
    - multiplier: number of times to oversample this dataset

    For each sequence:
    - Pick num_frames frames
    - Pick max_num_obj objects
    - Apply some random transforms that are the same for all frames
    - Apply random transform to each of the frames
    - The distance between frames is limited by max_skip

    With merge_probability, we sample another sequence and merge them as a single training sample 
    """
    def __init__(self, data_configs, seq_length=3, max_num_obj=3, size=480, merge_probability=0.0):

        self.configs = data_configs
        self.seq_length = seq_length
        self.max_num_obj = max_num_obj
        self.size = size
        self.merge_probability = merge_probability

        self.max_crop_trials = 5  # number of attempts at cropping a single frame
        self.max_seed_trials = 5  # number of attempts at changing the initial seed frame
        self.max_seq_trials = 100  # number of attempts at generating a sequence from the seed frame

        self.videos: Dict[List[str]] = {}
        self.frames: Dict[Dict[str, List[str]]] = {}
        self.video_frames: List[Tuple(str, str, int)] = []

        for dataset, config in data_configs.items():
            self.frames[dataset] = {}
            self.videos[dataset] = []
            total_frames = 0

            im_root = config['im_root']
            subset = config['subset']
            multiplier = config['multiplier']

            # Find all videos
            vid_list = sorted(os.listdir(im_root))
            for vid in vid_list:
                if subset is not None:
                    if vid not in subset:
                        continue
                frames = sorted(os.listdir(os.path.join(im_root, vid)))
                if len(frames) < seq_length:
                    continue
                self.frames[dataset][vid] = frames
                self.videos[dataset].append(vid)
                self.video_frames.extend([(dataset, vid, i)
                                          for i, _ in enumerate(frames)] * multiplier)
                total_frames += len(frames)

            if local_rank == 0:
                log.info(
                    f'{dataset}: {len(self.videos[dataset])}/{len(vid_list)} videos will be used in {im_root}.'
                )
                log.info(
                    f'{dataset}: {total_frames} frames found. Multiplied to {total_frames*multiplier} frames.'
                )

        if local_rank == 0:
            log.info(f'Total number of video-frames: {len(self.video_frames)}.')

        # The frame transforms are the same for each of the pairs,
        # but different for different pairs in the sequence
        self.frame_image_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.05, 0.05, 0),
        ])

        # The sequence transforms are the same for all pairs in the sampled sequence
        self.sequence_image_only_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.03, 0.03, 0),
            transforms.RandomGrayscale(0.05),
        ])

        self.sequence_image_dual_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=25,
                                    shear=20,
                                    interpolation=InterpolationMode.BILINEAR,
                                    fill=im_mean),
            transforms.RandomResizedCrop((self.size, self.size),
                                         scale=(0.36, 1.0),
                                         interpolation=InterpolationMode.BILINEAR)
        ])

        self.sequence_mask_dual_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=25,
                                    shear=20,
                                    interpolation=InterpolationMode.NEAREST,
                                    fill=0),
            transforms.RandomResizedCrop((self.size, self.size),
                                         scale=(0.36, 1.0),
                                         interpolation=InterpolationMode.NEAREST)
        ])

        self.output_image_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def _get_sample(self, idx=None):
        # pick, augment, and return a video sequence
        # We look at the sequence given by idx first, but there is no guarantee that we will use it
        if idx is None:
            idx = np.random.randint(len(self.video_frames))

        dataset, video, frame_idx = self.video_frames[idx]
        num_frames = self.seq_length
        while True:
            config = self.configs[dataset]
            empty_masks = config['empty_masks'][video] if config['empty_masks'] else None
            im_path = path.join(config['im_root'], video)
            gt_path = path.join(config['gt_root'], video)
            max_skip = config['max_skip']

            info = {'name': video}
            frames = self.frames[dataset][video]
            length = len(frames)
            this_max_skip = min(len(frames), max_skip)

            # This is reset if the sampled frames are not admissible
            frames_idx = [frame_idx]

            for seed_trial in range(self.max_seed_trials):
                seed_ok = True
                info['frames'] = []  # To be filled with sampled frames
                """
                From the seed frame, we expand it to a sequence without exceeding max_skip
                The first frame in the sequence should not be empty
                empty_masks contains a list of empty masks (as str, without extension) 
                (from external pre-processing)
                """
                for seq_trial in range(self.max_seq_trials):
                    sampled_frames = frames_idx.copy()
                    # acceptable_set contains the indices that are within
                    # max_skip from any sampled frames
                    acceptable_set = set(
                        range(max(0, sampled_frames[-1] - this_max_skip),
                              min(length, sampled_frames[-1] + this_max_skip + 1))).difference(
                                  set(sampled_frames))
                    while (len(sampled_frames) < num_frames):
                        idx = np.random.choice(list(acceptable_set))
                        sampled_frames.append(idx)
                        new_set = set(
                            range(max(0, sampled_frames[-1] - this_max_skip),
                                  min(length, sampled_frames[-1] + this_max_skip + 1)))
                        acceptable_set = acceptable_set.union(new_set).difference(
                            set(sampled_frames))

                    sampled_frames = sorted(sampled_frames)
                    if np.random.rand() < 0.5:
                        # Reverse time
                        sampled_frames = sampled_frames[::-1]

                    # admit the sequence if the first frame is not empty
                    if empty_masks is None or frames[sampled_frames[0]][:-4] not in empty_masks:
                        frames_idx = sampled_frames
                        break

                    # if we tried enough, just pass and consider this a failure
                    if seq_trial >= self.max_seq_trials - 1:
                        seed_ok = False
                        break

                # give up early if we failed to find a sequence
                if not seed_ok:
                    if seed_trial == self.max_seed_trials - 1:
                        # search for a new video-frame
                        break
                    else:
                        # reset seed frame and try again
                        frames_idx = [np.random.randint(length)]
                        continue
                """
                Read the frames in frames_idx one-by-one and augments them
                We want to find a good crop such that the first frame is not empty
                """
                images = []
                masks = []
                for i, f_idx in enumerate(frames_idx):
                    jpg_name = frames[f_idx][:-4] + '.jpg'
                    png_name = frames[f_idx][:-4] + '.png'
                    info['frames'].append(jpg_name)

                    if i == 0:
                        for crop_trial in range(self.max_crop_trials):
                            sequence_seed = np.random.randint(2147483647)

                            reseed(sequence_seed)
                            this_gt = Image.open(path.join(gt_path, png_name)).convert('P')
                            this_gt = self.sequence_mask_dual_transform(this_gt)
                            this_gt = np.array(this_gt)

                            # we want a non-empty crop for the first frame
                            if this_gt.max() == 0:
                                if crop_trial >= self.max_crop_trials - 1:
                                    # tried enough -- giving up
                                    seed_ok = False
                                    break
                            else:
                                # good enough
                                break
                    else:
                        # we don't check the other frames -- just read them
                        reseed(sequence_seed)
                        this_gt = Image.open(path.join(gt_path, png_name)).convert('P')
                        this_gt = self.sequence_mask_dual_transform(this_gt)
                        this_gt = np.array(this_gt)

                    if not seed_ok:
                        # fall-through from above
                        break

                    # No check requires for images
                    reseed(sequence_seed)
                    this_im = Image.open(path.join(im_path, jpg_name)).convert('RGB')
                    this_im = self.sequence_image_dual_transform(this_im)
                    this_im = self.sequence_image_only_transform(this_im)

                    this_im = self.frame_image_transform(this_im)
                    this_im = self.output_image_transform(this_im)

                    images.append(this_im)
                    masks.append(this_gt)

                # fall-through from above
                if not seed_ok:
                    if seed_trial == self.max_seed_trials - 1:
                        # search for a new video-frame
                        break
                    else:
                        # reset seed frame and try again
                        frames_idx = [np.random.randint(length)]
                        continue
                """
                Everything should be good if the code reaches here -- proceed to output
                """
                images = torch.stack(images, 0)
                masks = np.stack(masks, 0)
                return info, images, masks

            # get a new video-frame
            idx = np.random.randint(len(self.video_frames))
            dataset, video, frame_idx = self.video_frames[idx]

    def __getitem__(self, idx):

        info, images, masks = self._get_sample(idx)
        labels = np.unique(masks[0])
        labels = labels[labels != 0].tolist()
        num_labels = len(labels)

        # Potentially sample from another sequence and merge them as a training sample
        if num_labels < self.max_num_obj and np.random.rand() < self.merge_probability:
            _, images2, masks2 = self._get_sample()
            labels2 = np.unique(masks2[0])
            labels2 = labels2[labels2 != 0].tolist()
            for l2 in labels2:
                obj_masks2 = (masks2 == l2)
                blur_masks = obj_masks2.astype(np.float32).transpose(1, 2, 0)
                blur_masks = cv2.GaussianBlur(blur_masks, [5, 5], 1.0).transpose(2, 0, 1)[:, None]
                images = images * (1 - blur_masks) + images2 * blur_masks

                new_label = (l2 + 10) % 255
                while new_label in labels:
                    new_label = (new_label + 1) % 255
                masks[obj_masks2] = new_label
                labels.append(new_label)

        # recomputing labels as some might have been occluded
        labels = np.unique(masks[0])
        labels = labels[labels != 0].tolist()

        assert len(labels) > 0  # should not be empty at all times
        target_objects = labels

        # if there are more than max_num_obj objects, subsample them
        if len(target_objects) > self.max_num_obj:
            target_objects = np.random.choice(target_objects, size=self.max_num_obj, replace=False)

        info['num_objects'] = max(1, len(target_objects))

        # Generate one-hot ground-truth
        cls_gt = np.zeros((self.seq_length, self.size, self.size), dtype=np.int64)
        first_frame_gt = np.zeros((1, self.max_num_obj, self.size, self.size), dtype=np.int64)
        for i, l in enumerate(target_objects):
            this_mask = (masks == l)
            cls_gt[this_mask] = i + 1
            first_frame_gt[0, i] = (this_mask[0])
        cls_gt = np.expand_dims(cls_gt, 1)

        # 1 if object exist, 0 otherwise
        selector = [1 if i < info['num_objects'] else 0 for i in range(self.max_num_obj)]
        selector = torch.FloatTensor(selector)

        data = {
            'rgb': images,
            'first_frame_gt': first_frame_gt,
            'cls_gt': cls_gt,
            'selector': selector,
            'info': info,
        }

        return data

    def __len__(self):
        return len(self.video_frames)
