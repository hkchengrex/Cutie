import os
from os import path
import json
from typing import Iterable, Optional

from cutie.inference.data.video_reader import VideoReader


class VOSTestDataset:
    def __init__(self,
                 image_dir: str,
                 mask_dir: str,
                 *,
                 use_all_masks: bool,
                 req_frames_json: Optional[str] = None,
                 size: int = -1,
                 size_dir: Optional[str] = None,
                 subset: Optional[str] = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.use_all_masks = use_all_masks
        self.size = size
        self.size_dir = size_dir

        if subset is None:
            self.vid_list = sorted(os.listdir(self.mask_dir))
        else:
            # DAVIS-2017 style txt
            with open(subset) as f:
                self.vid_list = sorted([line.strip() for line in f])

        self.req_frame_list = {}
        if req_frames_json is not None:
            # YouTubeVOS style json
            with open(req_frames_json) as f:
                # read meta.json to know which frame is required for evaluation
                meta = json.load(f)['videos']

                for vid in self.vid_list:
                    req_frames = []
                    objects = meta[vid]['objects']
                    for value in objects.values():
                        req_frames.extend(value['frames'])

                    req_frames = list(set(req_frames))
                    self.req_frame_list[vid] = req_frames

    def get_datasets(self) -> Iterable[VideoReader]:
        for video in self.vid_list:
            yield VideoReader(
                video,
                path.join(self.image_dir, video),
                path.join(self.mask_dir, video),
                size=self.size,
                to_save=self.req_frame_list.get(video, None),
                use_all_masks=self.use_all_masks,
                size_dir=path.join(self.size_dir, video) if self.size_dir is not None else None,
            )

    def __len__(self):
        return len(self.vid_list)
