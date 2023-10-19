import json

from cutie.inference.data.burst_video_reader import BURSTVideoReader


class BURSTTestDataset:
    def __init__(self, image_dir: str, json_dir: str, *, size: int = -1, skip_frames: int = -1):
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.size = size
        self.skip_frames = skip_frames
        with open(json_dir) as f:
            self.json = json.load(f)
        self.sequences = self.json['sequences']

    def get_datasets(self):
        for sequence in self.sequences:
            yield BURSTVideoReader(
                self.image_dir,
                sequence,
                size=self.size,
                skip_frames=self.skip_frames,
            )

    def __len__(self):
        return len(self.sequences)
