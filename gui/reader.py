import sys
from typing import Literal

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

from gui.resource_manager import ResourceManager


class PropagationReader(Dataset):
    def __init__(self, res_man: ResourceManager, start_ti: int, direction: Literal['forward',
                                                                                   'backward']):
        self.res_man = res_man
        self.start_ti = start_ti
        self.direction = direction

        # skip the first frame
        if self.direction == 'forward':
            self.start_ti += 1
            self.length = self.res_man.T - self.start_ti
        elif self.direction == 'backward':
            self.start_ti -= 1
            self.length = self.start_ti + 1
        else:
            raise NotImplementedError

        self.to_tensor = ToTensor()

    def __getitem__(self, index: int):
        if self.direction == 'forward':
            ti = self.start_ti + index
        elif self.direction == 'backward':
            ti = self.start_ti - index
        else:
            raise NotImplementedError

        assert 0 <= ti < self.res_man.T

        image = self.res_man.get_image(ti)
        image_torch = self.to_tensor(image)

        return image, image_torch

    def __len__(self):
        return self.length


def get_data_loader(dataset: Dataset, num_workers: int):
    if 'linux' in sys.platform:
        loader = DataLoader(dataset,
                            batch_size=None,
                            shuffle=False,
                            num_workers=num_workers,
                            collate_fn=lambda x: x)
    else:
        print(f'Non-linux platform {sys.platform} detected, using single-threaded dataloader')
        loader = DataLoader(dataset,
                            batch_size=None,
                            shuffle=False,
                            num_workers=0,
                            collate_fn=lambda x: x)
    return loader