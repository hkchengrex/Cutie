import torch
from torch import distributed as dist
from torch.utils import data


def get_rank():
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def synchronize():
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return
    dist.barrier()


def get_world_size():
    if not dist.is_available() or not dist.is_initialized():
        return 1

    return dist.get_world_size()


def reduce_loss_dict(loss_dict):
    world_size = get_world_size()

    if world_size < 2:
        return loss_dict

    with torch.no_grad():
        keys = []
        losses = []

        for k in loss_dict.keys():
            keys.append(k)
            losses.append(loss_dict[k])

        losses = torch.stack(losses, 0)
        dist.reduce(losses, dst=0)

        if dist.get_rank() == 0:
            losses /= world_size

        reduced_losses = {k: v for k, v in zip(keys, losses)}

    return reduced_losses


def get_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)


def get_dp_wrapper(distributed):
    class DPWrapper(torch.nn.parallel.DistributedDataParallel if distributed else torch.nn.DataParallel):
        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)
    return DPWrapper
