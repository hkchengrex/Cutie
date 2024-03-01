from os import path, listdir
from omegaconf import DictConfig, open_dict
from hydra import compose, initialize

import torch

from cutie.model.cutie import CUTIE
from cutie.inference.inference_core import InferenceCore
from cutie.inference.utils.results_utils import ResultSaver

from tqdm import tqdm

from time import perf_counter
import cv2
from gui.interactive_utils import image_to_torch, index_numpy_to_one_hot_torch
import numpy as np
from PIL import Image

from argparse import ArgumentParser


def process_video(cfg: DictConfig):
    # general setup
    torch.set_grad_enabled(False)
    if cfg['device'] == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    elif cfg['device'] == 'mps' and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'Using device: {device}')

    use_amp = cfg.amp

    # Load the network weights
    print(f'Loading Cutie and weights')
    cutie = CUTIE(cfg).to(device).eval()
    if cfg.weights is not None:
        model_weights = torch.load(cfg.weights, map_location=device)
        cutie.load_weights(model_weights)
    else:
        print('No model weights loaded. Are you sure about this?')

    # Open video
    video = cfg['video']
    if video is None:
        print('No video defined. Please specify!')
        exit()
    video_name = path.splitext(video)[0]

    print(f'Opening video {video}')
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f'Unable to open video {video}!')
        exit()
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initial mask handling
    mask_dir = cfg['mask_dir']
    if mask_dir is None:
        print('No mask_dir defined. Please specify!')
        exit()

    # determine if the mask uses 3-channel long ID or 1-channel (0~255) short ID
    all_mask_frames = sorted(listdir(mask_dir))
    first_mask_frame = all_mask_frames[0]
    first_mask = Image.open(path.join(mask_dir, first_mask_frame))

    if first_mask.mode == 'P':
        use_long_id = False
        palette = first_mask.getpalette()
    elif first_mask.mode == 'RGB':
        use_long_id = True
        palette = None
    elif first_mask.mode == 'L':
        use_long_id = False
        palette = None
    else:
        print(f'Unknown mode {first_mask.mode} in {first_mask_frame}.')
        exit()

    num_objects = cfg['num_objects']
    if num_objects is None or num_objects < 1:
        num_objects = len(np.unique(first_mask)) - 1

    processor = InferenceCore(cutie, cfg=cfg)

    # First commit mask input into permanent memory
    num_masks = len(all_mask_frames)
    if num_masks == 0:
        print(f'No mask frames found!')
        exit()

    with torch.inference_mode():
        with torch.amp.autocast(device, enabled=(use_amp and device == 'cuda')):
            pbar = tqdm(total=num_masks)
            pbar.set_description('Commiting masks into permenent memory')
            for mask_name in all_mask_frames:
                mask = Image.open(path.join(mask_dir, mask_name))
                frame_number = int(mask_name[:-4])
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

                # load frame matching mask
                _, frame = cap.read()
                if frame is None:
                    break

                # convert numpy array to pytorch tensor format
                frame_torch = image_to_torch(frame, device=device)

                mask_np = np.array(mask)
                mask_torch = index_numpy_to_one_hot_torch(mask_np, num_objects + 1).to(device)

                # the background mask is fed into the model
                prob = processor.step(frame_torch,
                                      mask_torch[1:],
                                      idx_mask=False,
                                      force_permanent=True)

                pbar.update(1)

    # Next start inference on video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # reset frame reading
    total_process_time = 0
    current_frame_index = 0
    mask_output_root = cfg['output_dir']
    saver = ResultSaver(mask_output_root,
                        '',
                        dataset='',
                        object_manager=processor.object_manager,
                        use_long_id=use_long_id,
                        palette=palette)
    mem_cleanup_ratio = cfg['mem_cleanup_ratio']

    with torch.inference_mode():
        with torch.amp.autocast(device, enabled=(use_amp and device == 'cuda')):
            pbar = tqdm(total=total_frame_count)
            pbar.set_description(f'Processing video {video}')
            while (cap.isOpened()):
                # load frame-by-frame
                _, frame = cap.read()
                if frame is None or current_frame_index > total_frame_count:
                    break

                # timing start
                if 'cuda' in device:
                    torch.cuda.synchronize(device)
                    start = torch.cuda.Event(enable_timing=True) if device == 'cuda' else None
                    end = torch.cuda.Event(enable_timing=True) if device == 'cuda' else None
                    start.record()
                else:
                    a = perf_counter()

                frame_name = f'{current_frame_index:07d}' + '.png'

                # check if we have a mask to load
                mask = None
                mask_path = path.join(mask_dir, frame_name)
                if path.exists(mask_path):
                    mask = Image.open(mask_path)

                # convert numpy array to pytorch tensor format
                frame_torch = image_to_torch(frame, device=device)
                if mask is not None:
                    # initialize with the mask
                    mask_np = np.array(mask)
                    mask_torch = index_numpy_to_one_hot_torch(mask_np, num_objects + 1).to(device)
                    # the background mask is fed into the model
                    prob = processor.step(frame_torch, mask_torch[1:], idx_mask=False)
                else:
                    # propagate only
                    prob = processor.step(frame_torch)

                # timing end
                if 'cuda' in device:
                    end.record()
                    torch.cuda.synchronize(device)
                    total_process_time += (start.elapsed_time(end) / 1000)
                else:
                    b = perf_counter()
                    total_process_time += (b - a)

                saver.process(prob,
                              frame_name,
                              resize_needed=False,
                              shape=None,
                              last_frame=(current_frame_index == total_frame_count - 1),
                              path_to_image=None)

                check_to_clear_non_permanent_cuda_memory(processor=processor,
                                                         device=device,
                                                         mem_cleanup_ratio=mem_cleanup_ratio)

                current_frame_index += 1
                pbar.update(1)

    pbar.close()
    cap.release()  # Release the video capture object
    saver.end()  # Wait for saving to finish

    print(
        '------------------------------------------------------------------------------------------------------------------------------------------------'
    )
    print(f'Total processing time: {total_process_time}')
    print(f'Total processed frames: {current_frame_index}')
    print(f'FPS: {current_frame_index / total_process_time}')
    print(f'Max allocated memory (MB): {torch.cuda.max_memory_allocated() / (2**20)}'
          ) if device == 'cuda' else None
    print(
        '------------------------------------------------------------------------------------------------------------------------------------------------'
    )


def check_to_clear_non_permanent_cuda_memory(processor: InferenceCore, device, mem_cleanup_ratio):
    if 'cuda' in device:
        if mem_cleanup_ratio > 0 and mem_cleanup_ratio <= 1:
            info = torch.cuda.mem_get_info()

            global_free, global_total = info
            global_free /= (2**30)  # GB
            global_total /= (2**30)  # GB
            global_used = global_total - global_free
            #mem_ratio = round(global_used / global_total * 100)
            mem_ratio = global_used / global_total
            if mem_ratio > mem_cleanup_ratio:
                print(f'GPU cleanup triggered: {mem_ratio} > {mem_cleanup_ratio}')
                processor.clear_non_permanent_memory()
                torch.cuda.empty_cache()


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument('-v', '--video', help='Video file.', default=None)
    parser.add_argument(
        '-m',
        '--mask_dir',
        help=
        'Directory with mask files. Must be named with with corresponding video frame number syntax [07d].',
        default=None)
    parser.add_argument('-o',
                        '--output_dir',
                        help='Directory where processed mask files will be saved.',
                        default=None)
    parser.add_argument('-d',
                        '--device',
                        help='Target device for processing [cuda, cpu].',
                        default='cuda')
    parser.add_argument(
        '--mem_every',
        help='How often to update working memory; higher number speeds up processing.',
        type=int,
        default='10')
    parser.add_argument(
        '--max_internal_size',
        help=
        'maximum internal processing size; reducing this speeds up processing; -1 means no resizing.',
        type=int,
        default='480')
    parser.add_argument(
        '--mem_cleanup_ratio',
        help=
        'How often to clear non permanent GPU memory; when ratio of GPU memory used is above given mem_cleanup_ratio [0;1] then cleanup is triggered; only used when device=cuda.',
        type=float,
        default='-1')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # input arguments
    args = get_arguments()

    # getting hydra's config without using its decorator
    initialize(version_base='1.3.2', config_path="cutie/config", job_name="process_video")
    cfg = compose(config_name="video_config")

    # merge arguments into config
    args = vars(args)
    with open_dict(cfg):
        for k, v in args.items():
            cfg[k] = v

    process_video(cfg)
