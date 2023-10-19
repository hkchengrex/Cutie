import os
from typing import List

from PIL import Image
import numpy as np
import cv2
import av


def convert_frames_to_video(
        image_folder: str,
        output_path: str,
        fps: int = 24,
        bitrate: int = 1,  # in Mbps
        progress_callback=None) -> None:
    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    output = av.open(output_path, mode="w")

    stream = output.add_stream("h264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.bit_rate = bitrate * (10**7)

    for i, img_path in enumerate(images):
        img = cv2.imread(os.path.join(image_folder, img_path))
        frame = av.VideoFrame.from_ndarray(img, format='bgr24')
        packet = stream.encode(frame)
        output.mux(packet)

        if progress_callback is not None and i % 10 == 0:
            progress_callback(i / len(images))

    # flush
    packet = stream.encode(None)
    output.mux(packet)

    output.close()


def convert_mask_to_binary(mask_folder: str,
                           output_path: str,
                           target_objects: List[int],
                           progress_callback=None) -> None:
    masks = [img for img in sorted(os.listdir(mask_folder)) if img.endswith(".png")]

    for i, mask_path in enumerate(masks):
        mask = Image.open(os.path.join(mask_folder, mask_path))
        mask = np.array(mask)
        mask = np.where(np.isin(mask, target_objects), 255, 0)
        cv2.imwrite(os.path.join(output_path, mask_path), mask)

        if progress_callback is not None and i % 10 == 0:
            progress_callback(i / len(masks))
