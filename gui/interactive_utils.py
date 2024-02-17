# Modified from https://github.com/seoungwugoh/ivs-demo

from typing import Literal, List
import numpy as np

import torch
import torch.nn.functional as F
from cutie.utils.palette import davis_palette


def image_to_torch(frame: np.ndarray, device: str = 'cuda'):
    # frame: H*W*3 numpy array
    frame = frame.transpose(2, 0, 1)
    frame = torch.from_numpy(frame).float().to(device, non_blocking=True) / 255
    return frame


def torch_prob_to_numpy_mask(prob: torch.Tensor):
    mask = torch.max(prob, dim=0).indices
    mask = mask.cpu().numpy().astype(np.uint8)
    return mask


def index_numpy_to_one_hot_torch(mask: np.ndarray, num_classes: int):
    mask = torch.from_numpy(mask).long()
    return F.one_hot(mask, num_classes=num_classes).permute(2, 0, 1).float()


"""
Some constants fro visualization
"""
try:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
except:
    device = torch.device("cpu")

color_map_np = np.frombuffer(davis_palette, dtype=np.uint8).reshape(-1, 3).copy()
# scales for better visualization
color_map_np = (color_map_np.astype(np.float32) * 1.5).clip(0, 255).astype(np.uint8)
color_map = color_map_np.tolist()
color_map_torch = torch.from_numpy(color_map_np).to(device) / 255

grayscale_weights = np.array([[0.3, 0.59, 0.11]]).astype(np.float32)
grayscale_weights_torch = torch.from_numpy(grayscale_weights).to(device).unsqueeze(0)


def get_visualization(mode: Literal['image', 'mask', 'fade', 'davis', 'light', 'popup', 'layer',
                                    'rgba'], image: np.ndarray, mask: np.ndarray, layer: np.ndarray,
                      target_objects: List[int]) -> np.ndarray:
    if mode == 'image':
        return image
    elif mode == 'mask':
        return color_map_np[mask]
    elif mode == 'fade':
        return overlay_davis(image, mask, fade=True)
    elif mode == 'davis':
        return overlay_davis(image, mask)
    elif mode == 'light':
        return overlay_davis(image, mask, 0.9)
    elif mode == 'popup':
        return overlay_popup(image, mask, target_objects)
    elif mode == 'layer':
        if layer is None:
            print('Layer file not given. Defaulting to DAVIS.')
            return overlay_davis(image, mask)
        else:
            return overlay_layer(image, mask, layer, target_objects)
    elif mode == 'rgba':
        return overlay_rgba(image, mask, target_objects)
    else:
        raise NotImplementedError


def get_visualization_torch(mode: Literal['image', 'mask', 'fade', 'davis', 'light', 'popup',
                                          'layer', 'rgba'], image: torch.Tensor, prob: torch.Tensor,
                            layer: torch.Tensor, target_objects: List[int]) -> np.ndarray:
    if mode == 'image':
        return image
    elif mode == 'mask':
        mask = torch.max(prob, dim=0).indices
        return (color_map_torch[mask] * 255).byte().cpu().numpy()
    elif mode == 'fade':
        return overlay_davis_torch(image, prob, fade=True)
    elif mode == 'davis':
        return overlay_davis_torch(image, prob)
    elif mode == 'light':
        return overlay_davis_torch(image, prob, 0.9)
    elif mode == 'popup':
        return overlay_popup_torch(image, prob, target_objects)
    elif mode == 'layer':
        if layer is None:
            print('Layer file not given. Defaulting to DAVIS.')
            return overlay_davis_torch(image, prob)
        else:
            return overlay_layer_torch(image, prob, layer, target_objects)
    elif mode == 'rgba':
        return overlay_rgba_torch(image, prob, target_objects)
    else:
        raise NotImplementedError


def overlay_davis(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5, fade: bool = False):
    """ Overlay segmentation on top of RGB image. from davis official"""
    im_overlay = image.copy()

    colored_mask = color_map_np[mask]
    foreground = image * alpha + (1 - alpha) * colored_mask
    binary_mask = (mask > 0)
    # Compose image
    im_overlay[binary_mask] = foreground[binary_mask]
    if fade:
        im_overlay[~binary_mask] = im_overlay[~binary_mask] * 0.6
    return im_overlay.astype(image.dtype)


def overlay_popup(image: np.ndarray, mask: np.ndarray, target_objects: List[int]):
    # Keep foreground colored. Convert background to grayscale.
    im_overlay = image.copy()

    binary_mask = ~(np.isin(mask, target_objects))
    colored_region = (im_overlay[binary_mask] * grayscale_weights).sum(-1, keepdims=-1)
    im_overlay[binary_mask] = colored_region
    return im_overlay.astype(image.dtype)


def overlay_layer(image: np.ndarray, mask: np.ndarray, layer: np.ndarray,
                  target_objects: List[int]):
    # insert a layer between foreground and background
    # The CPU version is less accurate because we are using the hard mask
    # The GPU version has softer edges as it uses soft probabilities
    obj_mask = (np.isin(mask, target_objects)).astype(np.float32)[:, :, np.newaxis]
    layer_alpha = layer[:, :, 3].astype(np.float32)[:, :, np.newaxis] / 255
    layer_rgb = layer[:, :, :3]
    background_alpha = (1 - obj_mask) * (1 - layer_alpha)
    im_overlay = (image * background_alpha + layer_rgb * (1 - obj_mask) * layer_alpha +
                  image * obj_mask).clip(0, 255)
    return im_overlay.astype(image.dtype)


def overlay_rgba(image: np.ndarray, mask: np.ndarray, target_objects: List[int]):
    # Put the mask is in the alpha channel
    obj_mask = (np.isin(mask, target_objects)).astype(np.float32)[:, :, np.newaxis] * 255
    im_overlay = np.concatenate([image, obj_mask], axis=-1)
    return im_overlay.astype(image.dtype)


def overlay_davis_torch(image: torch.Tensor,
                        prob: torch.Tensor,
                        alpha: float = 0.5,
                        fade: bool = False):
    """ Overlay segmentation on top of RGB image. from davis official"""
    # Changes the image in-place to avoid copying
    # NOTE: Make sure you no longer use image after calling this function
    image = image.permute(1, 2, 0)
    im_overlay = image
    mask = torch.max(prob, dim=0).indices

    colored_mask = color_map_torch[mask]
    foreground = image * alpha + (1 - alpha) * colored_mask
    binary_mask = (mask > 0)
    # Compose image
    im_overlay[binary_mask] = foreground[binary_mask]
    if fade:
        im_overlay[~binary_mask] = im_overlay[~binary_mask] * 0.6

    im_overlay = (im_overlay * 255).byte().cpu().numpy()
    return im_overlay


def overlay_popup_torch(image: torch.Tensor, prob: torch.Tensor, target_objects: List[int]):
    # Keep foreground colored. Convert background to grayscale.
    image = image.permute(1, 2, 0)

    if len(target_objects) == 0:
        obj_mask = torch.zeros_like(prob[0]).unsqueeze(2)
    else:
        # I should not need to convert this to numpy.
        # Using list works most of the time but consistently fails
        # if I include first object -> exclude it -> include it again.
        # I check everywhere and it makes absolutely no sense.
        # I am blaming this on PyTorch and calling it a day
        obj_mask = prob[np.array(target_objects, dtype=np.int32)].sum(0).unsqueeze(2)
    gray_image = (image * grayscale_weights_torch).sum(-1, keepdim=True)
    im_overlay = obj_mask * image + (1 - obj_mask) * gray_image

    im_overlay = (im_overlay * 255).byte().cpu().numpy()
    return im_overlay


def overlay_layer_torch(image: torch.Tensor, prob: torch.Tensor, layer: torch.Tensor,
                        target_objects: List[int]):
    # insert a layer between foreground and background
    # The CPU version is less accurate because we are using the hard mask
    # The GPU version has softer edges as it uses soft probabilities
    image = image.permute(1, 2, 0)

    if len(target_objects) == 0:
        obj_mask = torch.zeros_like(prob[0]).unsqueeze(2)
    else:
        # TODO: figure out why we need to convert this to numpy array
        obj_mask = prob[np.array(target_objects, dtype=np.int32)].sum(0).unsqueeze(2)
    layer_alpha = layer[:, :, 3].unsqueeze(2)
    layer_rgb = layer[:, :, :3]
    # background_alpha = torch.maximum(obj_mask, layer_alpha)
    background_alpha = (1 - obj_mask) * (1 - layer_alpha)
    im_overlay = (image * background_alpha + layer_rgb * (1 - obj_mask) * layer_alpha +
                  image * obj_mask).clip(0, 1)

    im_overlay = (im_overlay * 255).byte().cpu().numpy()
    return im_overlay


def overlay_rgba_torch(image: torch.Tensor, prob: torch.Tensor, target_objects: List[int]):
    image = image.permute(1, 2, 0)

    if len(target_objects) == 0:
        obj_mask = torch.zeros_like(prob[0]).unsqueeze(2)
    else:
        # TODO: figure out why we need to convert this to numpy array
        obj_mask = prob[np.array(target_objects, dtype=np.int32)].sum(0).unsqueeze(2)

    im_overlay = torch.cat([image, obj_mask], dim=-1).clip(0, 1)
    im_overlay = (im_overlay * 255).byte().cpu().numpy()
    return im_overlay
