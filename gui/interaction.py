"""
Contains all the types of interaction related to the GUI
Not related to automatic evaluation in the DAVIS dataset

You can inherit the Interaction class to create new interaction types
undo is (sometimes partially) supported
"""
from typing import Tuple
import torch
import torch.nn.functional as F

from gui.click_controller import ClickController


def aggregate_wbg(prob: torch.Tensor, keep_bg: bool = False, hard: bool = False) -> torch.Tensor:
    k, h, w = prob.shape
    new_prob = torch.cat([torch.prod(1 - prob, dim=0, keepdim=True), prob], 0).clamp(1e-7, 1 - 1e-7)
    logits = torch.log((new_prob / (1 - new_prob)))

    if hard:
        # Very low temperature o((⊙﹏⊙))o 🥶
        logits *= 1000

    if keep_bg:
        return F.softmax(logits, dim=0)
    else:
        return F.softmax(logits, dim=0)[1:]


class Interaction:
    def __init__(self, image: torch.Tensor, prev_mask: torch.Tensor, true_size: Tuple[int, int],
                 controller: ClickController):
        self.image = image
        self.prev_mask = prev_mask
        self.controller = controller

        self.h, self.w = true_size

        self.out_prob = None
        self.out_mask = None

    def predict(self):
        pass


class ClickInteraction(Interaction):
    def __init__(self, image: torch.Tensor, prev_mask: torch.Tensor, true_size: Tuple[int, int],
                 controller: ClickController, tar_obj: int):
        """
        prev_mask in a prob. form
        """
        super().__init__(image, prev_mask, true_size, controller)
        self.tar_obj = tar_obj

        # negative/positive for each object
        self.pos_clicks = []
        self.neg_clicks = []

        self.first_click = True
        self.out_prob = self.prev_mask.clone()

    """
    neg - Negative interaction or not
    vis - a tuple (visualization map, pass through alpha). None if not needed.
    """

    def push_point(self, x: int, y: int, is_neg: bool) -> None:
        # Clicks
        if is_neg:
            self.neg_clicks.append((x, y))
        else:
            self.pos_clicks.append((x, y))

        # Do the prediction
        if self.first_click:
            last_obj_mask = self.prev_mask[self.tar_obj].unsqueeze(0).unsqueeze(0)
            self.obj_mask = self.controller.interact(self.image.unsqueeze(0),
                                                     x,
                                                     y,
                                                     not is_neg,
                                                     prev_mask=last_obj_mask)
        else:
            self.obj_mask = self.controller.interact(self.image.unsqueeze(0),
                                                     x,
                                                     y,
                                                     not is_neg,
                                                     prev_mask=None)

        if self.first_click:
            self.first_click = False

    def predict(self) -> torch.Tensor:
        self.out_prob = self.prev_mask.clone()
        # a small hack to allow the interacting object to overwrite existing masks
        # without remembering all the object probabilities
        self.out_prob = torch.clamp(self.out_prob, max=0.9)
        self.out_prob[self.tar_obj] = self.obj_mask
        self.out_prob = aggregate_wbg(self.out_prob[1:], keep_bg=True, hard=True)
        return self.out_prob
