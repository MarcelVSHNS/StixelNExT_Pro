from sympy.abc import alpha
from torch import nn
from torchvision.ops import focal_loss
from functools import partial
from typing import List, Tuple, Dict, Any, Optional
import torch


class DepthWeightedBCELoss(nn.Module):
    def __init__(self, min_alpha=0.25, max_alpha=1.0, n_cand=64):
        super(DepthWeightedBCELoss, self).__init__()
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.max_depth = n_cand

    def forward(self, inputs, targets):
        # avoid log(0) / log(1) in BCE terms
        inputs = inputs.clamp(min=1e-6, max=1 - 1e-6)
        depth_indices = torch.arange(self.max_depth, device=inputs.device).unsqueeze(0).unsqueeze(-1)

        alpha = self.min_alpha + (depth_indices / self.max_depth) * (
                self.max_alpha - self.min_alpha)
        alpha = alpha.expand(inputs.size(0), self.max_depth, inputs.size(2))  #

        bce_loss = - (targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs))
        weighted_bce_loss = alpha * bce_loss

        return weighted_bce_loss.mean()


class StixelObjectLoss(nn.Module):
    """
    Loss function for StixelObject. Input/ target shape is [batch_size, attributes, candidates, column]:
    e.g. [2, 4, 12, 240]. The inner dimension of the attributes are [vB, vT, d, P] with v = row Bottom and Top, depth d
    and Probability P.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        super(StixelObjectLoss, self).__init__()
        if weights is None:
            self.weights = {
                'P': 1.0,
                'vT': 1.0,
                'vB': 1.0,
                'd': 1.0,
                'h_d_ratio': 1.0
            }
        else:
            self.weights = weights
        # Focal Loss focus more on hard samples. BCE: universal probability loss
        # self.classify_loss = partial(focal_loss.sigmoid_focal_loss, reduction='mean')
        self.classify_loss: DepthWeightedBCELoss = DepthWeightedBCELoss(min_alpha=1, max_alpha=2, n_cand=64)
        # self.classify_loss: nn.BCELoss = nn.BCELoss(reduction="mean")
        # MSE: bottom point position loss + stixel/ object length loss, ...
        self.regress_loss: nn.MSELoss = nn.MSELoss(reduction="none")
        # self.regress_loss: nn.SmoothL1Loss = nn.SmoothL1Loss()

    def params(self) -> Dict[str, Any]:
        return self.weights

    def calc_height(self):
        pass

    # initial guess: reduce weight factor
    def depth_length_ratio_loss_fn(self, inputs, targets, epsilon=1e-6):
        # calc h, divide by d + epsilon, apply MSE
        h_in = inputs[:, 1, :, :] - inputs[:, 0, :, :]
        h_targ = targets[:, 1, :, :] - targets[:, 0, :, :]
        depth_in = inputs[:, 2, :, :]
        depth_targ = targets[:, 2, :, :]

        # Calculate ratios with broadcasting (element-wise division)
        ratio_in = h_in / (depth_in + epsilon)
        ratio_targ = h_targ / (depth_targ + epsilon)
        return self.regress_loss(ratio_in, ratio_targ)

    def bottom_point_depth_relation(self, inputs, targets):
        pass

    def forward(self, inputs, targets, vb_idx=0, vt_idx=1, p_idx=2):
        # masking for partial loss
        mask = (targets[:, p_idx, :, :] > 0).float()

        # currently no matching is implemented, double loss as possible strategy
        depth_bin_loss = self.classify_loss(inputs[:, p_idx, :, :], targets[:, p_idx, :, :]) * self.weights['P']
        bottom_loss = self.regress_loss(inputs[:, vb_idx, :, :], targets[:, vb_idx, :, :]) * self.weights['vB']
        top_loss = self.regress_loss(inputs[:, vt_idx, :, :], targets[:, vt_idx, :, :]) * self.weights['vT']
        # depth_loss = self.regress_loss(inputs[:, 2, :, :], targets[:, 2, :, :]) * self.weights['d']
        # depth_length_ratio_loss = self.depth_length_ratio_loss_fn(inputs, targets) * self.weights['depth_length_ratio']

        # summarize all partial losses and apply mask
        masked_seg_loss = (top_loss + bottom_loss) * mask
        mask_sum = mask.sum()
        if mask_sum > 0:
            seg_loss = masked_seg_loss.sum() / mask_sum
        else:
            seg_loss = torch.tensor(0.0, device=inputs.device)
        return depth_bin_loss + seg_loss  # + depth_loss  + depth_length_ratio_loss
