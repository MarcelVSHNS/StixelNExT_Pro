from torch import nn
from torchvision.ops import focal_loss
from typing import List, Tuple, Dict, Any, Optional
import torch


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
                'prob': 1.0,
                'obj_length': 1.0,
                'bottom': 1.0,
                'depth': 1.0,
                'depth_length_ratio': 1.0
            }
        else:
            self.weights = weights
        # Focal Loss focus more on hard samples. BCE: universal probability loss
        self.classify_loss = focal_loss.sigmoid_focal_loss
        # self.class_loss: nn.BCELoss = nn.BCELoss()
        # MSE: bottom point position loss + stixel/ object length loss, ...
        self.regress_loss: nn.MSELoss = nn.MSELoss(reduction="mean")
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

    def forward(self, inputs, targets):
        # currently no matching is implemented, double loss as possible strategy
        prob_loss = self.classify_loss(inputs[:, 3, :, :], targets[:, 3, :, :], reduction="mean") * self.weights['prob']
        bottom_loss = self.regress_loss(inputs[:, 0, :, :], targets[:, 0, :, :]) * self.weights['bottom']
        top_loss = self.regress_loss(inputs[:, 1, :, :], targets[:, 1, :, :]) * self.weights['obj_length']
        depth_loss = self.regress_loss(inputs[:, 2, :, :], targets[:, 2, :, :]) * self.weights['depth']
        depth_length_ratio_loss = self.depth_length_ratio_loss_fn(inputs, targets) * self.weights['depth_length_ratio']
        # summarize all partial losses
        return prob_loss + top_loss + bottom_loss + depth_loss + depth_length_ratio_loss
