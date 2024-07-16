from torch import nn
from typing import List, Tuple, Dict, Any
import torch


class WeightedLoss:
    def __init__(self, loss_fn, factor):
        self.loss_fn = loss_fn
        self.factor = factor

    def __call__(self, inputs, targets):
        return self.factor * self.loss_fn(inputs, targets)


class StixelObjectLoss(nn.Module):
    def __init__(self, weights: Dict[str, float]):
        super(StixelObjectLoss, self).__init__()
        self.weights = weights
        # BCE: probability loss
        self.l_prob: WeightedLoss = WeightedLoss(nn.BCELoss(), self.weights["probability"])
        # MSE: bottom point position loss + stixel/ object length loss
        self.l_bot_pos: WeightedLoss = WeightedLoss(nn.MSELoss(), self.weights["bottom_pos"])
        self.l_stixel_len: WeightedLoss = WeightedLoss(nn.MSELoss(), self.weights["obj_length"])

    def params(self) -> Dict[str, Any]:
        return self.weights

    def forward(self, inputs, targets):
        return self.l_prob(inputs[:, 3, :, :], targets[:, 3, :, :]), []
