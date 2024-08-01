from torch import nn
from torchvision.ops import focal_loss
from typing import List, Tuple, Dict, Any, Optional
import torch


class StixelVoxelLoss(nn.Module):
    """
    Loss function for StixelObject. Input/ target shape is [batch_size, attributes, candidates, column]:
    e.g. [2, 4, 12, 240]. The inner dimension of the attributes are [vB, vT, d, P] with v = row Bottom and Top, depth d
    and Probability P.
    """
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        super(StixelVoxelLoss, self).__init__()
        if weights is None:
            self.weights = {
                'occ': 1.0,
            }
        else:
            self.weights = weights
        # Focal Loss focus more on hard samples. BCE: universal probability loss
        # self.classify_loss = focal_loss.sigmoid_focal_loss
        self.occupancy_loss: nn.BCELoss = nn.BCELoss(reduction="mean")

    def params(self) -> Dict[str, Any]:
        return self.weights

    def forward(self, inputs, targets):
        occupancy_grid_loss = self.occupancy_loss(inputs, targets) * self.weights['occ']
        return occupancy_grid_loss
