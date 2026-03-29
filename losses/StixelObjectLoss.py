import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, Any, Optional


class RobustExistenceLoss(nn.Module):
    def __init__(
        self,
        min_alpha: float = 0.25,
        max_alpha: float = 1.0,
        n_cand: int = 64,
        label_smoothing: float = 0.05,
        neighbor_smoothing: float = 0.10,
        positive_weight: float = 1.0,
        negative_weight: float = 0.25,
    ):
        super(RobustExistenceLoss, self).__init__()
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.max_depth = n_cand
        self.label_smoothing = label_smoothing
        self.neighbor_smoothing = neighbor_smoothing
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight

    def _smooth_targets(self, targets: torch.Tensor) -> torch.Tensor:
        if self.neighbor_smoothing <= 0.0:
            smoothed = targets
        else:
            bsz, depth_bins, width = targets.shape
            flat_targets = targets.permute(0, 2, 1).reshape(-1, 1, depth_bins)
            kernel = targets.new_tensor([0.25, 0.5, 0.25], dtype=targets.dtype).view(1, 1, 3)
            blurred = F.conv1d(flat_targets, kernel, padding=1)
            blurred = blurred.reshape(bsz, width, depth_bins).permute(0, 2, 1)
            smoothed = (1.0 - self.neighbor_smoothing) * targets + self.neighbor_smoothing * blurred

        if self.label_smoothing > 0.0:
            smoothed = smoothed * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        return smoothed.clamp(0.0, 1.0)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # avoid log(0) / log(1) in BCE terms
        inputs = inputs.clamp(min=1e-6, max=1 - 1e-6)
        targets = self._smooth_targets(targets)
        depth_indices = torch.arange(self.max_depth, device=inputs.device).unsqueeze(0).unsqueeze(-1)

        alpha = self.min_alpha + (depth_indices / self.max_depth) * (
                self.max_alpha - self.min_alpha)
        alpha = alpha.expand(inputs.size(0), self.max_depth, inputs.size(2))

        bce_loss = -(targets * torch.log(inputs) + (1.0 - targets) * torch.log(1.0 - inputs))
        class_weights = targets * self.positive_weight + (1.0 - targets) * self.negative_weight
        weighted_bce_loss = alpha * class_weights * bce_loss

        return weighted_bce_loss.mean()


class StixelObjectLoss(nn.Module):
    """
    Loss function for StixelObject. Input/ target shape is [batch_size, attributes, candidates, column]:
    e.g. [2, 4, 12, 240]. The inner dimension of the attributes are [vB, vT, d, P] with v = row Bottom and Top, depth d
    and Probability P.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        classify_cfg: Optional[Dict[str, float]] = None,
    ):
        super(StixelObjectLoss, self).__init__()
        default_weights = {
            'P': 1.0,
            'vT': 1.0,
            'vB': 1.0,
            'd': 1.0,
            'h_d_ratio': 1.0
        }
        if weights is None:
            self.weights = default_weights
        else:
            self.weights = {**default_weights, **weights}
        classify_defaults = {
            "min_alpha": 1.0,
            "max_alpha": 1.5,
            "n_cand": 64,
            "label_smoothing": 0.05,
            "neighbor_smoothing": 0.10,
            "positive_weight": 1.0,
            "negative_weight": 0.25,
        }
        self.classify_cfg = {**classify_defaults, **(classify_cfg or {})}
        self.classify_loss = RobustExistenceLoss(**self.classify_cfg)
        # Use SmoothL1 for all regression terms to reduce sensitivity to noisy pseudo labels.
        self.regress_loss: nn.SmoothL1Loss = nn.SmoothL1Loss(reduction="none")
        self.depth_regress_loss: nn.SmoothL1Loss = nn.SmoothL1Loss(reduction="none")
        self.last_components: Dict[str, float] = {}

    def params(self) -> Dict[str, Any]:
        return {
            "weights": self.weights,
            "classify_cfg": self.classify_cfg,
        }

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
        vb_idx, vt_idx, d_idx, p_idx = 0, 1, 2, 3
        # masking for partial loss
        mask = (targets[:, p_idx, :, :] > 0).float()

        # currently no matching is implemented, double loss as possible strategy
        depth_bin_loss = self.classify_loss(inputs[:, p_idx, :, :], targets[:, p_idx, :, :]) * self.weights['P']
        bottom_loss = self.regress_loss(inputs[:, vb_idx, :, :], targets[:, vb_idx, :, :]) * self.weights['vB']
        top_loss = self.regress_loss(inputs[:, vt_idx, :, :], targets[:, vt_idx, :, :]) * self.weights['vT']
        depth_loss = self.depth_regress_loss(inputs[:, d_idx, :, :], targets[:, d_idx, :, :]) * self.weights['d']
        # depth_length_ratio_loss = self.depth_length_ratio_loss_fn(inputs, targets) * self.weights['depth_length_ratio']

        # summarize all partial losses and apply mask
        masked_seg_loss = (top_loss + bottom_loss + depth_loss) * mask
        mask_sum = mask.sum()
        if mask_sum > 0:
            seg_loss = masked_seg_loss.sum() / mask_sum
        else:
            seg_loss = torch.tensor(0.0, device=inputs.device)
        total_loss = depth_bin_loss + seg_loss
        self.last_components = {
            "P": float(depth_bin_loss.detach().item()),
            "vB": float((bottom_loss * mask).sum().detach().item() / max(float(mask_sum.detach().item()), 1.0)),
            "vT": float((top_loss * mask).sum().detach().item() / max(float(mask_sum.detach().item()), 1.0)),
            "d": float((depth_loss * mask).sum().detach().item() / max(float(mask_sum.detach().item()), 1.0)),
            "seg": float(seg_loss.detach().item()),
            "total": float(total_loss.detach().item()),
        }
        return total_loss  # + depth_length_ratio_loss
