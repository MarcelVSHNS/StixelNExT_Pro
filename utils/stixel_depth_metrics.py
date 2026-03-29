from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

LOG_DEPTH_EPS = 1e-6


def anchors_to_tensor(anchors: Union[pd.DataFrame, np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Convert depth anchors to a tensor with shape [N, U]."""
    if isinstance(anchors, pd.DataFrame):
        anchor_np = anchors.to_numpy(dtype=np.float32)
        return torch.from_numpy(anchor_np)
    if isinstance(anchors, np.ndarray):
        return torch.from_numpy(anchors.astype(np.float32))
    if isinstance(anchors, torch.Tensor):
        if anchors.dtype != torch.float32:
            return anchors.to(torch.float32)
        return anchors
    raise TypeError(f"Unsupported anchors type: {type(anchors)}")


def stixel_to_depth_map(
    pred: torch.Tensor,
    anchors: Union[pd.DataFrame, np.ndarray, torch.Tensor],
    img_height: int,
    u_scale: int = 8,
    v_scale: int = 8,
    p_is_logit: bool = False,
    p_threshold: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a stixel prediction tensor [4, N, U] into a depth map [H, U] and valid mask [H, U].
    """
    del u_scale  # width U is already in stixel columns

    if pred.ndim != 3:
        raise ValueError(f"Expected pred shape [4, N, U], got {tuple(pred.shape)}")
    if pred.shape[0] != 4:
        raise ValueError("Prediction tensor must have exactly 4 attributes [vB, vT, d, P].")

    pred = pred.detach().to(torch.float32)
    anchor_t = anchors_to_tensor(anchors).to(torch.float32)

    n_classes = pred.shape[1]
    width = pred.shape[2]
    if anchor_t.ndim != 2 or anchor_t.shape[0] != n_classes or anchor_t.shape[1] != width:
        raise ValueError(f"Anchors must be [N, U]=[{n_classes}, {width}], got {tuple(anchor_t.shape)}")

    p = pred[3]
    if p_is_logit:
        p = torch.sigmoid(p)

    best_idx = torch.argmax(p, dim=0)
    best_score = p.gather(0, best_idx.unsqueeze(0)).squeeze(0)

    full_height = max(int(img_height), 1)
    stixel_height = max(full_height // int(v_scale), 1)

    depth_map = torch.zeros((stixel_height, width), dtype=torch.float32)
    mask = torch.zeros((stixel_height, width), dtype=torch.bool)

    for u in range(width):
        if p_threshold is not None and float(best_score[u].item()) < p_threshold:
            continue

        k = int(best_idx[u].item())

        vt = int(torch.round(pred[1, k, u] * full_height).item())
        vb = int(torch.round(pred[0, k, u] * full_height).item())

        vt = max(0, min(vt, full_height - 1))
        vb = max(0, min(vb, full_height - 1))
        if vt > vb:
            vt, vb = vb, vt

        vt = max(0, min(vt // int(v_scale), stixel_height - 1))
        vb = max(0, min(vb // int(v_scale), stixel_height - 1))
        if vb < vt:
            continue

        depth_val = decode_depth_residual(pred[2, k, u].item(), anchor_t[k, u].item())
        if not np.isfinite(depth_val) or depth_val <= 0.0:
            continue

        depth_map[vt: vb + 1, u] = depth_val
        mask[vt: vb + 1, u] = True

    return depth_map, mask


def decode_depth_residual(depth_residual: float, anchor_depth: float) -> float:
    anchor_depth = max(float(anchor_depth), LOG_DEPTH_EPS)
    return float(np.exp(np.log(anchor_depth) + float(depth_residual)))
