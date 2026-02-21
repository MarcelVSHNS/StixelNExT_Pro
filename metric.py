from typing import Dict

import torch


def eval_depth(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Compute common monocular depth metrics on flattened valid vectors in meters."""
    if pred.ndim != 1 or target.ndim != 1:
        pred = pred.reshape(-1)
        target = target.reshape(-1)

    pred = pred.to(torch.float64)
    target = target.to(torch.float64)

    valid = torch.isfinite(pred) & torch.isfinite(target) & (pred > 0.0) & (target > 0.0)
    pred = pred[valid]
    target = target[valid]

    if pred.numel() == 0:
        return {
            "abs_rel": float("nan"),
            "sq_rel": float("nan"),
            "rmse": float("nan"),
            "rmse_log": float("nan"),
            "log10": float("nan"),
            "silog": float("nan"),
            "d1": float("nan"),
            "d2": float("nan"),
            "d3": float("nan"),
            "mae": float("nan"),
            "n_valid": 0.0,
        }

    diff = pred - target
    abs_diff = torch.abs(diff)

    abs_rel = torch.mean(abs_diff / target)
    sq_rel = torch.mean((diff ** 2) / target)
    rmse = torch.sqrt(torch.mean(diff ** 2))

    log_pred = torch.log(pred)
    log_tgt = torch.log(target)
    log_diff = log_pred - log_tgt

    rmse_log = torch.sqrt(torch.mean(log_diff ** 2))
    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
    silog = torch.sqrt(torch.mean(log_diff ** 2) - torch.mean(log_diff) ** 2) * 100.0

    ratio = torch.maximum(pred / target, target / pred)
    d1 = torch.mean((ratio < 1.25).to(torch.float64))
    d2 = torch.mean((ratio < (1.25 ** 2)).to(torch.float64))
    d3 = torch.mean((ratio < (1.25 ** 3)).to(torch.float64))
    mae = torch.mean(abs_diff)

    return {
        "abs_rel": float(abs_rel.item()),
        "sq_rel": float(sq_rel.item()),
        "rmse": float(rmse.item()),
        "rmse_log": float(rmse_log.item()),
        "log10": float(log10.item()),
        "silog": float(silog.item()),
        "d1": float(d1.item()),
        "d2": float(d2.item()),
        "d3": float(d3.item()),
        "mae": float(mae.item()),
        "n_valid": float(pred.numel()),
    }
