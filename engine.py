import torch
import torch.distributed as dist
from typing import Any, Dict, Tuple

from metric import eval_depth
from utils.stixel_depth_metrics import stixel_to_depth_map


# Training Function
def train_one_epoch(dataloader, model, loss_fn, optimizer, device, epoch: int, writer=None,
                    log_every: int = 100) -> float:
    num_batches = len(dataloader)
    train_loss = 0.0
    component_sum: Dict[str, float] = {}
    model.train()

    for batch_idx, (samples, targets, _) in enumerate(dataloader):
        samples = samples.to(device)
        targets = targets.to(device)

        outputs = model(samples)
        loss = loss_fn(outputs, targets)

        train_loss += float(loss.item())
        if hasattr(loss_fn, "last_components"):
            for key, value in getattr(loss_fn, "last_components").items():
                component_sum[key] = component_sum.get(key, 0.0) + float(value)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if writer is not None and log_every > 0 and (batch_idx % log_every == 0):
            global_step = epoch * num_batches + batch_idx
            writer.add_scalar("Loss/train_step", float(loss.item()), global_step)
            # log the first param group's lr
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], global_step)

        if batch_idx % 100 == 0:
            # optional: make this global progress nicer
            print(f"[rank {device}] epoch {epoch} batch {batch_idx}/{num_batches} loss={loss.item():.6f}")

    train_loss /= max(num_batches, 1)

    if writer is not None:
        writer.add_scalar("Loss/train", train_loss, epoch)
        for key in sorted(component_sum.keys()):
            writer.add_scalar(f"Loss/train_{key}", component_sum[key] / max(num_batches, 1), epoch)

    return train_loss


# Validation Function
def _unpack_eval_batch(batch: Tuple[Any, ...]) -> Tuple[torch.Tensor, torch.Tensor, Any, Any, Any]:
    if not isinstance(batch, (tuple, list)):
        raise TypeError("Expected batch to be a tuple/list.")
    if len(batch) == 3:
        samples, targets, paths = batch
        return samples, targets, paths, None, None
    if len(batch) >= 5:
        samples, targets, paths, gt_depth_map, gt_mask = batch[:5]
        return samples, targets, paths, gt_depth_map, gt_mask
    raise ValueError(f"Unexpected batch structure with length={len(batch)}")


def evaluate(
    dataloader,
    model,
    loss_fn,
    device,
    epoch: int,
    writer=None,
    min_depth: float = 1e-3,
    max_depth: float = 80.0,
    min_valid_pixels: int = 10,
    v_scale: int = 8,
    p_is_logit: bool = False,
    p_threshold: float = None,
) -> float:
    num_batches = len(dataloader)
    model.eval()
    eval_loss = 0.0
    component_sum: Dict[str, float] = {}
    metric_sum: Dict[str, float] = {}
    metric_count: Dict[str, int] = {}

    dataset_anchors = getattr(dataloader.dataset, "depth_anchors", None)

    with torch.no_grad():
        for batch in dataloader:
            samples, targets, _, gt_depth_map, gt_mask = _unpack_eval_batch(batch)
            samples = samples.to(device)
            targets = targets.to(device)

            outputs = model(samples)
            loss = loss_fn(outputs, targets)
            eval_loss += float(loss.item())
            if hasattr(loss_fn, "last_components"):
                for key, value in getattr(loss_fn, "last_components").items():
                    component_sum[key] = component_sum.get(key, 0.0) + float(value)

            # Depth metrics are optional and require GT depth maps from the dataset.
            if gt_depth_map is None or gt_mask is None or dataset_anchors is None:
                continue

            pred_cpu = outputs.detach().cpu()
            gt_depth_cpu = gt_depth_map.detach().cpu().to(torch.float32)
            gt_mask_cpu = gt_mask.detach().cpu().to(torch.bool)

            batch_size = pred_cpu.shape[0]
            for i in range(batch_size):
                pred_stixel = pred_cpu[i]
                gt_depth_i = gt_depth_cpu[i]
                gt_mask_i = gt_mask_cpu[i]
                img_height = int(gt_depth_i.shape[0] * int(v_scale))

                pred_depth_i, pred_mask_i = stixel_to_depth_map(
                    pred=pred_stixel,
                    anchors=dataset_anchors,
                    img_height=img_height,
                    v_scale=v_scale,
                    p_is_logit=p_is_logit,
                    p_threshold=p_threshold,
                )

                h = min(pred_depth_i.shape[0], gt_depth_i.shape[0])
                w = min(pred_depth_i.shape[1], gt_depth_i.shape[1])
                pred_depth_i = pred_depth_i[:h, :w]
                pred_mask_i = pred_mask_i[:h, :w]
                gt_depth_i = gt_depth_i[:h, :w]
                gt_mask_i = gt_mask_i[:h, :w]

                valid = (
                    pred_mask_i
                    & gt_mask_i
                    & torch.isfinite(pred_depth_i)
                    & torch.isfinite(gt_depth_i)
                    & (pred_depth_i > 0.0)
                    & (gt_depth_i > float(min_depth))
                    & (gt_depth_i < float(max_depth))
                )

                if int(valid.sum().item()) < int(min_valid_pixels):
                    continue

                metrics = eval_depth(pred_depth_i[valid], gt_depth_i[valid])
                for key, value in metrics.items():
                    if key == "n_valid":
                        continue
                    if value != value or value == float("inf") or value == float("-inf"):
                        continue
                    metric_sum[key] = metric_sum.get(key, 0.0) + float(value)
                    metric_count[key] = metric_count.get(key, 0) + 1

    eval_loss /= max(num_batches, 1)

    print(f"Validation Error: Avg loss: {eval_loss:.6f}")

    if writer is not None:
        writer.add_scalar("Loss/val", eval_loss, epoch)
        for key in sorted(component_sum.keys()):
            writer.add_scalar(f"Loss/val_{key}", component_sum[key] / max(num_batches, 1), epoch)
        for key in sorted(metric_sum.keys()):
            cnt = max(metric_count.get(key, 0), 1)
            writer.add_scalar(f"eval/{key}", metric_sum[key] / cnt, epoch)

    if len(metric_sum) > 0:
        avg_metrics = {k: metric_sum[k] / max(metric_count.get(k, 0), 1) for k in sorted(metric_sum.keys())}
        metric_str = ", ".join([f"{k}={v:.6f}" for k, v in avg_metrics.items()])
        print(f"Validation Depth Metrics: {metric_str}")
    else:
        print("Validation Depth Metrics: skipped (no valid depth samples or no GT depth maps provided).")

    return eval_loss


class EarlyStopping:
    def __init__(self, tolerance: int = 5, min_delta: float = 0.0):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def check_stop(self, validation_loss, rank: int) -> bool:
        # ensure python float on rank 0 for comparison
        if isinstance(validation_loss, torch.Tensor):
            validation_loss = float(validation_loss.detach().cpu().item())

        if rank == 0:
            improvement = self.best_loss - validation_loss
            if improvement > self.min_delta:
                self.best_loss = validation_loss
                self.counter = 0
            else:
                self.counter += 1

            if self.counter >= self.tolerance:
                self.early_stop = True

        # broadcast stop decision to all ranks on the correct device
        stop_tensor = torch.tensor(1 if self.early_stop else 0, device=f"cuda:{rank}", dtype=torch.int32)
        dist.broadcast(stop_tensor, src=0)
        self.early_stop = bool(stop_tensor.item())

        return self.early_stop
