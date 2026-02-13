import torch
import torch.distributed as dist


# Training Function
def train_one_epoch(dataloader, model, loss_fn, optimizer, device, epoch: int, writer=None,
                    log_every: int = 100) -> float:
    num_batches = len(dataloader)
    train_loss = 0.0
    model.train()

    for batch_idx, (samples, targets, _) in enumerate(dataloader):
        samples = samples.to(device)
        targets = targets.to(device)

        outputs = model(samples)
        loss = loss_fn(outputs, targets)

        train_loss += float(loss.item())

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

    return train_loss


# Validation Function
def evaluate(dataloader, model, loss_fn, device, epoch: int, writer=None) -> float:
    num_batches = len(dataloader)
    model.eval()
    eval_loss = 0.0

    with torch.no_grad():
        for (samples, targets, _) in dataloader:
            samples = samples.to(device)
            targets = targets.to(device)

            outputs = model(samples)
            loss = loss_fn(outputs, targets)
            eval_loss += float(loss.item())

    eval_loss /= max(num_batches, 1)

    print(f"Validation Error: Avg loss: {eval_loss:.6f}")

    if writer is not None:
        writer.add_scalar("Loss/val", eval_loss, epoch)

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
