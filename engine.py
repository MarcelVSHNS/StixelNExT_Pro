import torch
import torch.distributed as dist


# Training Function
def train_one_epoch(dataloader,
                    model,
                    loss_fn,
                    optimizer,
                    device,
                    scheduler=None,
                    writer=None) -> float:
    num_batches = len(dataloader.dataset)
    train_loss = 0.0
    model.train()
    # for every batch_sized chunk of data ...
    for batch_idx, (samples, targets, _) in enumerate(dataloader):
        # copy data to the computing device (normally the GPU)
        samples = samples.to(device)
        targets = targets.to(device)
        # Compute prediction a prediction
        outputs = model(samples)
        # Compute the error (loss) of that prediction [loss_fn(prediction, target)]
        loss = loss_fn(outputs, targets)
        train_loss += loss.item()
        # Backpropagation strategy/ optimization "zero_grad()"
        optimizer.zero_grad()
        # Apply the prediction loss (backpropagation)
        loss.backward()
        # write the weights to the NN
        optimizer.step()
        # scheduler update per batch
        if scheduler is not None:
            scheduler.step()
            
        if batch_idx % 100 == 0:
            loss, current = loss.item(), batch_idx * len(samples)
            print(f"loss: {loss:>7f}  [{current:>5d}/{num_batches:>5d}], \t")
    train_loss /= num_batches
    if writer:
        # Log the average loss for the epoch
        writer.log({"Train loss": train_loss})
    return train_loss


# Validation Function
def evaluate(dataloader, model, loss_fn, device, writer=None):
    num_batches = len(dataloader)
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for (samples, targets, _) in dataloader:
            samples = samples.to(device)
            targets = targets.to(device)
            outputs = model(samples)
            loss = loss_fn(outputs, targets)
            eval_loss += loss
    eval_loss /= num_batches
    print(f"Test Error: \n Avg loss: {eval_loss:>8f} \n")
    if writer:
        # Log the loss by adding scalars
        writer.log({"Eval loss": eval_loss})
    return eval_loss


class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0.0):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.validation_loss_minus_one = 10000
        self.counter = 0
        self.early_stop = False

    def check_stop(self, validation_loss, rank):
        if rank == 0:
            if (self.validation_loss_minus_one - validation_loss) > self.min_delta:
                self.counter = 0
            else:
                self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True
            self.validation_loss_minus_one = validation_loss

        early_stop_tensor = torch.tensor(self.early_stop, device='cuda')
        dist.broadcast(early_stop_tensor, src=0)

        self.early_stop = early_stop_tensor.item()

        return self.early_stop
