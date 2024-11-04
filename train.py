import yaml
# 0.1 Load configfile
with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

import os
import torch
import wandb
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp
from torchinfo import summary
from typing import Dict
from datetime import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from engine import train_one_epoch, evaluate, EarlyStopping
from dataloader import StixelData

if config['mode'] == "segmentation":
    import models.ConvNeXt_pretrained as model_file
    from models import get_model as model_fn
    from losses import StixelVoxelLoss as StixelLoss
elif config['mode'] == "classification":
    import models.ConvNeXt_pretrained as model_file
    from models import convnext_stixel as model_fn
    from losses import StixelObjectLoss as StixelLoss
else:
    raise ValueError("Invalid mode specified in config file!")

# starting time for all instances
overall_start_time = datetime.now()


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # 'gloo' for CPUs, 'nccl' for GPUs
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.barrier()
    dist.destroy_process_group()


def save_checkpoint(model, optimizer, epoch, loss, filename):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filename)


def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    continue_epoch = checkpoint['epoch'] + 1
    last_loss = checkpoint['loss']
    return continue_epoch, last_loss


def train(rank, world_size):
    # torch.cuda.init()
    setup(rank, world_size)

    """ 1.Load data """
    # Training data
    tmpdir = os.getenv("TMPDIR", "")
    data_dir = os.path.join(tmpdir, config['data_path'])
    training_data = StixelData(data_dir=data_dir, phase='training', mode=config['mode'], target_trans_blur=config['blur'])
    training_sampler = DistributedSampler(training_data, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(training_data, batch_size=config['batch_size'], pin_memory=True, drop_last=True,
                                  sampler=training_sampler)
    # Validation data
    validation_data = StixelData(data_dir=config['data_path'], phase='validation', mode=config['mode'])
    # validation_sampler = DistributedSampler(validation_data, num_replicas=world_size, rank=rank)
    val_dataloader = DataLoader(validation_data, batch_size=config['batch_size'], pin_memory=True, drop_last=True)

    """ 2.Define Model & Loss """
    model, model_cfg = model_fn()
    model = model.to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    # Optimizer definition
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    # Loss initialization
    loss_weights: Dict[str, float] = {}
    if config['mode'] == "segmentation":
        loss_weights = config['loss_w_seg']
    elif config['mode'] == "classification":
        loss_weights = config['loss_w_cls']
    loss_fn = StixelLoss(loss_weights)

    # Load checkpoint
    start_epoch = 0
    if config['load_checkpoint'] is not None:
        if rank == 0 and os.path.isfile(config['load_checkpoint']):
            start_epoch, loss = load_checkpoint(model, optimizer, config['load_checkpoint'])
            print(f"Checkpoint {os.path.basename(config['load_checkpoint'])} loaded. Training stopped on epoch {start_epoch - 1} with loss {loss}. Training will be continued ...")

    # Initialize Logger
    if config['logging'] and rank == 0:
        wandb_logger = wandb.init(project="StixelNExT-Pro",
                                  config={
                                      "learning_rate": config['learning_rate'],
                                      "loss_name": type(loss_fn).__name__,
                                      "loss": loss_fn.params(),
                                      "mode": config['mode'],
                                      "model": model_cfg,
                                      "dataset": training_data.name,
                                      "epochs": config['epochs'],
                                      "rank": rank,
                                      "batch_size": config['batch_size'],
                                      "checkpoint": config['load_checkpoint'],
                                      "early_stop": config['early_stop'],
                                      "num_gpu": world_size,
                                      "blur": config['blur']
                                  },
                                  job_type="training",
                                  tags=["training"]
                                  )
        artifact = wandb.Artifact(f"{model_cfg['name']}_weights_art", type='model', description="Automatic checkpoint pick by train/ eval loss.")
        wandb_logger.watch(model)
    else:
        wandb_logger = None

    """ 3.Training """
    # Inspect model
    summary(model, (config['batch_size'], 3, 1280, 1920))

    # Training
    early_stopping = EarlyStopping(tolerance=config['early_stop']['tol'],
                                   min_delta=config['early_stop']['min_delta'])
    best_loss = float('inf')
    for epoch in range(start_epoch, config['epochs']):
        print(f"\n   Epoch {epoch}\n----------------------------------------------------------------")
        training_sampler.set_epoch(epoch)
        train_loss = train_one_epoch(train_dataloader, model, loss_fn, optimizer,
                                      device=rank, writer=wandb_logger)
        eval_loss = evaluate(val_dataloader, model, loss_fn,
                              device=rank, writer=wandb_logger)
        # Save model
        if config['logging'] and rank == 0:
            saved_models_path = os.path.join('saved_models', wandb_logger.name)
            os.makedirs(saved_models_path, exist_ok=True)
            weights_name = f"StixelNExT-Pro_{wandb_logger.name}_{epoch}.pth"
            weights_path = os.path.join(saved_models_path, weights_name)
            save_checkpoint(model, optimizer, epoch, eval_loss, weights_path)
            print("Saved PyTorch Model State to " + weights_path)
            if eval_loss < best_loss:
                best_loss = eval_loss
                best_weights_path = weights_path
                artifact.metadata = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'eval_loss': eval_loss
                }
        step_time = datetime.now() - overall_start_time
        print("Time elapsed: {}".format(step_time))

        # early stopping
        early_stopping.check_stop(eval_loss, rank)
        if early_stopping.early_stop:
            print("Early stopping at epoch:", epoch)
            break

    overall_time = datetime.now() - overall_start_time
    print(f"Finished training in {str(overall_time).split('.')[0]}")
    if config['logging'] and rank == 0:
        artifact.metadata.update(model_cfg)
        artifact.metadata.update({"mode": config['mode']})
        artifact.add_file(best_weights_path)
        artifact.add_file(model_file.__file__)
        wandb_logger.log_artifact(artifact)
        wandb.finish()
    cleanup()


def main():
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} cuda devices.")
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()
