import yaml

# 0.1 Load configfile
with open('config.yaml') as yamlfile:
    config = yaml.safe_load(yamlfile)
import os
import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp
from torchinfo import summary
from typing import Dict
from datetime import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from engine import train_one_epoch, evaluate, EarlyStopping
from dataloader import StixelData

import models.ConvNeXt_pretrained as model_file
from models import convnext_stixel as model_fn
from losses import StixelObjectLoss as StixelLoss


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # 'gloo' for CPUs, 'nccl' for GPUs
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def save_checkpoint(model, optimizer, epoch, loss, filename):
    m = _unwrap(model)
    if isinstance(loss, torch.Tensor):
        loss = loss.detach().cpu().item()
    torch.save({
        "epoch": int(epoch),
        "model_state_dict": m.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": float(loss),
    }, filename)


def load_checkpoint(model, optimizer, filename, device):
    ckpt = torch.load(filename, map_location=f"cuda:{device}")
    m = _unwrap(model)
    m.load_state_dict(ckpt["model_state_dict"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return int(ckpt["epoch"]) + 1, float(ckpt["loss"])


def init_tensorboard(cfg, rank):
    if rank != 0:
        return None

    log_dir = cfg.get("tensorboard", {}).get("log_dir", "runs")
    run_name = cfg.get("tensorboard", {}).get("run_name", None)

    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    return SummaryWriter(log_dir=os.path.join(log_dir, run_name))


def train(rank, world_size):
    # starting time for all instances
    overall_start_time = datetime.now()
    # torch.cuda.init()
    setup(rank, world_size)

    """ 1.Load data """
    # Training data
    tmpdir = os.getenv("TMPDIR", "")
    data_dir = os.path.join(tmpdir, config['data_path'])
    training_data = StixelData(data_dir=data_dir, phase='training', mode=config['mode'],
                               target_trans_blur=config['blur'], depth_anchors=(4, 66, config['n_cand']))
    training_sampler = DistributedSampler(training_data, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(training_data, batch_size=config['batch_size'], pin_memory=True, drop_last=True,
                                  sampler=training_sampler)
    # Validation data
    validation_data = StixelData(data_dir=data_dir, phase='validation', mode=config['mode'],
                                 depth_anchors=(4, 66, config['n_cand']),
                                 return_depth_maps=True)
    # validation_sampler = DistributedSampler(validation_data, num_replicas=world_size, rank=rank)
    val_dataloader = DataLoader(validation_data, batch_size=config['batch_size'], pin_memory=True, drop_last=True)

    """ 2.Define Model & Loss """
    model, model_cfg = model_fn(n_candidates=config['n_cand'])
    model = model.to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    # Optimizer definition
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    # Loss initialization
    loss_weights: Dict[str, float] = {}
    if config['mode'] == "segmentation":
        loss_weights = config['loss_w_seg']
    elif config['mode'] == "classification":
        loss_weights = config['loss_w_cls']
    loss_cfg = config.get("loss_cfg_cls", {}) if config['mode'] == "classification" else {}
    loss_fn = StixelLoss(loss_weights, classify_cfg=loss_cfg)

    # logger
    tb_writer = None
    if config['tensorboard']['enabled']:
        tb_writer = init_tensorboard(config, rank)

    # Load checkpoint
    start_epoch = 0
    if config['load_checkpoint'] is not None and os.path.isfile(config['load_checkpoint']):
        start_epoch, loss = load_checkpoint(model, optimizer, config['load_checkpoint'], device=rank)
        if rank == 0:
            print(f"Loaded checkpoint {os.path.basename(config['load_checkpoint'])} (resume @ epoch {start_epoch}).")
    dist.barrier()

    """ 3.Training """
    # Inspect model
    if rank == 0:
        summary(model, (config['batch_size'], 3, 1280, 1920))

    # Training
    early_stopping = EarlyStopping(tolerance=config['early_stop']['tol'],
                                   min_delta=config['early_stop']['min_delta'])

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_models_path = os.path.join(config.get("checkpoint", {}).get("dir", "saved_models"), run_name)
    if rank == 0:
        os.makedirs(saved_models_path, exist_ok=True)
    dist.barrier()

    best_loss = float("inf")
    best_weights_path = os.path.join(saved_models_path, "best.pth")
    last_weights_path = os.path.join(saved_models_path, "last.pth")

    ckpt_cfg = config.get("checkpoint", {})
    save_best = ckpt_cfg.get("save_best", True)
    save_last = ckpt_cfg.get("save_last", True)
    save_every = int(ckpt_cfg.get("save_every", 1))

    log_every = int(config.get("tensorboard", {}).get("log_every", 100))
    for epoch in range(start_epoch, config["epochs"]):
        training_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(train_dataloader, model, loss_fn, optimizer, device=rank, epoch=epoch,
                                     writer=tb_writer, log_every=log_every)
        if rank == 0:
            eval_loss = evaluate(val_dataloader, model, loss_fn, device=rank, epoch=epoch, writer=tb_writer)
        else:
            eval_loss = torch.tensor(0.0, device=rank)
        dist.barrier()

        if rank == 0:
            # always keep "last" if enabled
            if save_last:
                save_checkpoint(model, optimizer, epoch, eval_loss, last_weights_path)

            # keep "best" if enabled
            if save_best and eval_loss < best_loss:
                best_loss = eval_loss
                save_checkpoint(model, optimizer, epoch, eval_loss, best_weights_path)
                print(f"New best model @ epoch {epoch}: val_loss={eval_loss:.6f}")

            # optionally, periodic snapshots
            if save_every > 0 and (epoch + 1) % save_every == 0:
                snap_path = os.path.join(saved_models_path, f"epoch_{epoch:04d}.pth")
                save_checkpoint(model, optimizer, epoch, eval_loss, snap_path)

        # early stopping
        if early_stopping.check_stop(eval_loss, rank):
            if rank == 0:
                print("Early stopping at epoch:", epoch)
            break

    if rank == 0:
        overall_time = datetime.now() - overall_start_time
        print(f"Finished training in {str(overall_time).split('.')[0]}")

    if tb_writer is not None:
        tb_writer.close()
    cleanup()


def main():
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} cuda devices.")
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()
