import yaml
# 0.1 Load configfile
with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

import torch
import wandb
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp
from torchinfo import summary
from datetime import datetime
import os
import shutil
from losses import StixelObjectLoss
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from engine import train_one_epoch, evaluate, EarlyStopping
from dataloader import StixelData

if config['model'] == "unet":
    from models import UNet as Model
elif config['model'] == "convnext":
    from models import ConvNeXt as Model
elif config['model'] == "convnext_pretrained":
    from models import convnext_stixel as Model
else:
    raise ValueError("Invalid model specified in config file!")

# starting time for all instances
overall_start_time = datetime.now()


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # 'gloo' for CPUs, 'nccl' für CPUs
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def train(rank, world_size):
    torch.cuda.init() 
    setup(rank, world_size)

    """ 1.Load data """
    # Training data, TODO: impact of shuffling or not?
    training_data = StixelData(data_dir=config['data_path'], phase='training', model=config['model'])
    training_sampler = DistributedSampler(training_data, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(training_data, batch_size=config['batch_size'], pin_memory=True, drop_last=True,
                                  sampler=training_sampler)
    # Validation data
    validation_data = StixelData(data_dir=config['data_path'], phase='validation', model=config['model'])
    validation_sampler = DistributedSampler(training_data, num_replicas=world_size, rank=rank)
    val_dataloader = DataLoader(validation_data, batch_size=config['batch_size'], pin_memory=True, drop_last=True,
                                sampler=validation_sampler)

    """ 2.Define Model """
    model, model_cfg = Model()
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    # Load Weights
    if config['load_checkpoint'] is not None:
        weights_file = config['load_checkpoint']
        checkpoint = os.path.splitext(weights_file)[0]  # checkpoint without ending
        run = checkpoint.split('_')[1]
        model.load_state_dict(
            torch.load(f=os.path.join("saved_models", run, weights_file),
                       map_location=torch.device(rank)))
        print(f'Weights loaded from: {weights_file}')

    """ 3.Loss function & Training functions """
    # Loss function
    loss_weights = {
        'prob': config['P'],
        'obj_length': config['h'],
        'bottom': config['vB'],
        'depth': config['d'],
        'depth_length_ratio': config['h_d_ratio']
    }
    loss_fn = StixelObjectLoss(loss_weights)

    # Optimizer definition
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

    # Initialize Logger
    if config['logging'] and rank == 0:
        wandb_logger = wandb.init(project="StixelNExT Pro",
                                  config={
                                      "learning_rate": config['learning_rate'],
                                      "loss": type(loss_fn).__name__,
                                      "loss_params": "-",
                                      "model": type(model).__name__,
                                      "model_params": model_cfg,
                                      "dataset": training_data.name,
                                      "epochs": config['epochs'],
                                      "rank": rank
                                  },
                                  tags=["training"]
                                  )
        wandb_logger.watch(model)
    else:
        wandb_logger = None

    """ 4.Training """
    # Inspect model
    summary(model, (config['batch_size'], 3, 1280, 1920))

    # Training
    checkpoints = []
    early_stopping = EarlyStopping(tolerance=config['early_stop']['tol'],
                                   min_delta=config['early_stop']['min_delta'])
    for epoch in range(config['epochs']):
        print(f"\n   Epoch {epoch + 1}\n----------------------------------------------------------------")
        train_error = train_one_epoch(train_dataloader, model, loss_fn, optimizer,
                                      device=rank, writer=wandb_logger)
        test_error = evaluate(val_dataloader, model, loss_fn,
                              device=rank, writer=wandb_logger)
        # Save model
        if config['logging'] and rank == 0:
            saved_models_path = os.path.join('saved_models', wandb_logger.name)
            os.makedirs(saved_models_path, exist_ok=True)
            weights_name = f"StixelNExT_Pro{wandb_logger.name}_epoch-{epoch}_test-error-{test_error}.pth"
            torch.save(model.state_dict(), os.path.join(saved_models_path, weights_name))
            checkpoints.append({'checkpoint': weights_name, 'test-error': test_error})
            print("Saved PyTorch Model State to " + os.path.join(saved_models_path, weights_name))
        step_time = datetime.now() - overall_start_time
        print("Time elapsed: {}".format(step_time))

        # early stopping
        early_stopping.check_stop(test_error)
        if early_stopping.early_stop:
            print("Early stopping at epoch:", epoch)
            break

    overall_time = datetime.now() - overall_start_time
    print(f"Finished training in {str(overall_time).split('.')[0]}")

    if config['logging'] and rank == 0:
        best_checkpoint = min(checkpoints, key=lambda x: x['test-error'])
        source_path = os.path.join(saved_models_path, best_checkpoint['checkpoint'])
        destination_path = os.path.join("best_models", best_checkpoint['checkpoint'])
        shutil.copy(source_path, destination_path)
        wandb.finish()

    cleanup()


def main():
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} cuda devices.")
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()
