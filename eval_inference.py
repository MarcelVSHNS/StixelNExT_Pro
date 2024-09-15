import yaml
# 0.1 Load configfile
with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

import os
import torch
from collections import OrderedDict
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp
from torchinfo import summary
import numpy as np
from datetime import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from engine import train_one_epoch, evaluate, EarlyStopping
from dataloader import StixelData
import stixel as stx

if config['mode'] == "segmentation":
    from models import unet_stixel as model_fn
    from dataloader import revert_segm as revert_fn
elif config['mode'] == "classification":
    from models import convnext_stixel as model_fn
    from dataloader import revert_class as revert_fn
else:
    raise ValueError("Invalid mode specified in config file!")


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # 'gloo' for CPUs, 'nccl' für CPUs
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    continue_epoch = checkpoint['epoch'] + 1
    last_loss = checkpoint['loss']
    return continue_epoch, last_loss


# Inference Function
def inference(rank, world_size):
    torch.cuda.init()
    setup(rank, world_size)

    """ 1.Load data """
    # Validation data
    validation_data = StixelData(data_dir=config['data_path'], phase='validation', mode=config['mode'])
    validation_sampler = DistributedSampler(validation_data, num_replicas=world_size, rank=rank)
    val_dataloader = DataLoader(validation_data, batch_size=config['batch_size'], pin_memory=True, drop_last=True,
                                sampler=validation_sampler)

    """ 2.Define Model & Loss """
    model, model_cfg = model_fn()
    model = model.to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

    assert config['load_checkpoint'] is not None; "Define Checkpoint for inference!"
    chckpt_epoch = os.path.split('/')[-1].split('.')[0]
    if rank == 0 and os.path.isfile(config['load_checkpoint']):
        start_epoch, loss = load_checkpoint(model, optimizer, config['load_checkpoint'])
        print(
            f"Checkpoint {os.path.basename(config['load_checkpoint'])} loaded. Training from epoch {start_epoch - 1} with loss {loss}.")
    """
    if config['load_checkpoint']:
        checkpoint = torch.load(config['load_checkpoint'])
        new_state_dict = OrderedDict()
        model_state = checkpoint['model_state_dict']
        for k, v in model_state.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        print("Loaded checkpoint '{}'".format(config['load_checkpoint']))
    """
    # Inspect model
    summary(model, (config['batch_size'], 3, 1280, 1920))

    num_batches = len(val_dataloader)
    model.eval()
    with torch.no_grad():
        for batch_idx, (samples, _, stxl_names) in enumerate(val_dataloader):
            samples = samples.to(rank)
            start_time = datetime.now()
            outputs = model(samples)
            inference_time = datetime.now() - start_time
            outputs = outputs.cpu().detach()
            for prob in np.arange(config['prob_from'], config['prob_to'] + config['prob_in'], config['prob_in']):
                stixel_world_batch = revert_fn(prediction=outputs,
                                               anchors=validation_data.depth_anchors,
                                               stxl_wrld_paths=stxl_names,
                                               prob=prob)
                for stxl_wrld in stixel_world_batch:
                    stx.save(stxl_wrld, os.path.join(config['output_path'], f"{chckpt_epoch}_{prob}-probability"))
                print(f"Probability: {prob}. Batch {batch_idx} from {num_batches} exported in {inference_time}. Batch size={val_dataloader.batch_size}")


def main():
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} cuda devices.")
    mp.spawn(inference, args=(world_size,), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()
