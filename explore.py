import yaml
from triton.ops import attention

# 0.1 Load configfile
with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

import os
import torch
# import torchvision.ops
# from torchvision.models.convnext import ConvNeXt
from models import convnext_stixel
from torchinfo import summary
from models.ConvNeXt_pretrained import ColumnAttention
from losses import StixelObjectLoss, StixelVoxelLoss
from dataloader import StixelData, revert_class, revert_segm
from torch.utils.data import DataLoader
from PIL import Image
import stixel as stx
from einops import rearrange
import time
from datetime import datetime


def main():
    """ data load
    testing_data = StixelData(data_dir=config['data_path'], phase='validation', mode=config['mode'],
                              target_trans_blur=True)
    testing_dataloader = DataLoader(testing_data, batch_size=config['batch_size'], pin_memory=True, drop_last=True,
                                    shuffle=True)
    start = datetime.now()
    img_tensors, target_tensors, stxl_wrld_paths = next(iter(testing_dataloader))
    print(f"Data loaded in {datetime.now() - start}")"""
    """ Data exploration 
    if config['mode'] == 'classification':
        stixel_world_batch = revert_class(target_tensors,
                                                     anchors=testing_data.depth_anchors,
                                                     stxl_wrld_paths=stxl_wrld_paths)
    elif config['mode'] == 'segmentation':
        stixel_world_batch = revert_segm(target_tensors,
                                                    anchors=testing_data.depth_anchors,
                                                    stxl_wrld_paths=stxl_wrld_paths)
    else:
        raise ValueError('Invalid mode!')
    stixel_world: stx.StixelWorld = stixel_world_batch[0]
    stixel_img = stx.draw_stixels_on_image(stixel_world)
    stixel_img.show()
    # original
    stixel_world_og: stx.StixelWorld = stx.read(stxl_wrld_paths[0])
    stixel_img = stx.draw_stixels_on_image(stixel_world_og)
    stixel_img.show()"""

    """ Model exploration 
    device = torch.device('cuda')
    input_shape = (1, 3, 1280, 1920)
    model, _ = convnext_stixel()
    summary(model, input_size=input_shape)
    model = model.to(device)

    times = []
    for i in range(1):
        input_tensor = torch.randn(input_shape).to(device)
        start = datetime.now()
        with torch.no_grad():
            output = model(input_tensor)
        times.append(datetime.now() - start)
    # output = output.cpu().detach()
    # print(f"Output shape: {output.shape}")
    times_in_ms = [t.total_seconds() * 1000 for t in times]
    average_inference_time_ms = sum(times_in_ms) / len(times_in_ms)
    print(f"Inference time: {average_inference_time_ms:.2f} ms") """

    """ Loss exploration """
    inputs = torch.rand(1, 3, 64, 240)
    loss_fn = StixelObjectLoss(weights=config['loss_w_cls'])
    l1 = loss_fn(inputs, inputs)
    print(loss_fn.params())
    print(f"Ident: {l1}")
    target = torch.rand(1, 3, 64, 240)
    l2 = loss_fn(inputs, target)
    print(f"Diff: {l2}")


if __name__ == '__main__':
    main()
