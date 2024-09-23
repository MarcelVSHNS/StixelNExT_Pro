import yaml

# 0.1 Load configfile
with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

import os
import torch
# import torchvision.ops
# from torchvision.models.convnext import ConvNeXt
from models import convnext_stixel, get_model, get_model
from torchinfo import summary
from losses import StixelObjectLoss, StixelVoxelLoss
from dataloader import StixelData
from torch.utils.data import DataLoader
from PIL import Image
import stixel as stx
from einops import rearrange
import time
from datetime import datetime


def main():
    """ data load"""
    testing_data = StixelData(data_dir=config['data_path'], phase='validation', mode=config['mode'],
                              target_trans_blur=True)
    testing_dataloader = DataLoader(testing_data, batch_size=config['batch_size'], pin_memory=True, drop_last=True,
                                    shuffle=True)
    start = datetime.now()
    img_tensors, target_tensors, stxl_wrld_paths = next(iter(testing_dataloader))
    print(f"Data loaded in {datetime.now() - start}")
    """ Data exploration """
    """ 
    if config['mode'] == 'classification':
        stixel_world_batch = StixelData.revert_class(target_tensor,
                                                     anchors=testing_data.depth_anchors,
                                                     stxl_wrld_paths=stxl_wrld_paths)
    elif config['mode'] == 'segmentation':
        stixel_world_batch = StixelData.revert_segm(target_tensor,
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
    stixel_img.show()

    Model exploration 
    model, _ = unet_stixel()
    # model = ConvNeXt(in_channels=3, c=60, depths_b=[3, 3, 27, 3])
    # model, _ = convnext_stixel() 

    input_shape = (1, 3, 1280, 1920)
    # x = torch.randn(input_shape)
    # output = model(x)
    summary(model, input_size=input_shape, device=torch.device('cpu'))
    # print(f"Output shape: {output.shape}") """

    """ Loss exploration 
    inputs = torch.rand(1, 64, 160, 240)
    loss_fn = StixelVoxelLoss(weights=config['loss_w_seg'])
    l1 = loss_fn(inputs, inputs)
    print(loss_fn.params())
    print(f"Ident: {l1}")
    target = torch.rand(1, 64, 160, 240)
    l2 = loss_fn(inputs, target)
    print(f"Diff: {l2}")"""


if __name__ == '__main__':
    main()
