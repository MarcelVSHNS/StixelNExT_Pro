import yaml

# 0.1 Load configfile
with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

import os
import torch
# import torchvision.ops
# from torchvision.models.convnext import ConvNeXt
from models import convnext_stixel, unet_stixel, convnext_stixel_segmentation
from torchinfo import summary
from losses import StixelObjectLoss, StixelVoxelLoss
from dataloader import StixelData
from torch.utils.data import DataLoader
from PIL import Image
from stixel import StixelWorld
from stixel.utils import draw_stixels_on_image
from einops import rearrange
import time


def main():
    """ data load
    testing_data = StixelData(data_dir=config['data_path'], phase='validation', return_name=True, mode=config['mode'])
    testing_dataloader = DataLoader(testing_data, batch_size=config['batch_size'], pin_memory=True, drop_last=True,
                                    shuffle=True)

    img_tensor, target_tensor, name = next(iter(testing_dataloader))"""

    """ Data exploration 
    if config['mode'] == 'classification':
        stixel_world_batch = StixelData.revert_class(target_tensor, testing_data.depth_anchors,
                                                     img_name=name,
                                                     img_size=testing_data.img_size)
    elif config['mode'] == 'segmentation':
        stixel_world_batch = StixelData.revert_segm(target_tensor, testing_data.depth_anchors, img_name=name)
    else:
        raise ValueError('Invalid mode!')
    stixel_world: StixelWorld = stixel_world_batch[0]
    image_path = os.path.join(config['data_path'], 'validation', 'FRONT', stixel_world.image_name + '.png')
    image: Image = Image.open(image_path)
    stixel_img = draw_stixels_on_image(image, stixel_world.stixel)
    stixel_img.show()
    # original
    stixel_path = os.path.join(config['data_path'], 'validation', 'Stixel', stixel_world.image_name + '.csv')
    stixel_world_og: StixelWorld = StixelWorld.read(stixel_path)
    stixel_img = draw_stixels_on_image(image, stixel_world_og.stixel)
    stixel_img.show()"""

    """ Model exploration """
    # model, _ = unet()
    # model = ConvNeXt(in_channels=3, c=60, depths_b=[3, 3, 27, 3])
    model, _ = convnext_stixel_segmentation()

    input_shape = (1, 3, 1280, 1920)
    # x = torch.randn(input_shape)
    # output = model(x)
    summary(model, input_size=input_shape, device=torch.device('cpu'))
    # print(f"Output shape: {output.shape}")

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
