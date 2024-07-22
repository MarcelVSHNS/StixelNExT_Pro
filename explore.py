import yaml
# 0.1 Load configfile
with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

import os
import torch
# import torchvision.ops
# from torchvision.models.convnext import ConvNeXt
from models import convnext_stixel
from torchinfo import summary
from losses import StixelObjectLoss
from dataloader import StixelData
from torch.utils.data import DataLoader
from PIL import Image
from stixel import StixelWorld
from stixel.utils import draw_stixels_on_image
from einops import rearrange
import time


def main():
    testing_data = StixelData(data_dir=config['data_path'], phase='testing', model=config['model'], return_name=True)
    testing_dataloader = DataLoader(testing_data, batch_size=config['batch_size'], pin_memory=True, drop_last=True)

    img_tensor, target_tensor, name = next(iter(testing_dataloader))
    """ Data exploration """
    stixel_world_batch = StixelData.revert(target_tensor,
                                           img_name=name,
                                           img_size=testing_data.img_size)
    stixel_world: StixelWorld = stixel_world_batch[0]
    image_path = os.path.join(config['data_path'], 'testing', 'FRONT', stixel_world.image_name + '.png')
    image: Image = Image.open(image_path)
    stixel_img = draw_stixels_on_image(image, stixel_world.stixel)
    stixel_img.show()
    """ Model exploration """
    # model = UNet(n_channels=3)
    # model = ConvNeXt(in_channels=3, c=60, depths_b=[3, 3, 27, 3])
    model, _ = convnext_stixel()

    input_shape = (4, 3, 1280, 1920)
    x = torch.randn(input_shape)
    output = model(x)
    summary(model, input_size=input_shape, device=torch.device('cpu'))
    print(f"Output shape: {output.shape}")

    """ Loss exploration """
    inputs = torch.rand(4, 4, 12, 240)
    loss_fn = StixelObjectLoss()
    l1, _ = loss_fn(output, output)
    print(f"Ident: {l1}")
    target = torch.rand(4, 4, 12, 240)
    l2, _ = loss_fn(output, target)
    print(f"Diff: {l2}")


if __name__ == '__main__':
    main()
