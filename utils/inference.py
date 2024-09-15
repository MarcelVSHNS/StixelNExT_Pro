import yaml
# 0.1 Load configfile
with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

import os.path
import stixel as stx
from dataloader import StixelData
from typing import List, Optional
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import DataLoader
from PIL import Image
from collections import OrderedDict
import matplotlib.pyplot as plt

if config['mode'] == "segmentation":
    from models import convnext_stixel_segmentation as model_fn
    from dataloader import revert_segm as revert_fn
elif config['mode'] == "classification":
    from models import convnext_stixel as model_fn
    from dataloader import revert_class as revert_fn
else:
    raise ValueError("Invalid mode specified in config file!")


def main():
    p_threshold = 0.84
    save_img: bool = False
    show_3d: bool = False
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    testing_data = StixelData(data_dir="dataset/waymo-od_tiny_new", phase='validation', mode=config['mode'])
    testing_dataloader = DataLoader(testing_data, batch_size=1, pin_memory=True, drop_last=True,
                                    shuffle=True)
    model, _ = model_fn()
    model = model.to(device)
    # summary(model, input_size=(1, 3, 1280, 1920), device=torch.device('cpu'))
    if config['load_checkpoint']:
        checkpoint = torch.load(config['load_checkpoint'])
        new_state_dict = OrderedDict()
        model_state = checkpoint['model_state_dict']
        for k, v in model_state.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        print("Loaded checkpoint '{}'".format(config['load_checkpoint']))

    # random sample
    img_tensor, target_tensor, stxl_wrld_paths = next(iter(testing_dataloader))
    img_tensor = img_tensor.to(device)
    # inference
    output = model(img_tensor)
    # extract Stixel
    output = output.cpu().detach()
    test = output[0, 0:3, :, 110].numpy()
    test_targ = target_tensor[0, 0:3, :, 110].numpy()
    stixel_world_batch = revert_fn(prediction=output,
                                   anchors=testing_data.depth_anchors,
                                   stxl_wrld_paths=stxl_wrld_paths,
                                   prob=p_threshold)
    stixel_world_batch_targ = revert_fn(prediction=target_tensor,
                                        anchors=testing_data.depth_anchors,
                                        stxl_wrld_paths=stxl_wrld_paths,
                                        prob=p_threshold)


    stixel_world: stx.StixelWorld = stixel_world_batch[0]
    stixel_img = stx.draw_stixels_on_image(stixel_world)
    # stixel_img.show(title="prediction")

    # Ground Truth
    stixel_world_targ: stx.StixelWorld = stixel_world_batch_targ[0]
    stixel_img_targ = stx.draw_stixels_on_image(stixel_world_targ)
    # stixel_img_targ.show(title="ground_truth")

    fig, axes = plt.subplots(2, 1, figsize=(10, 15), gridspec_kw={"hspace": 0, "wspace": 0})
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    for ax, bild, titel in zip(axes, [stixel_img, stixel_img_targ], ["Prediction", "Ground Truth"]):
        ax.imshow(bild)
        ax.set_title(titel)
        ax.axis("off")
        ax.grid(False)
        fig.suptitle(f"@ P={p_threshold}", fontsize=16)
    plt.show()

    if save_img:
        os.makedirs("results", exist_ok=True)
        stixel_img.save(os.path.join("results", f"Stixel_{stixel_world.context.calibration.img_name}"))
        print(f"Image: {stixel_world.context.calibration.img_name} saved.")

    if show_3d:
        stx.draw_stixels_in_3d(stixel_world)


if __name__ == "__main__":
    main()
