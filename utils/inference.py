import yaml
# 0.1 Load configfile
with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

import os.path
from stixel import StixelWorld
from dataloader import StixelData
from stixel.utils import draw_stixels_on_image
from typing import List, Optional
import torch
from torch.utils.data import DataLoader
from PIL import Image
from collections import OrderedDict
import matplotlib.pyplot as plt

if config['mode'] == "segmentation":
    from models import unet_stixel as model_fn
elif config['mode'] == "classification":
    from models import convnext_stixel as model_fn
else:
    raise ValueError("Invalid mode specified in config file!")


def main():
    p_threshold = 0.30
    save_img: bool = False
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    testing_data = StixelData(data_dir="/media/marcel/Data1/Datasets/waymo-od", phase='training', return_name=True, mode=config['mode'])
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
    img_tensor, target_tensor, name = next(iter(testing_dataloader))
    img_tensor = img_tensor.to(device)
    # inference
    output = model(img_tensor)
    # extract Stixel
    output = output.cpu().detach()
    test = output[0, 0:3, :, 110].numpy()
    test_targ = target_tensor[0, 0:3, :, 110].numpy()
    stixel_world_batch = StixelData.revert_class(output, testing_data.depth_anchors,
                                                 img_name=name,
                                                 img_size=testing_data.img_size,
                                                 prob=p_threshold)
    stixel_world: StixelWorld = stixel_world_batch[0]
    image = Image.open(os.path.join("/media/marcel/Data1/Datasets/waymo-od", "training", "FRONT", f"{name[0]}.png"))

    stixel_img = draw_stixels_on_image(image, stixel_world.stixel)
    # stixel_img.show(title="prediction")

    # Ground Truth
    stixel_world_batch_targ = StixelData.revert_class(target_tensor, testing_data.depth_anchors,
                                                      img_name=name,
                                                      img_size=testing_data.img_size,
                                                      prob=p_threshold)
    stixel_world_targ: StixelWorld = stixel_world_batch_targ[0]
    stixel_img_targ = draw_stixels_on_image(image, stixel_world_targ.stixel)
    # stixel_img_targ.show(title="ground_truth")

    fig, axes = plt.subplots(2, 1, figsize=(10, 15), gridspec_kw={"hspace": 0, "wspace": 0})
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    for ax, bild, titel in zip(axes, [stixel_img, stixel_img_targ], ["Prediction", "Ground Truth"]):
        ax.imshow(bild)
        ax.set_title(titel)
        ax.axis("off")
        ax.grid(False)
    plt.show()

    if save_img:
        os.makedirs("results", exist_ok=True)
        stixel_img.save(os.path.join("results", f"Stixel_{stixel_world.image_name}.png"))
        stixel_world.save(os.path.join("results"))
        print(f"Image and Stixel: {stixel_world.image_name} saved.")


if __name__ == "__main__":
    main()
