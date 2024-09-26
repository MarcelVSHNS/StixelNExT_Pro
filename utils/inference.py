"""
TODO: write function to find sweetspot prob_thres to get stixel
"""
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
    from models import get_model as model_fn
    from dataloader import revert_segm as revert_fn
elif config['mode'] == "classification":
    from models import convnext_stixel as model_fn
    from dataloader import revert_class as revert_fn
else:
    raise ValueError("Invalid mode specified in config file!")


def main():
    p_threshold = 0.56
    save_img: bool = False
    show_3d: bool = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    testing_data = StixelData(data_dir=config['data_path'], phase='validation', mode=config['mode'])
    testing_dataloader = DataLoader(testing_data, batch_size=1, pin_memory=True, drop_last=True,
                                    shuffle=True)
    print(f"Found {len(testing_data)} records.")
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
    with torch.no_grad():
        output = model(img_tensor)
    # extract Stixel
    output = output.cpu().detach()

    matrix = output.numpy()
    # Mittelwert
    mean = np.mean(matrix)
    # Median
    median = np.median(matrix)
    # Standardabweichung
    std_dev = np.std(matrix)
    # Minimum und Maximum
    min_value = np.min(matrix)
    max_value = np.max(matrix)
    print(f"Mittelwert: {mean}")
    print(f"Median: {median}")
    print(f"Standardabweichung: {std_dev}")
    print(f"Minimum: {min_value}")
    print(f"Maximum: {max_value}")

    num_stx_dict = {}
    for p in np.arange(0.5, 1.0, 0.02):
        stixel_world_batch = revert_fn(prediction=output,
                                       anchors=testing_data.depth_anchors,
                                       stxl_wrld_paths=stxl_wrld_paths,
                                       prob=p)
        stixel_world: stx.StixelWorld = stixel_world_batch[0]
        num_stx_dict[p] = len(stixel_world.stixel)
        print(f"p: {p} = {num_stx_dict[p]}")

    num_stx_dict = dict(sorted(num_stx_dict.items(), reverse=True))
    # Extrahiere x- und y-Werte
    x_values = list(num_stx_dict.keys())
    y_values = list(num_stx_dict.values())
    # Erstelle den Plot
    plt.plot(x_values, y_values, marker='o')  # Verwende 'o' als Marker für Datenpunkte
    plt.axvline(x=mean, color='green', linestyle='--', label=f'Mittelwert: {mean:.2f}')
    plt.axvline(x=median, color='purple', linestyle=':', label=f'Median: {median:.2f}')
    slopes = []
    x_slopes = []
    highlight_x = None
    highlight_y = None
    highlight2_x = None
    highlight2_y = None
    for i in range(1, len(x_values)):
        x1, y1 = x_values[i - 1], y_values[i - 1]
        x2, y2 = x_values[i], y_values[i]
        # Steigung berechnen
        slope = (y2 - y1) / (x2 - x1)
        slopes.append(slope)
        x_slopes.append(x2)
        print(f"p: {x2} = {slope}")
        if abs(slope) >= 1000 and highlight2_x is None:
            highlight2_x = x2
            highlight2_y = slope
        if abs(slope) >= 10000 and highlight_x is None:
            highlight_x = x2
            highlight_y = slope
    borders_x = [highlight2_x, highlight_x]
    print(f"Found area: {borders_x}")
    borders_y = [highlight2_y, highlight_y]
    plt.plot(x_slopes, slopes, marker='x')  # Verwende 'o' als Marker für Datenpunkte
    if highlight_x is not None and highlight_y is not None:
        plt.scatter(borders_x, borders_y, color='red', s=100, zorder=5,
                    label=f'area')
    plt.title('2D-Graph aus Dictionary')
    plt.xlabel('X-Werte')
    plt.ylabel('Y-Werte')
    # Zeige den Plot an
    plt.show()


    # test = output[0, 0:3, :, 110].numpy()
    # test_targ = target_tensor[0, 0:3, :, 110].numpy()
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
