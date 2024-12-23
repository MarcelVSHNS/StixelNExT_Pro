# import multiprocessing
import os
import os.path
from datetime import datetime
import random
from collections import OrderedDict
import io

import stixel as stx
import yaml
import torch
from torch.utils.data import DataLoader
from dataloader import StixelData
from models import convnext_stixel
from PIL import Image

with open('config.yaml') as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)

if config['mode'] == "segmentation":
    from dataloader import revert_segm as revert_fn
elif config['mode'] == "classification":
    from dataloader import revert_class as revert_fn

overall_start_time = datetime.now()
os.environ["WANDB_REPORT_API_ENABLE_V2"] = "True"
os.environ["WANDB_REPORT_API_DISABLE_MESSAGE"] = "True"


def main():
    testing_data = StixelData(data_dir=config['data_path'], phase='validation', mode=config['mode'],
                              depth_anchors=(4, 66, config['n_cand']), target_trans_blur=config['blur'],
                              transform=True)
    testing_dataloader = DataLoader(testing_data, batch_size=config['batch_size'], pin_memory=True, drop_last=True,
                                    shuffle=True)
    # model
    if config["device"] == "gpu":
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        dev = torch.device('cpu')
    model, _ = convnext_stixel(device=dev, n_cand=config["n_cand"])
    model.to(dev)
    chckpt_filename = config["load_checkpoint"]
    # stxl_model.info()
    checkpoint = torch.load(chckpt_filename, weights_only=True)
    new_state_dict = OrderedDict()
    model_state = checkpoint['model_state_dict']
    for k, v in model_state.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    # print("Loaded checkpoint '{}'".format(self.checkpoint_name))
    model.eval()

    # local results directory
    result_dir = os.path.join('sample_results', chckpt_filename)
    os.makedirs(result_dir, exist_ok=True)
    idx = random.randint(0, len(testing_data) - 1)
    print(f"random Idx: {idx}")
    sample, _, sample_path = testing_data[idx]
    stxl_original = stx.read(sample_path)  # 31 for default sample
    probability = config["explore_thres"]

    # Inference a Stixel World
    sample = sample.to(dev)
    start_time = datetime.now()
    sample_batch = sample.unsqueeze(0)
    stxl_infer = model(sample_batch)
    print(f"inference time: {datetime.now() - start_time}")
    stxl_infer = stxl_infer.detach().cpu()
    start_time = datetime.now()
    stxl_wrld = revert_fn(prediction=stxl_infer, anchors=testing_data.depth_anchors,
                          stxl_wrld_paths=[sample_path],
                          prob=probability)
    print(f"reverting time: {datetime.now() - start_time}")
    stxl_wrld_clustered = stx.attach_dbscan_clustering(stxl_wrld[0], min_samples=1)
    clustered_img = stx.draw_stixels_on_image(stxl_wrld_clustered, instances=True)
    stxl_img = stx.draw_stixels_on_image(stxl_wrld[0])
    input_img = Image.open(io.BytesIO(stxl_wrld[0].image))
    stxl_gt_img = stx.draw_stixels_on_image(stxl_original)

    width, height = input_img.size
    canvas_width = width
    canvas_height = height * 3
    canvas = Image.new("RGB", (canvas_width, canvas_height), (0, 0, 0))
    canvas.paste(input_img, (0, 0))
    canvas.paste(stxl_img, (0, height))
    canvas.paste(clustered_img, (0, height * 2))
    canvas.show()


if __name__ == "__main__":
    main()
