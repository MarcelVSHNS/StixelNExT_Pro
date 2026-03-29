import os
import glob
from datetime import datetime
from collections import OrderedDict

import yaml
import torch
from PIL import Image
import json
import numpy as np
import cv2

import stixel as stx
from models import convnext_stixel
from dataloader.StixelWorld import create_depth_bins

with open("config.yaml") as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)

from dataloader import revert_class as revert_fn

overall_start_time = datetime.now()
os.environ["WANDB_REPORT_API_ENABLE_V2"] = "True"
os.environ["WANDB_REPORT_API_DISABLE_MESSAGE"] = "True"


def get_device():
    if config["device"] == "gpu":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def load_model(device):
    model, _ = convnext_stixel(device=device, n_cand=config["n_cand"])
    model.to(device)

    checkpoint_path = config["load_checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    new_state_dict = OrderedDict()
    model_state = checkpoint["model_state_dict"]
    for k, v in model_state.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    return model


def attach_image_to_stixel_world(stxl_wrld, image_path):
    """
    Encode the image as JPEG and attach it to the stixel world protobuf.
    """
    img = cv2.imread(image_path)  # already BGR
    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    success, img_encoded = cv2.imencode(".png", img)

    if not success:
        raise RuntimeError("PNG encoding failed")

    stxl_wrld.image = img_encoded.tobytes()


def load_camera_matrix_from_json(calib_path):
    with open(calib_path, "r") as f:
        calib = json.load(f)

    fx = float(calib["fx"])
    fy = float(calib["fy"])
    cx = float(calib["cx"])
    cy = float(calib["cy"])

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    return K


def preprocess_png(image_path):
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = np.transpose(img_np, (2, 0, 1))  # HWC -> CHW
    tensor = torch.from_numpy(img_np).float()
    return tensor, img


def attach_camera_matrix(stxl_wrld, K):
    del stxl_wrld.context.calibration.K[:]
    stxl_wrld.context.calibration.K.extend(K.flatten().tolist())


def find_png_files(input_dir, recursive=False):
    if recursive:
        return sorted(glob.glob(os.path.join(input_dir, "**", "*.png"), recursive=True))
    return sorted(glob.glob(os.path.join(input_dir, "*.png")))


def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)


def infer_single_image(model, device, image_path, depth_anchors, probability, output_dir):
    sample, pil_img = preprocess_png(image_path)
    sample = sample.to(device)
    sample_batch = sample.unsqueeze(0)

    start_time = datetime.now()
    with torch.no_grad():
        stxl_infer = model(sample_batch)
    print(f"[INFO] {os.path.basename(image_path)} inference time: {datetime.now() - start_time}")

    stxl_infer = stxl_infer.detach().cpu()

    # IMPORTANT:
    # revert_fn previously received stxl_wrld_paths=[sample_path] from StixelData.
    # If revert_fn expects paths to .stx metadata/world files, passing PNG paths may not work.
    # If it only uses the path as an identifier, this is fine.
    stxl_wrld = revert_fn(
        prediction=stxl_infer,
        anchors=depth_anchors,
        img_height=1280,
        prob=probability,
    )

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.stx1")

    stx.save(
        stxl_wrld=stxl_wrld[0],
        filepath=output_path,
        export_image=True,
        sys_out=True,
    )

    print(f"[INFO] Saved: {output_path}")


def get_calib_path_for_image(image_path, calib_dir=None, calib_suffix=".json"):
    """
    Resolve calibration path for an image.

    Default assumption:
    image: /path/to/frame_000123.png
    calib: /some/calib_dir/frame_000123.json

    If calib_dir is None, calibration JSON is expected next to the image.
    """
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    if calib_dir is None:
        return os.path.join(os.path.dirname(image_path), base_name + calib_suffix)
    return os.path.join(calib_dir, base_name + calib_suffix)


def main():
    png_dir = config["png_input_dir"]
    png_files = find_png_files(png_dir, recursive=config.get("recursive_png_search", False))

    if not png_files:
        raise FileNotFoundError(f"No PNG files found in: {png_dir}")

    if config["device"] == "gpu":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device("cpu")

    model, _ = convnext_stixel(device=dev, n_cand=config["n_cand"])
    model.to(dev)

    chckpt_filename = config["load_checkpoint"]
    checkpoint = torch.load(chckpt_filename, map_location=dev, weights_only=True)

    new_state_dict = OrderedDict()
    model_state = checkpoint["model_state_dict"]
    for k, v in model_state.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.eval()

    output_dir = config.get(
        "output_dir",
        "/home/marcel/workspace/datasets/waymo/eval/stixelnextpp/"
    )
    os.makedirs(output_dir, exist_ok=True)

    calib_dir = config.get("calib_dir", None)
    probability = config["explore_thres"]
    depth_anchors = create_depth_bins((4, 66, config["n_cand"]))

    print(f"Found {len(png_files)} PNGs")

    for idx, image_path in enumerate(png_files):
        print(f"[{idx + 1}/{len(png_files)}] {image_path}")

        calib_path = get_calib_path_for_image(
            image_path=image_path,
            calib_dir=calib_dir,
            calib_suffix=config.get("calib_suffix", ".json")
        )

        if not os.path.exists(calib_path):
            print(f"[WARN] No calibration file found for {image_path}: {calib_path}")
            continue

        K = load_camera_matrix_from_json(calib_path)

        sample, pil_img = preprocess_png(image_path)
        sample = sample.to(dev)

        start_time = datetime.now()
        with torch.no_grad():
            sample_batch = sample.unsqueeze(0)
            stxl_infer = model(sample_batch)
        print(f"inference time: {datetime.now() - start_time}")

        stxl_infer = stxl_infer.detach().cpu()

        stxl_wrld = revert_fn(
            prediction=stxl_infer,
            anchors=depth_anchors,
            img_height=1280,
            prob=probability
        )

        # Kamera-Matrix ergänzen
        attach_camera_matrix(stxl_wrld[0], K)
        attach_image_to_stixel_world(stxl_wrld[0], image_path)

        filename = os.path.splitext(os.path.basename(image_path))[0]
        stxl_wrld[0].context.name = filename
        stxl_wrld[0].context.calibration.T.extend(np.identity(4).flatten().tolist())
        stxl_wrld[0].context.calibration.width = 1920

        stx.save(
            stxl_wrld=stxl_wrld[0],
            filepath=output_dir,
            export_image=True,
            sys_out=True
        )

        print(f"[INFO] Saved: {filename}")

    print(f"Done. Total runtime: {datetime.now() - overall_start_time}")


if __name__ == "__main__":
    main()
