import torch
import os
import io
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter
from einops import rearrange
import numpy as np
from typing import List, Tuple, Dict, Optional
import stixel as stx
import torch.nn.functional as F

from stixel import Stixel
from stixel.stixel_world_pb2 import StixelWorld
import datetime

LOG_DEPTH_EPS = 1e-6


# 0. Implementation of a Dataset
class StixelData(Dataset):
    # 1. Implement __init()__
    def __init__(self,
                 data_dir: str,
                 phase: str,
                 mode: str,
                 depth_anchors: Tuple[int, int, int],
                 return_depth_maps: bool = False,
                 u_scale: int = 8,
                 v_scale: int = 8,
                 transform: bool = False,
                 target_trans_blur: bool = False):
        self.data_dir = os.path.join(data_dir, phase)
        self.name: str = f"{os.path.basename(data_dir)}.{phase}"
        # self.depth_anchors = _create_depth_bins(depth_anchors)
        self.depth_anchors = create_depth_bins(depth_anchors)
        self.sample_map: List[str] = sorted(os.listdir(os.path.join(self.data_dir)))
        self.mode = mode
        self.return_depth_maps = return_depth_maps
        self.u_scale = u_scale
        self.v_scale = v_scale
        self.transform = transform
        self.target_trans_blur = target_trans_blur

    # 2. Implement __len()__
    def __len__(self) -> int:
        return len(self.sample_map)

    # 3. Implement __getitem()__
    def __getitem__(self, idx):
        stxl_wrld: stx.StixelWorld = stx.read(os.path.join(self.data_dir, self.sample_map[idx]))
        self.img_size = {'height': stxl_wrld.context.calibration.height, 'width': stxl_wrld.context.calibration.width}
        img = np.array(Image.open(io.BytesIO(stxl_wrld.image)))
        feature_image: torch.Tensor = torch.from_numpy(img).to(torch.float32)
        feature_image = rearrange(feature_image, "h w c -> c h w")
        target_labels = stx.convert_to_matrix(stxl_wrld)
        if self.mode == "classification":
            target_labels = self._classification_target_label(target_labels, out_bins=self.depth_anchors.shape[0])
        elif self.mode == "segmentation":
            target_labels = self._segmentation_target_label(target_labels, out_bins=self.depth_anchors.shape[0])
        else:
            raise ValueError(f"Mode {self.mode} not recognized.")
        if self.transform:
            feature_image = _feature_transform_resize(feature_image, self.img_size)
        if self.target_trans_blur and self.mode == "segmentation":
            target_labels = _target_transform_gaussian_blur(target_labels)
        if self.return_depth_maps:
            gt_depth_map, gt_mask = _stixel_world_to_depth_map(stxl_wrld, u_scale=self.u_scale, v_scale=self.v_scale)
            return feature_image, target_labels, os.path.join(self.data_dir, self.sample_map[idx]), gt_depth_map, gt_mask
        # delete ground truth Stixel from object
        # del stxl_wrld.stixel[:]
        return feature_image, target_labels, os.path.join(self.data_dir, self.sample_map[idx])

    def _classification_target_label(self, y_target: np.array,
                                     out_bins: int = 64,
                                     u_scale: int = 8,
                                     ) -> torch.tensor:
        y_target = pd.DataFrame(y_target)
        # img_path,x,yT,yB,class,depth: prepare data like normalization and scaling
        y_target['u'] = (y_target['u'] // u_scale).astype(int)  # u as index
        y_target['vT'] = (y_target['vT'] / self.img_size['height']).astype(float)  # vT
        y_target['vB'] = (y_target['vB'] / self.img_size['height']).astype(float)  # vB
        # inverted depth and scaled over 100 m
        # y_target['d'] = 1 - y_target['d'] / d_scale
        width = self.img_size['width'] // u_scale
        y_target = y_target.sort_values(by='vT', ascending=False)

        gt_stx_mtx = np.zeros((width, out_bins, 4))
        for index, stixel in y_target.iterrows():
            col = int(stixel['u'])
            if col < 0:
                continue
            # ignore instances with label "sign"
            if int(stixel['label']) == 3:
                continue
            anchor, anchor_idx = _find_nearest_depth(self.depth_anchors[f'{col}'], stixel['d'], floor=True)
            depth_residual = _encode_depth_residual(stixel['d'], anchor)
            gt_stx_mtx[col][anchor_idx] = [stixel['vB'], stixel['vT'], depth_residual, 1.0]
        # e.g. w=240 x n=12 x a=4
        label = torch.from_numpy(gt_stx_mtx).to(torch.float32)
        label = rearrange(label, "w n a -> a n w")
        return label 


def revert_class(prediction: torch.Tensor,
                 anchors: pd.DataFrame,
                 img_height: int,
                 prob: float = 0.9,
                 u_scale: int = 8,
                 stxl_wrld_paths: Optional[List[str]] = None
                 ) -> List[StixelWorld]:
    """ extract stixel information from prediction """
    pred_np = prediction.numpy()
    stixel_world_batch = []
    for batch in pred_np:
        # print(f"Batch1: {batch.shape}")
        stxl_wrld = stx.StixelWorld()
        stxl_wrld.context.calibration.height = img_height
        columns = rearrange(batch, "a n u -> u n a")
        for u in range(len(columns)):
            # print(f"Col1: {column.shape}")
            for n in range(len(columns[u])):
                # print(f"candidate1: {candidate.shape}")
                # if columns[u][n][3] >= prob:
                stxl = Stixel()
                stxl.u = int(u * u_scale)
                stxl.vT = int(columns[u][n][1] * img_height + 1)
                stxl.vB = int(columns[u][n][0] * img_height + 1)
                anchor_depth = anchors[f'{u}'][n]
                stxl.d = _decode_depth_residual(columns[u][n][2], anchor_depth)
                stxl.confidence = columns[u][n][3]
                stxl.width = u_scale
                stxl_wrld.stixel.append(stxl)
        stixel_world_batch.append(stxl_wrld)
    return stixel_world_batch


def _find_nearest_depth(column_anchors: pd.DataFrame, depth, floor=False):
    # Filter the column to get only values smaller or equal to the given value
    if floor:
        floor_anchors = column_anchors[column_anchors <= depth]
        if floor_anchors.empty:
            return column_anchors.iloc[0], 0
        column_anchors = floor_anchors
    # If no such values exist, return the min val
    if column_anchors.empty:
        return depth, 0
    diff = (column_anchors - depth).abs()
    # Find the index of the minimum difference
    idx = diff.idxmin()
    # Get the nearest value using the index
    nearest_value = column_anchors.loc[idx]
    return nearest_value, idx


def _encode_depth_residual(depth: float, anchor_depth: float) -> float:
    depth = max(float(depth), LOG_DEPTH_EPS)
    anchor_depth = max(float(anchor_depth), LOG_DEPTH_EPS)
    return float(np.log(depth) - np.log(anchor_depth))


def _decode_depth_residual(depth_residual: float, anchor_depth: float) -> float:
    anchor_depth = max(float(anchor_depth), LOG_DEPTH_EPS)
    return float(np.exp(np.log(anchor_depth) + float(depth_residual)))


def _feature_transform_resize(x_features: torch.Tensor, target_size: Dict[str, int]) -> torch.Tensor:
    size = (target_size['width'], target_size['height'])
    x_features_resized = F.interpolate(x_features.unsqueeze(0), size=size, mode='bilinear', align_corners=False)
    return x_features_resized.squeeze(0)


def _target_transform_gaussian_blur(y_target: torch.Tensor, sigma: float = 0.96,
                                    normalize: bool = False) -> torch.Tensor:
    stixel_mtx = y_target.numpy()
    blurred_matrix = gaussian_filter(stixel_mtx, sigma=sigma)
    if normalize:
        max_vals = np.max(blurred_matrix, axis=0)
        blurred_matrix = np.divide(blurred_matrix, max_vals, out=np.zeros_like(blurred_matrix), where=max_vals != 0)
    stixel_mtx = np.maximum(blurred_matrix, stixel_mtx)
    return torch.from_numpy(stixel_mtx).to(torch.float32)


def create_depth_bins(cfg: Tuple[int, int, int]):
    start, end, num_bins = cfg
    min_value = 0
    max_value = np.pi / 2.72  # 2.4

    linear_space = np.linspace(min_value, max_value, num_bins)
    tangent_space = np.tan(linear_space)
    bin_vals = start + (tangent_space - tangent_space.min()) / (tangent_space.max() - tangent_space.min()) * (
            end - start)

    bin_mtx = np.tile(bin_vals, (240, 1))
    df = pd.DataFrame(bin_mtx)
    df = df.T
    df.columns = [str(i) for i in range(240)]
    return df


def _create_depth_bins_linear(cfg: Tuple[int, int, int]):
    start, end, num_bins = cfg
    bin_vals = np.linspace(start, end, num_bins)
    bin_mtx = np.tile(bin_vals, (240, 1))
    df = pd.DataFrame(bin_mtx)
    df = df.T
    df.columns = [str(i) for i in range(240)]
    return df


def _stixel_world_to_depth_map(stxl_wrld: stx.StixelWorld,
                               u_scale: int = 8,
                               v_scale: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert StixelWorld GT stixels into a stixel-raster depth map [H, W] in meters and valid mask.
    """
    img_height = int(stxl_wrld.context.calibration.height)
    img_width = int(stxl_wrld.context.calibration.width)
    stixel_height = max(img_height // int(v_scale), 1)
    stixel_width = max(img_width // int(u_scale), 1)

    depth_map = np.full((stixel_height, stixel_width), np.inf, dtype=np.float32)
    valid_mask = np.zeros((stixel_height, stixel_width), dtype=bool)

    for stxl in stxl_wrld.stixel:
        depth_val = float(stxl.d)
        if not np.isfinite(depth_val) or depth_val <= 0.0:
            continue

        u_start = int(stxl.u) // int(u_scale)
        u_span = max(int(stxl.width) // int(u_scale), 1)
        u_end = u_start + u_span - 1

        v_top = int(stxl.vT)
        v_bottom = int(stxl.vB)
        if v_top > v_bottom:
            v_top, v_bottom = v_bottom, v_top

        v_top = max(0, min(v_top, img_height - 1))
        v_bottom = max(0, min(v_bottom, img_height - 1))

        v_top = max(0, min(v_top // int(v_scale), stixel_height - 1))
        v_bottom = max(0, min(v_bottom // int(v_scale), stixel_height - 1))

        u_start = max(0, min(u_start, stixel_width - 1))
        u_end = max(0, min(u_end, stixel_width - 1))
        if v_bottom < v_top or u_end < u_start:
            continue

        current = depth_map[v_top: v_bottom + 1, u_start: u_end + 1]
        depth_map[v_top: v_bottom + 1, u_start: u_end + 1] = np.minimum(current, depth_val)
        valid_mask[v_top: v_bottom + 1, u_start: u_end + 1] = True

    depth_map[~valid_mask] = 0.0
    return torch.from_numpy(depth_map), torch.from_numpy(valid_mask)
