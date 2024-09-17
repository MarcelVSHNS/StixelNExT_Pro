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


# 0. Implementation of a Dataset
class StixelData(Dataset):
    # 1. Implement __init()__
    def __init__(self,
                 data_dir: str,
                 phase: str,
                 mode: str,
                 transform: bool = False,
                 target_trans_blur: bool = False,
                 depth_anchors=False):
        self.data_dir = os.path.join(data_dir, phase)
        self.name: str = f"{os.path.basename(data_dir)}.{phase}"
        if depth_anchors:
            self.depth_anchors = pd.read_csv(os.path.join(data_dir, "depth_anchors.csv"), index_col=0)
        else:
            self.depth_anchors = _create_depth_bins(5, 50, 64)
        self.sample_map: List[str] = os.listdir(os.path.join(self.data_dir))
        self.mode = mode
        self.transform = transform
        self.target_trans_blur = target_trans_blur

    # 2. Implement __len()__
    def __len__(self) -> int:
        return len(self.sample_map)

    # 3. Implement __getitem()__
    def __getitem__(self, idx):
        stxl_wrld: stx.StixelWorld = stx.read(os.path.join(self.data_dir ,self.sample_map[idx]))
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
        # delete ground truth Stixel from object
        # del stxl_wrld.stixel[:]
        return feature_image, target_labels, os.path.join(self.data_dir ,self.sample_map[idx])

    def _classification_target_label(self, y_target: np.array,
                                     out_bins: int = 12,
                                     i_attr: int = 3,
                                     u_scale: int = 8,
                                     d_scale: float = 50.0,
                                     shadowing: bool = False
                                     ) -> torch.tensor:
        d_scale = d_scale * 0.1
        y_target = pd.DataFrame(y_target)
        # img_path,x,yT,yB,class,depth: prepare data like normalization and scaling
        y_target['u'] = (y_target['u'] // u_scale).astype(int)  # u as index
        y_target['vT'] = (y_target['vT'] / self.img_size['height']).astype(float)  # vT
        y_target['vB'] = (y_target['vB'] / self.img_size['height']).astype(float)  # vB
        # inverted depth and scaled over 100 m
        # y_target['d'] = 1 - y_target['d'] / d_scale
        width = self.img_size['width'] // u_scale
        y_target = y_target.sort_values(by='vT', ascending=False)

        gt_stx_mtx = np.zeros((width, out_bins, i_attr))
        for index, stixel in y_target.iterrows():
            col = int(stixel['u'])
            anchor, anchor_idx = _find_nearest_depth(self.depth_anchors[f'{col}'], stixel['d'])
            # encoding: bottom point vB, top point vT, distance d, probability P
            if i_attr == 4:
                anchor_depth = (stixel['d'] - anchor) / d_scale
                gt_stx_mtx[col][anchor_idx] = [stixel['vB'], stixel['vT'], anchor_depth, 1.0]
                # adds a negative shadow to every entry (2 times) and set the probability accordingly
                if shadowing and anchor_idx >= 2 and gt_stx_mtx[col][anchor_idx - 1][3] == 0.0:
                    anchor_depth_1 = (stixel['d'] - self.depth_anchors[f'{col}'][anchor_idx - 1]) / d_scale
                    gt_stx_mtx[col][anchor_idx - 1] = [stixel['vB'], stixel['vT'], anchor_depth_1, 0.66]
                    anchor_depth_2 = (stixel['d'] - self.depth_anchors[f'{col}'][anchor_idx - 2]) / d_scale
                    gt_stx_mtx[col][anchor_idx - 2] = [stixel['vB'], stixel['vT'], anchor_depth_2, 0.25]
            elif i_attr == 3:
                gt_stx_mtx[col][anchor_idx] = [stixel['vB'], stixel['vT'], 1.0]
                if shadowing and anchor_idx >= 2 and gt_stx_mtx[col][anchor_idx - 1][2] == 0.0:
                    gt_stx_mtx[col][anchor_idx - 1] = [stixel['vB'], stixel['vT'], 0.66]
                    gt_stx_mtx[col][anchor_idx - 2] = [stixel['vB'], stixel['vT'], 0.25]
            else:
                raise NotImplementedError("Check num attributes.")
        # e.g. w=240 x n=12 x a=4
        label = torch.from_numpy(gt_stx_mtx).to(torch.float32)
        label = rearrange(label, "w n a -> a n w")
        return label

    def _segmentation_target_label(self, y_target: np.array,
                                   out_bins: int = 192,
                                   u_scale: int = 8,
                                   v_scale: int = 8
                                   ) -> torch.tensor:
        y_target['u'] //= u_scale
        y_target['vT'] //= v_scale
        y_target['vB'] //= v_scale
        width = self.img_size['width'] // u_scale
        height = self.img_size['height'] // v_scale

        # shape: depth, height, width
        gt_stx_mtx = np.zeros((out_bins, height, width))
        for stixel in y_target:
            col = stixel['u']
            if col < 0:
                continue
            anchor, anchor_idx = _find_nearest_depth(self.depth_anchors[f'{col}'], stixel['d'])
            for voxel_col in range(stixel['vT'], stixel['vB']):
                gt_stx_mtx[anchor_idx, int(voxel_col), int(stixel['u'])] = 1
        label = torch.from_numpy(gt_stx_mtx).to(torch.float32)
        return label


def revert_class(prediction: torch.Tensor,
                 anchors: pd.DataFrame,
                 stxl_wrld_paths: List[str],
                 prob: float = 0.9,
                 u_scale: int = 8,
                 d_scale: float = 50.0,
                 four_attr: bool = False
                 ) -> List[StixelWorld]:
    """ extract stixel information from prediction """
    d_scale = d_scale * 0.1
    pred_np = prediction.numpy()
    stixel_world_batch = []
    for batch, path in zip(pred_np, stxl_wrld_paths):
        # print(f"Batch1: {batch.shape}")
        stxl_wrld = stx.read(path)
        del stxl_wrld.stixel[:]
        img_height = stxl_wrld.context.calibration.height
        columns = rearrange(batch, "a n u -> u n a")
        for u in range(len(columns)):
            # print(f"Col1: {column.shape}")
            for n in range(len(columns[u])):
                # print(f"candidate1: {candidate.shape}")
                if four_attr:
                    if columns[u][n][3] >= prob:
                        stxl = Stixel()
                        stxl.u = int(u * u_scale)
                        stxl.vT = int(columns[u][n][1] * img_height)
                        stxl.vB = int(columns[u][n][0] * img_height)
                        stxl.d = columns[u][n][2] * d_scale + anchors[f'{u}'][n]
                        stxl.confidence = columns[u][n][3]
                        stxl.width = u_scale
                        stxl_wrld.stixel.append(stxl)
                else:
                    if columns[u][n][2] >= prob:
                        stxl = Stixel()
                        stxl.u = int(u * u_scale)
                        stxl.vT = int(columns[u][n][1] * img_height +1)
                        stxl.vB = int(columns[u][n][0] * img_height +1)
                        stxl.d = anchors[f'{u}'][n]
                        stxl.confidence = columns[u][n][2]
                        stxl.width = u_scale
                        stxl_wrld.stixel.append(stxl)
        stixel_world_batch.append(stxl_wrld)
    return stixel_world_batch


def revert_segm(prediction: torch.Tensor,
                anchors: pd.DataFrame,
                stxl_wrld_paths: List[str],
                prob: float = 0.9,
                u_scale: int = 8,
                v_scale: int = 8
                ) -> List[StixelWorld]:
    pred_np = prediction.numpy()
    stixel_world_batch = []
    for batch, path in zip(pred_np, stxl_wrld_paths):
        # print(f"Batch1: {batch.shape}")
        stxl_wrld = stx.read(path)
        del stxl_wrld.stixel[:]
        columns = rearrange(batch, "d h w -> w d h")
        for u in range(len(columns)):
            for d in range(len(columns[u])):
                stixel_start = 0
                in_stixel = False
                stixel_prob = []
                for v in range(len(columns[u][d])):
                    if in_stixel:
                        if columns[u][d][v] < prob:
                            stxl = Stixel()
                            stxl.u = int(u * u_scale)
                            stxl.vT = int(stixel_start * v_scale)
                            stxl.vB = int(v * v_scale)
                            stxl.d = anchors[f'{u}'][d]
                            stxl.confidence = sum(stixel_prob) / len(stixel_prob)
                            stxl.width = u_scale
                            stxl_wrld.stixel.append(stxl)
                            in_stixel = False
                        else:
                            stixel_prob.append(columns[u][d][v])
                    else:
                        if columns[u][d][v] >= prob:
                            stixel_start = v
                            stixel_prob.append(columns[u][d][v])
                            in_stixel = True
                        else:
                            pass
        stixel_world_batch.append(stxl_wrld)
    return stixel_world_batch


def _find_nearest_depth(column_anchors: pd.DataFrame, depth, floor=False):
    # Filter the column to get only values smaller or equal to the given value
    if floor:
        column_anchors = column_anchors[column_anchors <= depth]
    # If no such values exist, return the min val
    if column_anchors.empty:
        return depth, 0
    diff = (column_anchors - depth).abs()
    # Find the index of the minimum difference
    idx = diff.idxmin()
    # Get the nearest value using the index
    nearest_value = column_anchors.loc[idx]
    return nearest_value, idx


def _feature_transform_resize(x_features: torch.Tensor, target_size: Dict[str, int]) -> torch.Tensor:
    size = (target_size['width'], target_size['height'])
    x_features_resized = F.interpolate(x_features.unsqueeze(0), size=size, mode='bilinear', align_corners=False)
    return x_features_resized.squeeze(0)


def _target_transform_gaussian_blur(y_target: torch.Tensor, sigma: float = 0.96, normalize: bool = False) -> torch.Tensor:
    stixel_mtx = y_target.numpy()
    blurred_matrix = gaussian_filter(stixel_mtx, sigma=sigma)
    if normalize:
        max_vals = np.max(blurred_matrix, axis=0)
        blurred_matrix = np.divide(blurred_matrix, max_vals, out=np.zeros_like(blurred_matrix), where=max_vals != 0)
    stixel_mtx = np.maximum(blurred_matrix, stixel_mtx)
    return torch.from_numpy(stixel_mtx).to(torch.float32)


def _create_depth_bins(start=5, end=55, num_bins=192):
    bin_vals = np.linspace(start, end, num_bins)
    bin_mtx = np.tile(bin_vals, (240, 1))
    df = pd.DataFrame(bin_mtx)
    df = df.T
    df.columns = [str(i) for i in range(240)]
    return df
