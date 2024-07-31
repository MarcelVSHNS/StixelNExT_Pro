import pandas
import torch
import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from einops import rearrange
import numpy as np
from typing import List, Tuple, Dict, Optional
import cv2
import torch.nn.functional as F
import yaml
from stixel import Stixel, StixelWorld
import time


# 0. Implementation of a Dataset
class StixelData(Dataset):
    # 1. Implement __init()__
    def __init__(self, data_dir: str,
                 phase: str,
                 model:  str,
                 mode: str,
                 annotation_dir="Stixel",
                 img_dir="FRONT",
                 bin_dir="targets",
                 transform=None,
                 target_transform=None,
                 return_name=False,
                 depth_anchors=False):
        self.data_dir = os.path.join(data_dir, phase)
        with open(data_dir + '/dataset-config.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            self.name: str = f"{os.path.basename(config['name'])}.{phase}"
            self.img_size = {'height': int(config['img_height']), 'width': int(config['img_width'])}
        if depth_anchors:
            self.depth_anchors = pd.read_csv(os.path.join(data_dir, "depth_anchors.csv"), index_col=0)
        else:
            self.depth_anchors = _create_depth_bins(5, 50, 64)
        self.img_path = os.path.join(self.data_dir, img_dir)
        self.annotation_path = os.path.join(self.data_dir, annotation_dir)
        self.bin_dir = os.path.join(self.data_dir, bin_dir)
        filenames: List[str] = os.listdir(os.path.join(self.data_dir, img_dir))
        self.sample_map: List[str] = [os.path.splitext(filename)[0] for filename in filenames]
        self.mode = mode
        self.transform = transform
        self.return_name: bool = return_name
        self.model = model
        self.target_transform = target_transform

    # 2. Implement __len()__
    def __len__(self) -> int:
        return len(self.sample_map)

    # 3. Implement __getitem()__
    def __getitem__(self, idx):
        img_path_full: str = os.path.join(self.img_path, self.sample_map[idx] + ".png")
        feature_image: torch.Tensor = read_image(img_path_full, ImageReadMode.RGB).to(torch.float32)
        target_labels = pd.read_csv(os.path.join(self.annotation_path, os.path.basename(self.sample_map[idx]) + ".csv"))
        if self.mode == "classification":
            target_labels = self._classification_target_label(target_labels, out_bins=self.depth_anchors.shape[0])
        elif self.mode == "segmentation":
            target_labels = self._segmentation_target_label(target_labels, out_bins=self.depth_anchors.shape[0])
        else:
            raise ValueError(f"Mode {self.mode} not recognized.")
        if self.transform:
            feature_image = self.transform(feature_image)
        if self.target_transform:
            target_labels = self.target_transform(target_labels)
        # data type needs to be like the NN layer like .to(torch.float32)
        if self.return_name:
            return feature_image, target_labels, self.sample_map[idx]
        else:
            return feature_image, target_labels

    def _classification_target_label(self, y_target: pandas.DataFrame,
                                     out_bins: int = 12,
                                     i_attr: int = 3,
                                     u_scale: int = 8,
                                     d_scale: float = 50.0,
                                     shadowing: bool = False) -> torch.tensor:
        d_scale = d_scale * 0.1
        # img_path,x,yT,yB,class,depth: prepare data like normalization and scaling
        y_target['u'] = (y_target['u'] // u_scale).astype(int)                              # u as index
        y_target['vT'] = (y_target['vT'] / self.img_size['height']).astype(float)           # vT
        y_target['vB'] = (y_target['vB'] / self.img_size['height']).astype(float)           # vB
        # inverted depth and scaled over 100 m
        # y_target['d'] = 1 - y_target['d'] / d_scale
        width = int(self.img_size['width'] / u_scale)
        y_target = y_target.sort_values(by='vT', ascending=False)

        gt_stx_mtx = np.zeros((width, out_bins, i_attr))
        for index, stixel in y_target.iterrows():
            col = stixel['u']
            anchor, anchor_idx = find_nearest_depth(self.depth_anchors[f'{col}'], stixel['d'])
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

    @staticmethod
    def revert_class(prediction: torch.Tensor,
                     anchors: pd.DataFrame,
                     img_name: List[str] = [""],
                     prob: float = 0.9,
                     img_size: Dict[str, int] = {'height': 1280, 'width': 1920},
                     u_scale: int = 8,
                     d_scale: float = 50.0,
                     four_attr: bool = False) -> List[StixelWorld]:
        """ extract stixel information from prediction """
        d_scale = d_scale * 0.1
        pred_np = prediction.numpy()
        stixel_world_batch = []
        for batch, name in zip(pred_np, img_name):
            stixel_world = []
            # print(f"Batch1: {batch.shape}")
            columns = rearrange(batch, "a n u -> u n a")
            for u in range(len(columns)):
                # print(f"Col1: {column.shape}")
                for n in range(len(columns[u])):
                    # print(f"candidate1: {candidate.shape}")
                    if four_attr:
                        if columns[u][n][3] >= prob:
                            stixel = Stixel(u=int(u * u_scale),
                                            v_b=int(columns[u][n][0] * img_size['height']),
                                            v_t=int(columns[u][n][1] * img_size['height']),
                                            d=columns[u][n][2] * d_scale + anchors[f'{u}'][n],
                                            prob=columns[u][n][3])
                            stixel_world.append(stixel)
                    else:
                        if columns[u][n][2] >= prob:
                            stixel = Stixel(u=int(u * u_scale),
                                            v_b=int(columns[u][n][0] * img_size['height']),
                                            v_t=int(columns[u][n][1] * img_size['height']),
                                            d=anchors[f'{u}'][n],
                                            prob=columns[u][n][2])
                            stixel_world.append(stixel)
            stixel_world_batch.append(StixelWorld(stixel_world, img_name=name))
        return stixel_world_batch

    def _segmentation_target_label(self, y_target: pandas.DataFrame,
                                   out_bins: int = 64,
                                   u_scale: int = 8,
                                   v_scale: int = 8) -> torch.tensor:
        y_target['u'] = (y_target['u'] // u_scale).astype(int)
        y_target['vT'] = (y_target['vT'] // v_scale).astype(int)
        y_target['vB'] = (y_target['vB'] // v_scale).astype(int)
        width = int(self.img_size['width'] / u_scale)
        height = int(self.img_size['height'] / v_scale)

        # shape: depth, height, width
        gt_stx_mtx = np.zeros((out_bins, height, width))
        for index, stixel in y_target.iterrows():
            col: int = stixel['u']
            anchor, anchor_idx = find_nearest_depth(self.depth_anchors[f'{col}'], stixel['d'])
            for voxel_col in range(stixel['vT'], stixel['vB'] + 1):
                gt_stx_mtx[anchor_idx, int(voxel_col), int(stixel['u'])] = 1
        label = torch.from_numpy(gt_stx_mtx).to(torch.float32)
        return label

    @staticmethod
    def revert_segm(prediction: torch.Tensor,
                    anchors: pd.DataFrame,
                    img_name: List[str] = [""],
                    prob: float = 0.9,
                    u_scale: int = 8,
                    v_scale: int = 8) -> List[StixelWorld]:
        pred_np = prediction.numpy()
        stixel_world_batch = []
        for batch, name in zip(pred_np, img_name):
            stixel_world = []
            # print(f"Batch1: {batch.shape}")
            columns = rearrange(batch, "d h w -> w d h")
            for u in range(len(columns)):
                for d in range(len(columns[u])):
                    stixel_start = 0
                    in_stixel = False
                    stixel_prob = []
                    for v in range(len(columns[u][d])):
                        if in_stixel:
                            if columns[u][d][v] < prob:
                                stixel = Stixel(u=int(u * u_scale),
                                                v_b=int(v * v_scale),
                                                v_t=int(stixel_start * v_scale),
                                                d=anchors[f'{u}'][d],
                                                prob=sum(stixel_prob) / len(stixel_prob))
                                stixel_world.append(stixel)
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
            stixel_world_batch.append(StixelWorld(stixel_world, img_name=name))
        return stixel_world_batch


def find_nearest_depth(column_anchors: pd.DataFrame, depth, floor=False):
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


def feature_transform_resize(x_features: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
    x_features_resized = F.interpolate(x_features.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)
    return x_features_resized.squeeze(0)


def overlay_original(matrix: np.array, original: np.array) -> np.array:
    # calculate col-wise to equalize dense points
    max_vals = np.max(matrix, axis=0)
    normalized_matrix = np.divide(matrix, max_vals, out=np.zeros_like(matrix), where=max_vals!=0)
    return np.maximum(normalized_matrix, original)


def target_transform_gaussian_blur(y_target: torch.Tensor) -> torch.Tensor:
    stixel_mtx = y_target.numpy()
    # Occupancy grid [0]
    blur_occupancy = cv2.GaussianBlur(stixel_mtx[0], (5, 7), sigmaX=1.42, sigmaY=1.21)
    stixel_mtx[0] = overlay_original(blur_occupancy, stixel_mtx[0])
    # Cut matrix [1]
    for i in range(5):
        blur_cuts = cv2.GaussianBlur(stixel_mtx[1], (3, 3), sigmaX=2.1, sigmaY=1.01)
        stixel_mtx[1] = overlay_original(blur_cuts, stixel_mtx[1])

    return torch.from_numpy(stixel_mtx).to(torch.float32)


def _create_depth_bins(start=5, end=55, num_bins=192):
    bin_vals = np.linspace(start, end, num_bins)
    bin_mtx = np.tile(bin_vals, (240, 1))
    df = pd.DataFrame(bin_mtx)
    df = df.T
    df.columns = [str(i) for i in range(240)]
    return df
