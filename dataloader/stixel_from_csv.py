import pandas
import torch
import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from einops import rearrange
import numpy as np
from typing import List, Tuple
import cv2
import torch.nn.functional as F
import yaml
from stixel import Stixel, StixelWorld


# 0. Implementation of a Dataset
class StixelData(Dataset):
    # 1. Implement __init()__
    def __init__(self, data_dir: str, phase: str, model:  str, annotation_dir="targets", img_dir="FRONT", transform=None,
                 target_transform=None, return_name=False):
        self.data_dir = os.path.join(data_dir, phase)
        with open(data_dir + '/dataset-config.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            self.name = config['name']
            self.img_size = {'height': int(config['img_height']), 'width': int(config['img_width'])}
        self.img_path = os.path.join(self.data_dir, img_dir)
        self.annotation_path = os.path.join(self.data_dir, annotation_dir)
        filenames: List[str] = os.listdir(os.path.join(self.data_dir, img_dir))
        self.sample_map: List[str] = [os.path.splitext(filename)[0] for filename in filenames]
        self.transform = transform
        self.return_name: bool = return_name
        self.model = model
        self.target_transform = target_transform
        self.name: str = os.path.basename(data_dir)

    # 2. Implement __len()__
    def __len__(self) -> int:
        return len(self.sample_map)

    # 3. Implement __getitem()__
    def __getitem__(self, idx):
        img_path_full: str = os.path.join(self.img_path, self.sample_map[idx] + ".png")
        feature_image: torch.Tensor = read_image(img_path_full, ImageReadMode.RGB).to(torch.float32)
        target_labels: pd.DataFrame = pd.read_csv(os.path.join(self.annotation_path, os.path.basename(self.sample_map[idx]) + ".csv"))
        target_labels = self._preparation_of_target_label(target_labels)
        if self.transform:
            feature_image = self.transform(feature_image)
        if self.target_transform:
            target_labels = self.target_transform(target_labels)
        # data type needs to be like the NN layer like .to(torch.float32)
        if self.return_name:
            return feature_image, target_labels, self.sample_map[idx]
        else:
            return feature_image, target_labels

    def _preparation_of_target_label(self, y_target: pandas.DataFrame, epsilon=1e-6, n_obj_preds: int = 12) -> torch.tensor:
        # img_path,x,yT,yB,class,depth: prepare data like normalization and scaling
        y_target['x'] = (y_target['x'] // 8).astype(int)                            # u as index
        y_target['yT'] = (y_target['yT'] / self.img_size['height']).astype(float)           # vT
        y_target['yB'] = (y_target['yB'] / self.img_size['height']).astype(float)           # vB
        # inverted depth and scaled over 100 m
        y_target['depth'] = 100.0 / (y_target['depth'] + epsilon)
        width = int(self.img_size['width'] / 8)

        gt_lst = []
        for _ in range(width):
            gt_lst.append([])
        for index, stixel in y_target.iterrows():
            col = stixel['x']
            # encoding: bottom point vB, top point vT, distance d, probability P
            if len(gt_lst[col]) < n_obj_preds:
                gt_lst[col].append([stixel['yB'], stixel['yT'], stixel['depth'], 1])
        # fill cols with zeros
        for col_list in gt_lst:
            while len(col_list) != n_obj_preds:
                col_list.append([0, 0, 0, 0])
        gt_mtx: np.array = np.array(gt_lst)
        # e.g. w=240 x n=12 x a=4
        label = torch.from_numpy(gt_mtx).to(torch.float32)
        label = rearrange(label, "w n a -> a n w")
        if self.model == "unet":
            # 15 x 4 x 240
            return rearrange(label, "a n w -> n a w")
        return label

    @staticmethod
    def revert(prediction: torch.Tensor, image_name: str = "", prob: float = 0.9) -> List[StixelWorld]:
        """ extract stixel information from prediction """
        pred_np = prediction.numpy()
        stixel_world_batch = []
        for batch in pred_np:
            stixel_world = []
            # print(f"Batch1: {batch.shape}")
            columns = rearrange(batch, "a n w -> w n a")
            for u in range(len(columns)):
                # print(f"Col1: {column.shape}")
                for candidate in columns[u]:
                    # print(f"candidate1: {candidate.shape}")
                    if candidate[3] >= prob:
                        stixel = Stixel(u=u,
                                        v_b=candidate[0],
                                        v_t=candidate[1],
                                        d=candidate[2],
                                        prob=candidate[3])
                        stixel_world.append(stixel)
            stixel_world_batch.append(StixelWorld(stixel_world, img_name=image_name))
        return stixel_world_batch


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
