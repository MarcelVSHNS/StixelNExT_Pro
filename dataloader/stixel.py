import pandas
import torch
import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
import numpy as np
from typing import List
import cv2
from utilities.visualization import draw_stixels_on_image
from dataloader.stixel_multicut_interpreter import Stixel
import torch.nn.functional as F
import yaml

with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)


def _fill_mtx_with_points(obj_mtx, points):
    for x, y_t, y_b, cls in np.array(points).astype(int):
        if cls == 1:
            obj_mtx[y_t:y_b + 1, x] = 1


# 0. Implementation of a Dataset
class MultiCutStixelData(Dataset):
    # 1. Implement __init()__
    def __init__(self, data_dir, phase, annotation_dir="targets_from_lidar", img_dir="STEREO_LEFT", transform=None,
                 target_transform=None, return_original_image=False, return_name=False):
        self.data_dir: str = os.path.join(data_dir, phase)
        self.img_path: str = os.path.join(self.data_dir, img_dir)
        self.annotation_path: str= os.path.join(self.data_dir, annotation_dir)
        filenames: List[str] = os.listdir(os.path.join(self.data_dir, img_dir))
        self.sample_map: List[str] = [os.path.splitext(filename)[0] for filename in filenames]
        self.transform = transform
        self.return_original_image: bool = return_original_image
        self.return_name: bool = return_name
        self.target_transform = target_transform
        self.name: str = os.path.basename(data_dir)
        self.img_size = self._determine_image_size()

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
        if self.return_original_image and self.return_name:
            cv2_test_img = cv2.imread(img_path_full)
            if cv2_test_img.shape != (self.img_size['height'], self.img_size['width'], 3):
                cv2_test_img = cv2.resize(cv2_test_img, (self.img_size['width'], self.img_size['height']),
                                          interpolation=cv2.INTER_LINEAR)
            return feature_image, target_labels, cv2_test_img, self.sample_map[idx]
        elif self.return_original_image:
            # a consistent batch stack is mandatory
            cv2_test_img = cv2.imread(img_path_full)
            if cv2_test_img.shape != (self.img_size['height'], self.img_size['width'], 3):
                cv2_test_img = cv2.resize(cv2_test_img, (self.img_size['width'], self.img_size['height']), interpolation=cv2.INTER_LINEAR)
            return feature_image, target_labels, cv2_test_img
        elif self.return_name:
            return feature_image, target_labels, self.sample_map[idx]
        else:
            return feature_image, target_labels

    def _determine_image_size(self):
        if config['img_width'] is None or config['img_height'] is None:
            test_img_path = os.path.join(self.img_path, self.sample_map[0] + ".png")
            test_feature_image = read_image(test_img_path, ImageReadMode.RGB)
            channels, height, width = test_feature_image.shape
            return {'height': height, 'width': width, 'channels': channels}

        else:
            return {'height': config['img_height'], 'width': config['img_width'], 'channels': 3}

    def _preparation_of_target_label(self, y_target: pandas.DataFrame, grid_step: int = config['grid_step']) -> torch.tensor:
        y_target['x'] = (y_target['x'] // grid_step).astype(int)
        y_target['yT'] = (y_target['yT'] // grid_step).astype(int)
        y_target['yB'] = (y_target['yB'] // grid_step).astype(int)
        assert self.img_size['height'] % grid_step == 0
        assert self.img_size['width'] % grid_step == 0
        mtx_width, mtx_height = int(self.img_size['width'] / grid_step), int(self.img_size['height'] / grid_step)
        stixel_mtx = np.zeros((2, mtx_height, mtx_width))
        for index, stixel in y_target.iterrows():
            if stixel['class'] == 1:
                for row in range(stixel['yT'], stixel['yB'] + 1):
                    stixel_mtx[0, int(row), int(stixel['x'])] = 1
                stixel_mtx[1, int(stixel['yB']), int(stixel['x'])] = 1
                stixel_mtx[1, int(stixel['yT']), int(stixel['x'])] = 1
        label = torch.from_numpy(stixel_mtx).to(torch.float32)
        return label

    def check_target(self, tensor_image, target_labels):
        coordinates = target_labels.loc[:, ['x', 'yT', 'yB', 'class']].to_numpy()
        stixels = []
        for x, y_t, y_b, cls in coordinates.astype(int):
            stixels.append(Stixel(x, y_t, y_b, cls))
        print(f"len: {len(stixels)}")
        img = draw_stixels_on_image(tensor_image, stixels)
        #img.show()


def feature_transform_resize(x_features: torch.Tensor) -> torch.Tensor:
    x_features_resized = F.interpolate(x_features.unsqueeze(0), size=(config['img_height'], config['img_width']), mode='bilinear', align_corners=False)
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
