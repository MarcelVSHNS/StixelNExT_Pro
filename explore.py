import torch
import numpy as np

from models import UNet, ConvNeXt
from torchinfo import summary

# model = UNet(n_channels=3)
# model = ConvNeXt(in_channels=3, c=60, depths_b=[3, 3, 27, 3])
# summary(model, input_size=(1, 3, 1280, 1920), device=torch.device('cpu'))

gt_lst = []
for _ in range(240):
    gt_lst.append([])
gt_lst[3].append([0.1, 0.3, 0.2, 1.0])
gt_lst[3].append([0.1, 0.3, 0.2, 1.0])
mein_array = np.array(gt_lst)

print("hi")

