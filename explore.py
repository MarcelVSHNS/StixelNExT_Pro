import torch

from models import UNet
from torchinfo import summary

model = UNet(n_channels=3,
             n_obj_preds=16)
summary(model, input_size=(1, 3, 1280, 1920), device=torch.device('cpu'))
