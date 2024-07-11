import torch

from models import UNet
from torchsummary import summary

model = UNet(n_channels=3,
             n_classes=2)
summary(model, input_size=(3, 1280, 1920), device=torch.device('cpu'))
