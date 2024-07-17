import torch
from torchvision.models.convnext import ConvNeXt
from models import convnext_stixel
from torchinfo import summary
from losses import StixelObjectLoss


""" Model exploration """
# model = UNet(n_channels=3)
# model = ConvNeXt(in_channels=3, c=60, depths_b=[3, 3, 27, 3])
model, _ = convnext_stixel()

input_shape = (1, 3, 1280, 1920)
x = torch.randn(input_shape)
output = model(x)

summary(model, input_size=input_shape, device=torch.device('cpu'))
print(f"Output shape: {output.shape}")


""" Loss exploration """
inputs = torch.rand(1, 4, 12, 240)
loss_fn = StixelObjectLoss()
print(f"Ident: {loss_fn(output, output)}")
target = torch.rand(1, 4, 12, 240)
print(f"Diff: {loss_fn(output, target)}")


