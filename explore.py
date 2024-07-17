import torch
# import torchvision.ops
# from torchvision.models.convnext import ConvNeXt
from models import convnext_stixel
from torchinfo import summary
from losses import StixelObjectLoss


""" Model exploration """
# model = UNet(n_channels=3)
# model = ConvNeXt(in_channels=3, c=60, depths_b=[3, 3, 27, 3])
model, _ = convnext_stixel()

input_shape = (4, 3, 1280, 1920)
x = torch.randn(input_shape)
output = model(x)

summary(model, input_size=input_shape, device=torch.device('cpu'))
print(f"Output shape: {output.shape}")


""" Loss exploration """
inputs = torch.rand(4, 4, 12, 240)
loss_fn = StixelObjectLoss()
l1, _ = loss_fn(output, output)
print(f"Ident: {l1}")
target = torch.rand(4, 4, 12, 240)
l2, _ = loss_fn(output, target)
print(f"Diff: {l2}")


