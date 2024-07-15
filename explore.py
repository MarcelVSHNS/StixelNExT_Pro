import torch
from torchvision.models.convnext import ConvNeXt
from models import convnext_stixel
from torchinfo import summary

# model = UNet(n_channels=3)
# model = ConvNeXt(in_channels=3, c=60, depths_b=[3, 3, 27, 3])
model: ConvNeXt = convnext_stixel()

input_shape = (1, 3, 1280, 1920)
output = torch.randn(input_shape)
output = model(output)

summary(model, input_size=input_shape, device=torch.device('cpu'))
print(f"Output shape: {output.shape}")
