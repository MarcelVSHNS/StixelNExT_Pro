from torch import nn
from torch import Tensor
from typing import List
from torchvision.ops import StochasticDepth
import torch


class ConvNextStem(nn.Sequential):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(nn.Conv2d(in_features, out_features, kernel_size=4, stride=4),                                 # 1.conv2d
                         nn.BatchNorm2d(out_features))                                                                  # 2.BatchNorm2d


class ConvNexStage(nn.Sequential):
    def __init__(self, in_features: int, out_features: int, depth: int) -> None:
        super().__init__(
            # add the downsampler
            nn.Sequential(
                nn.GroupNorm(num_groups=1, num_channels=in_features),                                                   # 3.groupnorm
                nn.Conv2d(in_features, out_features, kernel_size=2, stride=2)),                                         # 4. conv2d
            *[
                BottleNeckBlock(out_features, out_features)
                for _ in range(depth)
            ],
        )


class BottleNeckBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, expansion: int = 4):
        super().__init__()
        self.block = nn.Sequential(
            # narrow -> wide (with depth-wise and bigger kernel)
            nn.Conv2d(in_features, in_features, kernel_size=7, padding=3, bias=False, groups=in_features),              # 5. conv
            # GroupNorm with num_groups=1 is the same as LayerNorm but works for 2D data
            nn.GroupNorm(num_groups=1, num_channels=in_features),                                                       # 6. groupnorm
            # wide -> wide
            nn.Conv2d(in_features, out_features * expansion, kernel_size=1),                                            # 7. conv
            nn.GELU(),                                                                                                  # 8 gelu
            # wide -> narrow
            nn.Conv2d(out_features * expansion, out_features, kernel_size=1),                                           # 9.conv
        )

    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.block(x)
        x += res
        return x


class Head(nn.Module):
    def __init__(self, in_channels):
        super(Head, self).__init__()
        # self.down = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=(5, 2), stride=(5, 2))

    def forward(self, x):
        return self.conv(x)


class ConvNeXt(nn.Module):
    """ Defaults: in_channels=3, widths_c=96, depths_b=[3, 3, 9, 3]"""
    def __init__(self, in_channels, c, depths_b, stem_features=64):
        super().__init__()
        # Stem
        self.stem = ConvNextStem(in_channels, stem_features)
        self.stage_1 = ConvNexStage(in_features=stem_features, out_features=c, depth=depths_b[0])
        self.stage_2 = ConvNexStage(in_features=c, out_features=c * 2, depth=depths_b[1])
        self.stage_3 = ConvNexStage(in_features=c * 2, out_features=c * 4, depth=depths_b[2])
        self.stage_4 = ConvNexStage(in_features=c * 4, out_features=c * 8, depth=depths_b[3])
        # Head
        self.head = Head(in_channels=c * 8)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.stage_4(x)
        x = self.head(x)
        # Drop all dimensions with size 1
        return x.squeeze()
