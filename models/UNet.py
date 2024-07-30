import yaml
import torch
from functools import partial
from typing import List, Dict, Any, Tuple
from torch.utils.checkpoint import checkpoint
from torch import nn, Tensor
from torchvision.ops.misc import Conv2dNormActivation, Permute
from torchvision.ops.stochastic_depth import StochasticDepth
from einops import rearrange
import torch.nn.functional as F


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = rearrange(x, 'b c h w -> b h w c')
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = rearrange(x, 'b h w c -> b c h w')
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_bins=64, c_width=96, b_level=3, block_expansion=4):
        super(UNet, self).__init__()
        c = c_width
        norm_layer = partial(LayerNorm2d, eps=1e-6)
        self.stem = Conv2dNormActivation(
                in_channels=in_channels,
                out_channels=c_width,
                kernel_size=8,
                stride=8,
                padding=0,
                norm_layer=norm_layer,
                activation_layer=None,
                bias=True,
            )
        self.level_layer_dwn = nn.ModuleList()
        self.level_layer_up = nn.ModuleList()
        for level in range(b_level):
            self.level_layer_dwn.append(
                Down(in_channels=c * 2 ** level, out_channels=c * 2 ** (level + 1), expansion=block_expansion)
            )
            self.level_layer_up.append(
                Up(in_channels=c * 2 ** (level + 1), out_channels=c * 2 ** level, expansion=block_expansion)
            )
        self.level_layer_up = self.level_layer_up[::-1]

        self.head = UnetHead(in_channels=c, out_channels=out_bins)

    def forward(self, x):
        x1 = self.stem(x)
        # Encoder
        down_x = [x1]
        for level_layer in self.level_layer_dwn:
            down_x.append(level_layer(down_x[-1]))
        # Decoder
        result = down_x[-1]
        down_x.reverse()
        for x_i in range(len(down_x) - 1):
            result = self.level_layer_up[x_i](result, down_x[x_i + 1])

        pred = self.head(result)
        return pred


class UNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, expansion: int = 4,
                 layer_scale: float = 1e-6, stochastic_depth_prob: float = 0.0) -> None:
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels, bias=True),
            Permute([0, 2, 3, 1]),
            norm_layer(in_channels),
            nn.Linear(in_features=in_channels, out_features=expansion * in_channels, bias=True),
            nn.GELU(),
            nn.Linear(in_features=expansion * in_channels, out_features=out_channels, bias=True),
            Permute([0, 3, 1, 2]),
        )
        self.layer_scale = nn.Parameter(torch.ones(out_channels, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        result = self.layer_scale * result
        result = self.stochastic_depth(result)
        return result


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=4):
        super().__init__()
        self.in_c = in_channels
        self.out_c = out_channels
        norm_layer = partial(LayerNorm2d, eps=1e-6)
        self.downsampling = nn.Sequential(
            norm_layer(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2))
        self.ublock = UNetBlock(out_channels, out_channels, expansion=expansion)

    def forward(self, x):
        x = self.downsampling(x)
        x = self.ublock(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=4):
        super().__init__()
        self.in_c = in_channels
        self.out_c = out_channels
        norm_layer = partial(LayerNorm2d, eps=1e-6)
        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            norm_layer(out_channels))
        self.ublock = UNetBlock(in_channels, out_channels, expansion=expansion)

    def forward(self, x1, x2):
        x1 = self.upsampling(x1)
        if x1.shape != x2.shape:
            # input is c, h, w
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.ublock(x)
        return x


class UnetHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv(x)
        return x


def unet() -> Tuple[UNet, Dict[str, Any]]:
    with open('models/unet-config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    c_width: int = config['widths_c']
    level_b: int = config['level_b']
    out_bins: int = config['out_bins']
    block_exp = config['block_expansion']
    model_cfg = {'C': c_width, 'B': level_b, 'out_bins': out_bins, 'block_exp': block_exp}
    model = UNet(c_width=c_width,
                 b_level=level_b,
                 out_bins=out_bins,
                 block_expansion=block_exp)
    return model, model_cfg
