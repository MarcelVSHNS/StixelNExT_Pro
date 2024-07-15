import yaml
import torch
from typing import List, Dict
from torch.utils.checkpoint import checkpoint
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels=3):
        """
        Initializes the UNet class.
        :param n_channels: Number of input channels.
        :param n_obj_preds: Number of object occurrences. Means how many possible objects per row
        """
        super(UNet, self).__init__()
        with open('unet-config.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        n_obj_preds: int = config['n_obj_preds']

        self.inc = (DoubleConv(in_channels, 64))
        self.down1 = (Down(64, 128, 1))
        self.down2 = (Down(128, 256, 2))
        self.down3 = (Down(256, 512, 3))
        self.down4 = (Down(512, 512, 4))
        self.up1 = (Up(1024, 256, 1))
        self.up2 = (Up(512, 128, 2))
        self.up3 = (Up(256, 64, 3))
        self.up4 = (Up(128, 64, 4))
        self.outc = (OutConv(64, n_obj_preds))

    def params(self) -> Dict[str, List[int]]:
        return {}

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        pred = self.outc(x)
        return pred

    def use_checkpointing(self):
        self.inc = checkpoint(self.inc)
        self.down1 = checkpoint(self.down1)
        self.down2 = checkpoint(self.down2)
        self.down3 = checkpoint(self.down3)
        self.down4 = checkpoint(self.down4)
        self.up1 = checkpoint(self.up1)
        self.up2 = checkpoint(self.up2)
        self.up3 = checkpoint(self.up3)
        self.up4 = checkpoint(self.up4)
        self.outc = checkpoint(self.outc)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, stage):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, stage):
        super().__init__()
        self.up = nn.Upsample(scale_factor=(1, 1.193), mode='bilinear', align_corners=True)
        self.long = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.skip = nn.MaxPool2d(kernel_size=(stage ** 2, stage), stride=(stage ** 2, stage))
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.long(x1)
        x2 = self.skip(x2)
        x1_height, x1_width = x1.size()[2:]
        x2_height, x2_width = x2.size()[2:]
        diffY = x2_height - x1_height
        diffX = x2_width - x1_width
        x2 = x2[:, :, diffY // 2: diffY // 2 + x1_height, diffX // 2: diffX // 2 + x1_width]
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(4, 1), stride=(1, 1), padding=(1, 0))
        # self.conv = F.interpolate(input_tensor, size=(4, 240), mode='linear', align_corners=False)

    def forward(self, x):
        return self.conv(x)
