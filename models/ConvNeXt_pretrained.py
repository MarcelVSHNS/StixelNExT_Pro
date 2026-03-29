from typing import Any, List, Optional, Tuple, Dict
import torch
from torch import nn, Tensor
from torchvision.models import ConvNeXt, ConvNeXt_Tiny_Weights
from torchvision.models.convnext import CNBlockConfig, _convnext
import torch.nn.functional as F
from einops import rearrange
from functools import partial
import yaml


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = rearrange(x, 'b c h w -> b h w c')
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = rearrange(x, 'b h w c -> b c h w')
        return x


class LocalColumnMixing(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5):
        super(LocalColumnMixing, self).__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_size),
            padding=(0, padding),
            groups=channels,
        )
        self.pointwise = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.pointwise(self.activation(self.depthwise(x)))


class StixelHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StixelHead, self).__init__()
        n_attributes = 4
        assert out_channels % n_attributes == 0, "NN depth does not match, adapt n_channels."
        self.n_candidates = out_channels // n_attributes
        norm_layer = partial(LayerNorm2d, eps=1e-6)
        self.up = nn.Sequential(
            nn.Upsample(size=(1, 240), mode='nearest'),
            norm_layer(out_channels))
        self.local_mixing = LocalColumnMixing(channels=in_channels)
        self.shared_reduce = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.geometry_head = nn.Conv2d(in_channels=out_channels, out_channels=2 * self.n_candidates, kernel_size=1)
        self.depth_head = nn.Conv2d(in_channels=out_channels, out_channels=self.n_candidates, kernel_size=1)
        self.probability_head = nn.Conv2d(in_channels=out_channels, out_channels=self.n_candidates, kernel_size=1)
        self.geometry_activation = nn.Sigmoid()
        self.depth_activation = nn.Sigmoid()
        self.probability_activation = nn.Sigmoid()

    def forward(self, x):
        x = x + self.local_mixing(x)
        shared_features = self.up(self.shared_reduce(x))

        geometry = self.geometry_activation(self.geometry_head(shared_features))
        geometry = rearrange(geometry, 'b (a n) h w -> b a n h w', a=2, n=self.n_candidates)

        depth = self.depth_activation(self.depth_head(shared_features))
        depth = rearrange(depth, 'b n h w -> b 1 n h w')

        probability = self.probability_activation(self.probability_head(shared_features))
        probability = rearrange(probability, 'b n h w -> b 1 n h w')

        output = torch.cat((geometry, depth, probability), dim=1)
        return output.squeeze(dim=3)


class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegmentationHead, self).__init__()
        self.channel_reduce = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.out_channels = out_channels
        norm_layer = partial(LayerNorm2d, eps=1e-6)
        self.up = nn.Upsample(size=(160, 240), mode='nearest')
        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=4),
            norm_layer(out_channels))
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # x = self.channel_reduce(x)
        x = self.upsampling(x)
        return self.activation(x)


def convnext_stixel(n_candidates: int = 64, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Tuple[
    ConvNeXt, Dict[str, Any]]:
    if config is None:
        with open('models/convnext-config.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    c: int = config['C']
    depths_b: List[int] = config['B']
    n_attributes = 4
    model_params = {'name': "ConvNeXt", 'C': c, 'B': depths_b, 'n_cand': n_candidates}
    if c == 96 and depths_b == [3, 3, 9, 3]:
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        # weights = ConvNeXt_Tiny_Weights.verify(weights)
        print(f"Pretrained weights loaded for {weights}.")
    else:
        weights = None
        print(f"Custom ConvNeXt settings, C={c}, B={depths_b}.")

    block_setting = [
        CNBlockConfig(input_channels=c, out_channels=c * 2, num_layers=depths_b[0]),
        CNBlockConfig(input_channels=c * 2, out_channels=c * 4, num_layers=depths_b[1]),
        CNBlockConfig(input_channels=c * 4, out_channels=c * 8, num_layers=depths_b[2]),
        CNBlockConfig(input_channels=c * 8, out_channels=None, num_layers=depths_b[3]),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.1)
    model = _convnext(block_setting, stochastic_depth_prob, weights, True, **kwargs)
    model.avgpool = nn.AvgPool2d(kernel_size=(40, 1), stride=(40, 1))
    model.classifier = StixelHead(in_channels=c * 8,
                                  out_channels=n_attributes * n_candidates)
    return model, model_params


def convnext_stixel_segmentation(config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Tuple[
    ConvNeXt, Dict[str, Any]]:
    if config is None:
        with open('models/convnext-config.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    c: int = config['C']
    depths_b: List[int] = config['B']
    n_bins = config['n_bins']
    model_cfg = {'name': "ConvNeXt", 'C': c, 'B': depths_b, 'n_bins': n_bins}
    if c == 96 and depths_b == [3, 3, 9, 3]:
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        # weights = ConvNeXt_Tiny_Weights.verify(weights)
        print(f"Pretrained weights loaded for {weights}.")
    else:
        weights = None
        print(f"Custom ConvNeXt settings, C={c}, B={depths_b}.")

    block_setting = [
        CNBlockConfig(input_channels=c, out_channels=c * 2, num_layers=depths_b[0]),
        CNBlockConfig(input_channels=c * 2, out_channels=c * 4, num_layers=depths_b[1]),
        CNBlockConfig(input_channels=c * 4, out_channels=c * 8, num_layers=depths_b[2]),
        CNBlockConfig(input_channels=c * 8, out_channels=None, num_layers=depths_b[3]),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.1)
    model = _convnext(block_setting, stochastic_depth_prob, weights, True, **kwargs)
    model.avgpool = nn.Identity()
    model.classifier = SegmentationHead(in_channels=c * 8,
                                        out_channels=n_bins)
    return model, model_cfg
