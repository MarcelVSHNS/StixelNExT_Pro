from typing import Any, List, Optional, Tuple, Dict
import torch
from torch import nn, Tensor
from torchvision.models import ConvNeXt, ConvNeXt_Tiny_Weights
from torchvision.models.convnext import CNBlockConfig, _convnext
import torch.nn.functional as F
from einops import rearrange, reduce
from functools import partial
import yaml


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = rearrange(x, 'b c h w -> b h w c')
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = rearrange(x, 'b h w c -> b c h w')
        return x


class ColumnAttention(nn.Module):
    def __init__(self, feature_dim):
        super(ColumnAttention, self).__init__()

        self.query_layer = nn.Linear(feature_dim, feature_dim)
        self.key_layer = nn.Linear(feature_dim, feature_dim)
        self.value_layer = nn.Linear(feature_dim, feature_dim)
        self.scale = feature_dim ** 0.5

    def forward(self, x):
        batch, feature_dim, height, width = x.size()
        # If height > 1, pool across height to get per-column features
        if height > 1:
            x = x.mean(dim=2)
        else:
            x = x.squeeze(2)

        # shape: [B, W, C]
        x = x.permute(0, 2, 1)
        queries = self.query_layer(x)
        keys = self.key_layer(x)
        values = self.value_layer(x)

        # global attention across all columns
        attention_scores = torch.bmm(queries, keys.transpose(1, 2)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.bmm(attention_weights, values)

        # back to [B, C, 1, W]
        attention_output = attention_output.permute(0, 2, 1).unsqueeze(2)
        return attention_output


def combine_attention_prediction(attention_output, prediction_output, method='concat'):
    if method == 'concat':
        combined = torch.cat((attention_output, prediction_output), dim=1)
    elif method == 'add':
        combined = attention_output + prediction_output
    elif method == 'mean':
        combined = (attention_output + prediction_output) / 2
    elif method == 'weighted_sum':
        combined = 0.7 * attention_output + 0.3 * prediction_output
    else:
        raise ValueError("Unknown")
    return combined


class StixelHead(nn.Module):
    def __init__(self, in_channels, out_channels, i_attributes):
        super(StixelHead, self).__init__()
        norm_layer = partial(LayerNorm2d, eps=1e-6)
        self.up = nn.Sequential(
            nn.Upsample(size=(1, 240), mode='nearest'),
            norm_layer(out_channels))
        self.channel_reduce = nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels, kernel_size=1)
        self.attention_layer = ColumnAttention(768)
        self.attention_influence = partial(combine_attention_prediction, method="concat")
        self.out_channels = out_channels
        self.i_attributes = i_attributes
        self.activation = nn.Sigmoid()

    def forward(self, x):
        attention_x = self.attention_layer(x)
        x = self.attention_influence(attention_x, x)
        x = self.channel_reduce(x)
        x = self.up(x)
        assert self.out_channels % self.i_attributes == 0, "NN depth does not match, adapt n_channels."
        n_candidates = self.out_channels // self.i_attributes
        x = rearrange(x, 'b (a n) h w -> b a n h w', a=self.i_attributes, n=n_candidates)
        # output = reduce(output, 'b a n h w -> b a n w', 'mean')
        return self.activation(x.squeeze(dim=3))


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
    i_attr: int = config['i_attr']
    model_params = {'name': "ConvNeXt", 'C': c, 'B': depths_b, 'n_cand': n_candidates, 'i_attr': i_attr}
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
                                  out_channels=i_attr * n_candidates,
                                  i_attributes=i_attr)
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
