from typing import Any, List, Optional, Tuple, Dict
import torch
from torch import nn, Tensor
from torchvision.models import ConvNeXt, ConvNeXt_Tiny_Weights
from torchvision.models.convnext import CNBlockConfig, _convnext
from torchvision.models._utils import _ovewrite_named_param
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
        self.scale = torch.sqrt(torch.tensor(feature_dim, dtype=torch.float32))

    def forward(self, x):
        batch, feature_dim, height, width = x.size()
        x = x.squeeze(2)
        queries = self.query_layer(x.permute(0, 2, 1))
        keys = self.key_layer(x.permute(0, 2, 1))
        values = self.value_layer(x.permute(0, 2, 1))

        attention_output = torch.zeros_like(values)
        for i in range(width):
            left_index = max(0, i - 1)
            right_index = min(width - 1, i + 1)
            neighborhood_keys = keys[:, left_index:right_index + 1, :]
            neighborhood_values = values[:, left_index:right_index + 1, :]
            query = queries[:, i, :].unsqueeze(1)
            attention_scores = torch.bmm(query, neighborhood_keys.transpose(1, 2)) / self.scale
            attention_weights = F.softmax(attention_scores, dim=-1)
            weighted_sum = torch.bmm(attention_weights, neighborhood_values)
            attention_output[:, i, :] = weighted_sum.squeeze(1)
        attention_output = attention_output.unsqueeze(2)
        return attention_output.permute(0, 3, 2, 1)


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
    def __init__(self, in_channels, out_channels, i_attributes, width, attention=True):
        super(StixelHead, self).__init__()
        norm_layer = partial(LayerNorm2d, eps=1e-6)
        self.up = nn.Sequential(
            nn.Upsample(size=(1, width), mode='nearest'),
            norm_layer(out_channels))
        self.use_attention = attention
        if self.use_attention:
            self.channel_reduce = nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels, kernel_size=1)
            self.attention_layer = ColumnAttention(768)
            self.attention_influence = partial(combine_attention_prediction, method="concat")
        else:
            self.channel_reduce = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.out_channels = out_channels
        self.i_attributes = i_attributes
        self.activation = nn.Sigmoid()

    def forward(self, x):
        if self.use_attention:
            attention_x = self.attention_layer(x)
            x = self.attention_influence(attention_x, x)
        x = self.channel_reduce(x)
        x = self.up(x)
        assert self.out_channels % self.i_attributes == 0, "NN depth does not match, adapt n_channels."
        n_candidates = self.out_channels // self.i_attributes
        x = rearrange(x, 'b (a n) h w -> b a n h w', a=self.i_attributes, n=n_candidates)
        # output = reduce(output, 'b a n h w -> b a n w', 'mean')
        return self.activation(x.squeeze(dim=3))


class StixelConvNeXt(ConvNeXt):
    def forward(self, x: Tensor):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


def convnext_stixel(config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Tuple[
    ConvNeXt, Dict[str, Any]]:
    if config is None:
        with open('models/convnext-config.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    c: int = config['C']
    depths_b: List[int] = config['B']
    i_attr: int = config['i_attr']
    n_candidates: int = config['n_cand']
    name: str = config['name']
    model_params = {'name': name, 'C': c, 'B': depths_b, 'n_cand': n_candidates}
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
    model.avgpool = nn.AvgPool2d(kernel_size=(37, 1), stride=(37, 1))
    model.classifier = StixelHead(in_channels=c * 8,
                                  out_channels=i_attr * n_candidates,
                                  i_attributes=i_attr,
                                  width=120)
    return model, model_params
