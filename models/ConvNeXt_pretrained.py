from typing import Any, List, Optional, Tuple, Dict
import torch
from torch import nn, Tensor
from torchvision.models import ConvNeXt, ConvNeXt_Tiny_Weights, EfficientNet, EfficientNet_V2_S_Weights, MobileNetV3, \
    MobileNet_V3_Large_Weights, SwinTransformer, Swin_V2_T_Weights, ShuffleNetV2, ShuffleNet_V2_X2_0_Weights
from torchvision.models.convnext import CNBlockConfig, _convnext
from torchvision.models.efficientnet import _efficientnet_conf
from torchvision.models.mobilenetv3 import _mobilenet_v3_conf
from torchvision.models.swin_transformer import SwinTransformerBlockV2, PatchMergingV2
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
    def __init__(self, in_channels, out_channels, i_attributes):
        super(StixelHead, self).__init__()
        norm_layer = partial(LayerNorm2d, eps=1e-6)
        self.up = nn.Sequential(
            nn.Upsample(size=(1, 240), mode='nearest'),
            norm_layer(out_channels))
        self.channel_reduce = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.attention_layer = ColumnAttention(768)
        self.attention_influence = partial(combine_attention_prediction, method="concat")
        self.out_channels = out_channels
        self.i_attributes = i_attributes
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # attention_x = self.attention_layer(x)
        # x = self.attention_influence(attention_x, x)
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
        self.channel_reduce = nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels, kernel_size=1)
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


class StixelEfficientNetV2(EfficientNet):
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


def efficientnet_stixel(n_candidates: int = 64, **kwargs: Any) -> Tuple[EfficientNet, Dict[str, Any]]:
    model_params = {'name': "ConvNeXt", 'n_cand': n_candidates, 'i_attr': 3}

    weights = EfficientNet_V2_S_Weights.DEFAULT
    weights = EfficientNet_V2_S_Weights.verify(weights)
    print(f"Pretrained weights loaded for {weights}.")
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_s")

    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
    model = StixelEfficientNetV2(inverted_residual_setting=inverted_residual_setting,
                                 dropout=kwargs.pop("dropout", 0.2),
                                 last_channel=last_channel,
                                 norm_layer=partial(nn.BatchNorm2d, eps=1e-03),
                                 **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=True, check_hash=True))

    model.avgpool = nn.AvgPool2d(kernel_size=(40, 1), stride=(40, 1))
    model.classifier = StixelHead(in_channels=1280,
                                  out_channels=3 * n_candidates,
                                  i_attributes=3)
    return model, model_params


class StixelMobileNetV3(MobileNetV3):
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def mobilenet_stixel(n_candidates: int = 64, **kwargs: Any) -> Tuple[MobileNetV3, Dict[str, Any]]:
    model_params = {'name': "ConvNeXt", 'n_cand': n_candidates, 'i_attr': 3}

    weights = MobileNet_V3_Large_Weights.DEFAULT
    weights = MobileNet_V3_Large_Weights.verify(weights)
    print(f"Pretrained weights loaded for {weights}.")
    inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_large", **kwargs)

    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
    model = StixelMobileNetV3(inverted_residual_setting=inverted_residual_setting,
                              last_channel=last_channel,
                              **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=True, check_hash=True))

    model.avgpool = nn.AvgPool2d(kernel_size=(40, 1), stride=(40, 1))
    model.classifier = StixelHead(in_channels=960,
                                  out_channels=3 * n_candidates,
                                  i_attributes=3)
    return model, model_params


class StixelShuffleNetV2(ShuffleNetV2):
    def __init__(self, stages_repeats: List[int], stages_out_channels: List[int]):
        super(StixelShuffleNetV2, self).__init__(stages_repeats=stages_repeats, stages_out_channels=stages_out_channels)
        self.avgpool = nn.AvgPool2d(kernel_size=(40, 1), stride=(40, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        # x = x.mean([2, 3])  # globalpool
        x = self.avgpool(x)
        x = self.fc(x)
        return x


def shufflenet_stixel(n_candidates: int = 64, **kwargs: Any) -> Tuple[MobileNetV3, Dict[str, Any]]:
    model_params = {'name': "ConvNeXt", 'n_cand': n_candidates, 'i_attr': 3}

    weights = ShuffleNet_V2_X2_0_Weights.DEFAULT
    weights = ShuffleNet_V2_X2_0_Weights.verify(weights)
    print(f"Pretrained weights loaded for {weights}.")

    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = StixelShuffleNetV2(stages_repeats=[4, 8, 4],
                               stages_out_channels=[24, 244, 488, 976, 2048])

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=True, check_hash=True))

    model.fc = StixelHead(in_channels=2048,
                          out_channels=3 * n_candidates,
                          i_attributes=3)
    return model, model_params


class StixelSwinTransformer(SwinTransformer):
    def forward(self, x):
        x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)
        x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        x = self.head(x)
        return x


def swin_transformer_stixel(n_candidates: int = 64, **kwargs: Any) -> Tuple[SwinTransformer, Dict[str, Any]]:
    model_params = {'name': "SwinTransformer", 'n_cand': n_candidates, 'i_attr': 3}

    weights = Swin_V2_T_Weights.DEFAULT
    weights = Swin_V2_T_Weights.verify(weights)
    print(f"Pretrained weights loaded for {weights}.")

    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
    model = StixelSwinTransformer(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[8, 8],
        stochastic_depth_prob=0.2,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
        **kwargs,
    )
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=True, check_hash=True))

    model.avgpool = nn.AvgPool2d(kernel_size=(40, 1), stride=(40, 1))
    model.head = StixelHead(in_channels=768,
                            out_channels=3 * n_candidates,
                            i_attributes=3)
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
