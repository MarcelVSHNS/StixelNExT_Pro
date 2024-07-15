from typing import Any, List, Optional, Tuple, Dict
from torch import nn
from torchvision.models import ConvNeXt, ConvNeXt_Tiny_Weights
from torchvision.models.convnext import CNBlockConfig, _convnext
from einops import rearrange, reduce
import yaml


class ConvNeXtHead(nn.Module):
    def __init__(self, in_channels):
        super(ConvNeXtHead, self).__init__()
        self.up = nn.Upsample(size=(1, 240), mode='nearest')
        self.identity = nn.Identity()
        self.c = in_channels

    def forward(self, x):
        x = self.up(x)
        assert self.c % 4 == 0, "NN depth does not match, adapt n_channels."
        num_pred = self.c // 4
        x = rearrange(x, 'b (a n) h w -> b a n h w', a=4, n=num_pred)
        # output = reduce(output, 'b a n h w -> b a n w', 'mean')
        return self.identity(x.squeeze(dim=3))


def convnext_stixel(weights: Optional[ConvNeXt_Tiny_Weights] = None, **kwargs: Any) -> Tuple[ConvNeXt, Dict[str, Any]]:
    with open('models/convnext-config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    c: int = config['widths_c']
    depths_b: List[int] = config['depths_b']
    model_params = {'C': c, 'B': depths_b}
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
    model.classifier = ConvNeXtHead(in_channels=c * 8)
    return model, model_params
