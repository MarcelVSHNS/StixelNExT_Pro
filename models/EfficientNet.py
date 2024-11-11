from functools import partial
from torch import nn, Tensor
from typing import Any, List, Optional, Tuple, Dict
from torchvision.models import EfficientNet, EfficientNet_V2_S_Weights
from torchvision.models.efficientnet import _efficientnet, efficientnet_v2_s, _efficientnet_conf
import yaml


def efficientnet_stixel(config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Tuple[EfficientNet, Dict[str, Any]]:
    model_params = {}

    weights = EfficientNet_V2_S_Weights.DEFAULT
    weights = EfficientNet_V2_S_Weights.verify(weights)
    print(f"Pretrained weights loaded for {weights}.")

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_s")

    model = _efficientnet(inverted_residual_setting=inverted_residual_setting,
                          dropout=kwargs.pop("dropout", 0.2),
                          last_channel=last_channel,
                          weights=weights,
                          progress=True,
                          norm_layer=partial(nn.BatchNorm2d, eps=1e-03),
                          **kwargs)
    # model.avgpool = nn.AvgPool2d(kernel_size=(40, 1), stride=(40, 1))
    return model, model_params