import warnings
from collections import namedtuple

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Callable, Any

model_urls = {
    # GoogLeNet ported from TensorFlow
    'googlenet': 'https://download.pytorch.org/models/googlenet-1378be20.pth',
}

GoogLeNetOutputs = namedtuple('GoogLeNetOutputs', ['logits', 'aux_logits2', 'aux_logits1'])
GoogLeNetOutputs.__annotations__ = {'logits': torch.Tensor, 'aux_logits2': Optional[torch.Tensor],
                                    'aux_logits1': Optional[torch.Tensor]}

# Script annotations failed with _GoogleNetOutputs = namedtuple ...
# _GoogLeNetOutputs set here for backwards compat
_GoogLeNetOutputs = GoogLeNetOutputs


class GoogleNet_(nn.Module):
    def __init__(self, num_classes: int = 10, aux_logits: bool = True, transform_input: bool = False, blocks: Optional[List[Callable[..., nn.Module]]] = None) -> None:
        super().__init__()
        if blocks is None:
            blocks = [ConvBlock, Inception, InceptionAux]
        assert len(blocks) == 3, "blocks size should be 3!"

        conv_block = blocks[0]
        inception_block = blocks[1]
        inception_aux_block = blocks[2]

        self.aux_logits = aux_logits
        self.transform_input = transform_input

        # 没有加入原文中的LocalRespNorm，因这是来自Alexnet的概念，实际上在VGG表明LRN没有多大用，所有改为BN
        self.features_1 = nn.Sequential(
            conv_block(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            conv_block(64, 64, kernel_size=1),
            conv_block(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            inception_block(192, 64, 96, 128, 16, 32, 32),
            inception_block(256, 128, 128, 192, 32, 96, 64),  # in_channels说明： inception中的conv1+conv3+conv5+maxpool = 256
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            inception_block(480, 192, 96, 208, 16, 48, 64)
        )

        self.features_2 = nn.Sequential(
            inception_block(512, 160, 112, 224, 24, 64, 64),
            inception_block(512, 128, 128, 256, 24, 64, 64),
            inception_block(512, 112, 144, 288, 32, 64, 64),
            inception_block(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        )

        self.features_3 = nn.Sequential(
            inception_block(832, 256, 160, 320, 32, 128, 128),
            inception_block(832, 384, 192, 384, 48, 128, 128)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1024, num_classes)
        )

        if aux_logits:
            self.aux1 = inception_aux_block(512, num_classes)
            self.aux2 = inception_aux_block(528, num_classes)
        else:
            self.aux1 = None  # type: ignore[assignment]
            self.aux2 = None  # type: ignore[assignment]

    # @torch.jit.unused
    def eager_outputs(self, x: torch.Tensor, aux2: torch.Tensor, aux1: Optional[torch.Tensor]) -> GoogLeNetOutputs:
        if self.training and self.aux_logits:
            return _GoogLeNetOutputs(x, aux2, aux1)
        else:
            return x   # type: ignore[return-value]

    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        x = self.features_1(x)
        aux1: Optional[torch.Tensor] = None
        if self.aux1 is not None:
            if self.training:
                aux1 = self.aux1(x)

        x = self.features_2(x)
        aux2: Optional[torch.Tensor] = None
        if self.aux2 is not None:
            if self.training:
                aux2 = self.aux2(x)

        x = self.features_3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x, aux2, aux1

    def forward(self, x: torch.Tensor) -> GoogLeNetOutputs:
        x = self._transform_input(x)
        x, aux1, aux2 = self._forward(x)
        aux_defined = self.training and self.aux_logits
        # if torch.jit.is_scripting():
        if not aux_defined:
            warnings.warn("Scripted GoogleNet always returns GoogleNetOutputs Tuple")
        return GoogLeNetOutputs(x, aux2, aux1)
        # else:
        #     return self.eager_outputs(x, aux2, aux1)


class Inception(nn.Module):
    def __init__(
            self,
            in_channels: int,
            ch1x1: int,
            ch3x3red: int,
            ch3x3: int,
            ch5x5red: int,
            ch5x5: int,
            pool_proj: int,
            conv_block: Optional[Callable[..., nn.Module]] = None) -> None:  # Callable可调用类型 Optional[int] 类似联合类型默认 + None， Union[int, None]
        super().__init__()
        if conv_block is None:
            conv_block = ConvBlock

        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)  # 注意此处的kernel_size 通过kwargs传递
        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3x3red, kernel_size=1),  # red 为reduce缩写，由于google大量存在降维的操作，所以先降维，再卷积
            conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, ch5x5red, kernel_size=1),
            conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_1 = self.branch1(x)
        x_2 = self.branch2(x)
        x_3 = self.branch3(x)
        x_4 = self.branch4(x)

        y = torch.cat([x_1, x_2, x_3, x_4], 1)  # 0为batch
        return y


class InceptionAux(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, conv_block: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = ConvBlock
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            conv_block(in_channels, 128, Kernel_size=1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(inplace=True),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ConvBlock(nn.Module):  # 通用的卷积块
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


if __name__ == "__main__":
    net = GoogleNet_(2, aux_logits=False, transform_input=False)
    net.eval()
    x = torch.randn(2, 3, 224, 224)
    out = net(x)
    y = out._asdict()['logits']
    import os
    import torchviz
    g = torchviz.make_dot(y, params=dict(list(net.named_parameters()) + [('x', x)]))
    if not os.path.isdir("./graph"):
        os.mkdir("./graph")
    g.view(os.path.join("./graph/googlenet.gv"))
