import torch
import torch.nn as nn
# 参考了：https://github.com/pochih/FCN-pytorch/blob/master/python/fcn.py，如需实现其他fcn，则需要使用分开讨论不使用nn.Sequential


class Fcn_32(nn.Module):
    def __init__(self, features: nn.Module, num_classes: int = 10) -> None:
        super().__init__()
        self.features = features
        self.fcn_1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        )
        self.bn_1 = nn.BatchNorm2d(512)

        self.fcn_2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
        )
        self.bn_2 = nn.BatchNorm2d(256)
        self.fcn_layers = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.fcn_layers(x)
        return x


if __name__ == "__main__":
    from vgg_ import vgg
    num_classes = 10
    net_feature = vgg('A', True, num_classes)
    net = Fcn_32(net_feature.features, num_classes=num_classes)
    x = torch.rand(2, 3, 224, 224)
    y = net(x)
    import os
    import torchviz
    if not os.path.isdir("./graph"):
        os.mkdir("./graph")
    g = torchviz.make_dot(y, params=dict(list(net.named_parameters()) + [('x', x)]))
    g.view("./graph/fcn_32.gv")
