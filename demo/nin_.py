import torch
import torch.nn as nn


class NiN_(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 160, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(160, 96, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # ceil_mode 天花板模式，在不足kernel_size时，仍然输出一个结果
            nn.Dropout(inplace=True),

            nn.Conv2d(96, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.Dropout(inplace=True),

            nn.Conv2d(192, 192, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, self.num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=8, stride=1)
        )
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), self.num_classes, -1)  # 将输出对应成类，不使用fc
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()


if __name__ == "__main__":
    net = NiN_(2)
    x = torch.rand(1, 3, 224, 224)
    y = net(x)
    import os
    import torchviz
    g = torchviz.make_dot(y, params=dict(list(net.named_parameters()) + [('x', x)]))
    if not os.path.isdir("./graph"):
        os.mkdir("./graph")
    g.view("./graph/nin.gv")
