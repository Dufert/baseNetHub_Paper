import torch
import torch.nn as nn


class AlexNet_(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # 将每个data的feature map 压扁
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    net = AlexNet_(2)
    x = torch.randn(2, 3, 224, 224)
    y = net(x)
    import os
    import torchviz
    g = torchviz.make_dot(y, params=dict(list(net.named_parameters()) + [('x', x)]))
    if not os.path.isdir("./graph"):
        os.mkdir("./graph")
    g.view(os.path.join("./graph/alexnet.gv"))
