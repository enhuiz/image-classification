from torch import nn


class ToyDilatedModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, dilation=2, padding=2),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, dilation=4, padding=4),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, dilation=8, padding=8),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.seq(x)
