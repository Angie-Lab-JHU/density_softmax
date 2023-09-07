import torch
import torch.nn.functional as F
from torch import nn


class Lenet_5(nn.Module):
    def __init__(self):
        super(Lenet_5, self).__init__()
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )
        self.feature_extractor_2 = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.feature_extractor_2(x)
        return x