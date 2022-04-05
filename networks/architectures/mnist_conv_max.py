from torch import nn
import torch
from networks.network import Network


def createMnistConvMax(Convolution, Linear):
    class MnistConvMax(Network):

        def __init__(self, **kwargs) -> None:

            super().__init__()

            self.model = nn.Sequential(
                Convolution(1, 10, 5, 1, **kwargs),
                nn.MaxPool2d(2),
                Convolution(10, 20, 5, 1, **kwargs),
                nn.MaxPool2d(2),
                torch.nn.Flatten(start_dim=1),  # type: ignore
                Linear(320, 10, **kwargs),
                nn.LogSoftmax(1)
            )

        def forward(self, x):

            return self.model(x)

    return MnistConvMax
