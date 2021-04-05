from torch import nn
import torch
from networks.network import Network


def createCifar10MiniBase(Convolution, Linear):
    class Cifar10MiniBase(Network):

        def __init__(self, **kwargs) -> None:

            super().__init__()

            self.model = nn.Sequential(
                Convolution(3, 8, 3, 1, **kwargs),
                Convolution(8, 8, 3, 2, **kwargs),
                Convolution(8, 16, 3, 1, **kwargs),
                Convolution(16, 16, 3, 2, **kwargs),
                Convolution(16, 32, 3, 1, **kwargs),
                Convolution(32, 32, 3, 2, **kwargs),
                torch.nn.Flatten(start_dim=1),  # type: ignore
                Linear(1 * 1 * 32, 10, **kwargs),
            )

        def forward(self, x):

            return self.model(x)

    return Cifar10MiniBase
