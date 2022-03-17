from torch import nn
import torch
from networks.network import Network


def createCifar10Base(Convolution, Linear):
    class Cifar10Base(Network):

        def __init__(self, **kwargs) -> None:

            super().__init__()

            self.model = nn.Sequential(
                Convolution(3, 32, 3, 1, **kwargs),
                Convolution(32, 32, 3, 2, **kwargs),
                Convolution(32, 64, 3, 1, **kwargs),
                Convolution(64, 64, 3, 2, **kwargs),
                Convolution(64, 128, 3, 1, **kwargs),
                Convolution(128, 128, 3, 2, **kwargs),
                torch.nn.Flatten(start_dim=1),  # type: ignore
                Linear(1 * 1 * 128, 10, **kwargs),
            )

        def forward(self, x):

            return self.model(x)

    return Cifar10Base
