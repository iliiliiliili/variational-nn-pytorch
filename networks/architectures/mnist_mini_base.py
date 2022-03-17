from torch import nn
import torch
from networks.network import Network


def createMnistMiniBase(Convolution, Linear):
    class MnistMiniBase(Network):

        def __init__(self, **kwargs) -> None:

            super().__init__()

            self.model = nn.Sequential(
                Convolution(1, 32, 9, 1, **kwargs),
                Convolution(32, 32, 9, 2, **kwargs),
                Convolution(32, 16, 4, 1, **kwargs),
                torch.nn.Flatten(start_dim=1),  # type: ignore
                Linear(3 * 3 * 16, 10, **kwargs),
            )

        def forward(self, x):

            return self.model(x)

    return MnistMiniBase
