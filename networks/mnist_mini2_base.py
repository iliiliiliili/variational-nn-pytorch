from torch import nn
import torch
from networks.network import Network


def createMnistMini2Base(Convolution, Linear):
    class MnistMini2Base(Network):

        def __init__(self, **kwargs) -> None:

            super().__init__()

            self.model = nn.Sequential(
                Convolution(1, 8, 9, 1, **kwargs),
                Convolution(8, 16, 9, 2, **kwargs),
                Convolution(16, 4, 4, 1, **kwargs),
                torch.nn.Flatten(start_dim=1),  # type: ignore
                Linear(3 * 3 * 4, 10, **kwargs),
            )

        def forward(self, x):

            return self.model(x)

    return MnistMini2Base
