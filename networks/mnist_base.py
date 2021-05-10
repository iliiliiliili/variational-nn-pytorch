from torch import nn
import torch
from networks.network import Network


def createMnistBase(Convolution, Linear):
    class MnistBase(Network):
        def __init__(self, **kwargs) -> None:

            super().__init__()

            self.model = nn.Sequential(
                Convolution(1, 256, 9, 1, **kwargs),
                Convolution(256, 256, 9, 2, **kwargs),
                Convolution(256, 16, 4, 1, **kwargs),
                torch.nn.Flatten(start_dim=1),  # type: ignore
                Linear(
                    3 * 3 * 16,
                    10,
                    uncertainty_placeholder=self,
                    **kwargs
                ),
            )

        def forward(self, x):

            return self.model(x)

    return MnistBase
