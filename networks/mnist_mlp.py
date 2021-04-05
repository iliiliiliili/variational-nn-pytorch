from torch import nn
import torch
from networks.network import Network


def createMnistMlp(Convolution, Linear):
    class MnistMlp(Network):

        def __init__(self, **kwargs) -> None:

            super().__init__()

            self.model = nn.Sequential(
                torch.nn.Flatten(start_dim=1),  # type: ignore
                Linear(28 * 28, 50, **kwargs),
                Linear(50, 50, **kwargs),
                Linear(50, 10, **kwargs),
                nn.LogSoftmax(1)
            )

        def forward(self, x):

            return self.model(x)

    return MnistMlp
