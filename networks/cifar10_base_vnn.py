from torch import nn
import torch
from networks.network import Network
from networks.variational import VariationalConvolution, VariationalLinear


class Cifar10BaseVNN(Network):

    def __init__(self, **kwargs) -> None:

        super().__init__()

        self.model = nn.Sequential(
            VariationalConvolution(3, 32, 3, 1, **kwargs),
            VariationalConvolution(32, 32, 3, 2, **kwargs),
            VariationalConvolution(32, 64, 3, 1, **kwargs),
            VariationalConvolution(64, 64, 3, 2, **kwargs),
            VariationalConvolution(64, 128, 3, 1, **kwargs),
            VariationalConvolution(128, 128, 3, 2, **kwargs),
            torch.nn.Flatten(start_dim=1),  # type: ignore
            VariationalLinear(1 * 1 * 128, 10, **kwargs),
        )

    def forward(self, x):

        return self.model(x)
