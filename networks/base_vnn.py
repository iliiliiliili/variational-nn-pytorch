from torch import nn
import torch
from networks.network import Network
from networks.variational import VariationalConvolution, VariationalLinear


class BaseVNN(Network):

    def __init__(self, **kwargs) -> None:

        super().__init__()

        self.model = nn.Sequential(
            VariationalConvolution(1, 256, 9, 1, **kwargs),
            VariationalConvolution(256, 256, 9, 2, **kwargs),
            VariationalConvolution(256, 16, 4, 1, **kwargs),
            torch.nn.Flatten(start_dim=1),  # type: ignore
            VariationalLinear(3 * 3 * 16, 10, **kwargs),
        )

    def forward(self, x):

        return self.model(x)
