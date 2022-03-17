from core import filter_out_dict
from torch import nn
import torch
from networks.architectures.auto_encoder_network import AutoEncoderNetwork


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


def createMnistAutoEncoderBase(Convolution, Linear):
    class MnistAutoEncoderBase(AutoEncoderNetwork):
        def __init__(self, input_size=(28, 28), **kwargs) -> None:

            super().__init__()

            total_input_size = 1
            for a in input_size:
                total_input_size *= a

            encoder = nn.Sequential(
                torch.nn.Flatten(start_dim=1),  # type: ignore
                Linear(total_input_size, 400, **kwargs),
                Linear(400, 20, **kwargs),
            )

            decoder = nn.Sequential(
                Linear(20, 400, **kwargs),
                Linear(
                    400,
                    total_input_size,
                    activation=None,
                    **filter_out_dict(kwargs, ["activation"])
                ),
                torch.nn.Sigmoid(),
                Reshape(-1, 1, *input_size),
            )

            super().build(encoder, decoder, [20])

        def forward(self, x):

            return self.decoder(self.encoder(x))

    return MnistAutoEncoderBase
