from typing import Any, Literal, Optional, Tuple, Union
from torch import nn


class DropoutBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def build(
        self,
        body: nn.Module,
        dropout_module: Any,
        activation: Optional[nn.Module] = None,
        dropout_probability: float = 0.05,
        dropout_inplace: bool = False,
    ) -> None:

        super().__init__()

        self.body = body
        self.body = nn.Sequential(
            self.body, dropout_module(dropout_probability, dropout_inplace,)
        )

        if activation is not None:

            self.body = nn.Sequential(self.body, activation,)

    def forward(self, x):

        result = self.body(x)
        return result


class DropoutConvolution(DropoutBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: Union[Tuple, int] = 1,
        activation: Optional[nn.Module] = None,
        dropout_probability: float = 0.05,
        dropout_inplace: bool = False,
        dropout_type: Literal[
            "alpha", "feature_alpha", "standart"
        ] = "standart",
        bias=True,
        **kwargs,
    ) -> None:

        super().__init__()

        body = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
            **kwargs,
        )

        droput = {
            "alpha": nn.AlphaDropout,
            "standart": nn.Dropout,
            "feature_alpha": nn.FeatureAlphaDropout
        }[dropout_type]

        super().build(
            body,
            droput,
            activation=activation,
            dropout_probability=dropout_probability,
            dropout_inplace=dropout_inplace,
        )


class DropoutLinear(DropoutBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Optional[nn.Module] = None,
        dropout_probability: float = 0.05,
        dropout_inplace: bool = False,
        dropout_type: Literal[
            "alpha", "feature_alpha", "standart"
        ] = "standart",
        bias=True,
        **kwargs,
    ) -> None:

        super().__init__()

        body = nn.Linear(in_features, out_features, bias=bias, **kwargs)

        droput = {
            "alpha": nn.AlphaDropout,
            "standart": nn.Dropout,
            "feature_alpha": nn.FeatureAlphaDropout
        }[dropout_type]

        nn.FeatureAlphaDropout

        super().build(
            body,
            droput,
            activation=activation,
            dropout_probability=dropout_probability,
            dropout_inplace=dropout_inplace,
        )


class DropoutConvolutionTranspose(DropoutBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: Union[Tuple, int] = 1,
        activation: Optional[nn.Module] = None,
        dropout_probability: float = 0.05,
        dropout_inplace: bool = False,
        dropout_type: Literal[
            "alpha", "feature_alpha", "standart"
        ] = "standart",
        bias=True,
        **kwargs,
    ) -> None:

        super().__init__()

        body = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
            **kwargs,
        )

        droput = {
            "alpha": nn.AlphaDropout,
            "standart": nn.Dropout,
            "feature_alpha": nn.FeatureAlphaDropout
        }[dropout_type]

        super().build(
            body,
            droput,
            activation=activation,
            dropout_probability=dropout_probability,
            dropout_inplace=dropout_inplace,
        )
