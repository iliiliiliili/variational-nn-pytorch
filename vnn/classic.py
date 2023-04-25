from typing import Any, Optional, Tuple, Union
from torch import nn


class ClassicBase(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def build(
        self, body: nn.Module,
        batch_norm_module: Any,
        batch_norm_size: int,
        activation: Optional[nn.Module] = None,
        use_batch_norm: bool = False,
        batch_norm_eps: float = 1e-3,
        batch_norm_momentum: float = 0.01,
    ) -> None:

        super().__init__()

        self.body = body

        if use_batch_norm:

            self.body = nn.Sequential(
                self.body,
                batch_norm_module(
                    batch_norm_size,
                    eps=batch_norm_eps,
                    momentum=batch_norm_momentum,
                )
            )

        if activation is not None:

            self.body = nn.Sequential(
                self.body,
                activation,
            )

    def forward(self, x):

        result = self.body(x)
        return result


class ClassicConvolution(ClassicBase):

    def __init__(
        self, in_channels: int, out_channels: int,
        kernel_size: int, stride: Union[Tuple, int] = 1,
        activation: Optional[nn.Module] = None,
        use_batch_norm: bool = False,
        batch_norm_eps: float = 1e-3,
        batch_norm_momentum: float = 0.01,
        bias=True,
        **kwargs,
    ) -> None:

        super().__init__()

        if use_batch_norm:
            bias = False

        body = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
            **kwargs
        )
        super().build(
            body,
            nn.BatchNorm2d,
            out_channels,
            activation=activation,
            use_batch_norm=use_batch_norm,
            batch_norm_eps=batch_norm_eps,
            batch_norm_momentum=batch_norm_momentum,
        )


class ClassicLinear(ClassicBase):

    def __init__(
        self, in_features: int, out_features: int,
        activation: Optional[nn.Module] = None,
        use_batch_norm: bool = False,
        batch_norm_eps: float = 1e-3,
        batch_norm_momentum: float = 0.01,
        bias=True,
        **kwargs,
    ) -> None:

        super().__init__()

        if use_batch_norm:
            bias = False

        body = nn.Linear(
            in_features, out_features,
            bias=bias,
            **kwargs
        )

        super().build(
            body,
            nn.BatchNorm1d,
            out_features,
            activation=activation,
            use_batch_norm=use_batch_norm,
            batch_norm_eps=batch_norm_eps,
            batch_norm_momentum=batch_norm_momentum,
        )


class ClassicConvolutionTranspose(ClassicBase):

    def __init__(
        self, in_channels: int, out_channels: int,
        kernel_size: int, stride: Union[Tuple, int] = 1,
        activation: Optional[nn.Module] = None,
        use_batch_norm: bool = False,
        batch_norm_eps: float = 1e-3,
        batch_norm_momentum: float = 0.01,
        bias=True,
        **kwargs,
    ) -> None:

        super().__init__()

        if use_batch_norm:
            bias = False

        body = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
            **kwargs
        )
        super().build(
            body,
            nn.BatchNorm2d,
            out_channels,
            activation=activation,
            use_batch_norm=use_batch_norm,
            batch_norm_eps=batch_norm_eps,
            batch_norm_momentum=batch_norm_momentum,
        )
