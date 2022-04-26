from typing import Any, Callable, Optional, Tuple, Union
from torch import nn
import torch


class FunctionalBase(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def build(
        self, layer: Callable,
        batch_norm_module: Any,
        batch_norm_size: int,
        activation: Optional[nn.Module] = None,
        use_batch_norm: bool = False,
        batch_norm_eps: float = 1e-3,
        batch_norm_momentum: float = 0.01,
    ) -> None:

        super().__init__()

        self.layer = layer
        self.body = nn.Identity()

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

    def forward(self, input, weights):

        x = self.layer(input, weights)
        result = self.body(x)
        return result


class FunctionalConvolution(FunctionalBase):

    def __init__(
        self,
        stride: Union[Tuple, int] = 1,
        activation: Optional[nn.Module] = None,
        use_batch_norm: bool = False,
        batch_norm_size: int = 0,
        batch_norm_eps: float = 1e-3,
        batch_norm_momentum: float = 0.01,
        **kwargs,
    ) -> None:

        super().__init__()

        layer = lambda inputs, weights: torch.nn.functional.conv2d(
            inputs,
            weights[0],
            None if weights[1].shape[0] == 0 else weights[1],
            stride=stride,
            **kwargs
        )
        super().build(
            layer,
            nn.BatchNorm2d,
            batch_norm_size,
            activation=activation,
            use_batch_norm=use_batch_norm,
            batch_norm_eps=batch_norm_eps,
            batch_norm_momentum=batch_norm_momentum,
        )


class FunctionalLinear(FunctionalBase):

    def __init__(
        self,
        activation: Optional[nn.Module] = None,
        use_batch_norm: bool = False,
        batch_norm_eps: float = 1e-3,
        batch_norm_size: int = 0,
        batch_norm_momentum: float = 0.01,
        **kwargs,
    ) -> None:

        super().__init__()

        layer = lambda inputs, weights: torch.nn.functional.linear(
            inputs,
            weights[0],
            None if weights[1].shape[0] == 0 else weights[1],
            **kwargs
        )

        super().build(
            layer,
            nn.BatchNorm1d,
            batch_norm_size,
            activation=activation,
            use_batch_norm=use_batch_norm,
            batch_norm_eps=batch_norm_eps,
            batch_norm_momentum=batch_norm_momentum,
        )


class FunctionalConvolutionTranspose(FunctionalBase):

    def __init__(
        self,
        stride: Union[Tuple, int] = 1,
        activation: Optional[nn.Module] = None,
        use_batch_norm: bool = False,
        batch_norm_eps: float = 1e-3,
        batch_norm_size: int = 0,
        batch_norm_momentum: float = 0.01,
        **kwargs,
    ) -> None:

        super().__init__()

        if use_batch_norm:
            bias = False

        body = lambda inputs, weights: torch.nn.functional.conv_transpose2d(
            inputs,
            weights[0],
            None if weights[1].shape[0] == 0 else weights[1],
            stride=stride,
            **kwargs
        )
        super().build(
            body,
            nn.BatchNorm2d,
            batch_norm_size,
            activation=activation,
            use_batch_norm=use_batch_norm,
            batch_norm_eps=batch_norm_eps,
            batch_norm_momentum=batch_norm_momentum,
        )
