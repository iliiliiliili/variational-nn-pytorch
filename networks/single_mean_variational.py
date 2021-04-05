from typing import Any, List, Optional, Literal, Tuple, Union
import torch
from torch import nn


class SingleMeanVariationalBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def build(
        self,
        stds: Any,
        batch_norm_module: Any,
        batch_norm_size: int,
        activation: Optional[Union[nn.Module, List[nn.Module]]] = None,
        activation_mode: Union[
            Literal["std"],
            Literal["end"],
            Literal["std+end"],
            Literal["std+end"],
        ] = "std",
        use_batch_norm: bool = False,
        batch_norm_mode: Union[
            Literal["mean"],
            Literal["std"],
            Literal["mean+std"],
            Literal["end"],
            Literal["mean+end"],
            Literal["std+end"],
            Literal["mean+std+end"],
        ] = "std",
        batch_norm_eps: float = 1e-3,
        batch_norm_momentum: float = 0.01,
    ) -> None:

        super().__init__()

        self.end_activation = None
        self.end_batch_norm = None

        self.mean = torch.nn.Parameter(torch.tensor(0.1), requires_grad=True,)
        self.stds = stds

        if use_batch_norm:

            batch_norm_targets = batch_norm_mode.split("+")

            for i, target in enumerate(batch_norm_targets):

                if target == "std":
                    if self.stds is not None:
                        self.stds = nn.Sequential(
                            self.stds,
                            batch_norm_module(
                                batch_norm_size,
                                eps=batch_norm_eps,
                                momentum=batch_norm_momentum,
                            ),
                        )
                elif target == "end":
                    self.end_batch_norm = batch_norm_module(
                        batch_norm_size,
                        eps=batch_norm_eps,
                        momentum=batch_norm_momentum,
                    )
                else:
                    raise ValueError("Unknown batch norm target: " + target)

        if activation is not None:

            activation_targets = activation_mode.split("+")

            for i, target in enumerate(activation_targets):

                if len(activation_targets) == 1:
                    current_activation: nn.Module = activation  # type: ignore
                else:
                    current_activation: nn.Module = activation[
                        i
                    ]  # type: ignore

                if target == "std":
                    if self.stds is not None:
                        self.stds = nn.Sequential(
                            self.stds, current_activation,
                        )
                elif target == "end":
                    self.end_activation = current_activation
                else:
                    raise ValueError("Unknown activation target: " + target)

    def forward(self, x):

        stds = self.stds(x)

        result = torch.distributions.Normal(self.mean, stds).rsample()

        if self.end_batch_norm is not None:
            result = self.end_batch_norm(result)

        if self.end_activation is not None:
            result = self.end_activation(result)

        return result


class SingleMeanVariationalConvolution(SingleMeanVariationalBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: Union[Tuple, int] = 1,
        activation: Optional[Union[nn.Module, List[nn.Module]]] = None,
        activation_mode: Union[
            Literal["std"],
            Literal["end"],
            Literal["std+end"],
            Literal["std+end"],
        ] = "std",
        use_batch_norm: bool = False,
        batch_norm_mode: Union[
            Literal["mean"],
            Literal["std"],
            Literal["mean+std"],
            Literal["end"],
            Literal["mean+end"],
            Literal["std+end"],
            Literal["mean+std+end"],
        ] = "std",
        batch_norm_eps: float = 1e-3,
        batch_norm_momentum: float = 0.01,
        bias=True,
        **kwargs,
    ) -> None:

        super().__init__()

        if use_batch_norm:
            bias = False

        stds = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
            **kwargs,
        )

        super().build(
            stds,
            nn.BatchNorm2d,
            out_channels,
            activation=activation,
            activation_mode=activation_mode,
            use_batch_norm=use_batch_norm,
            batch_norm_mode=batch_norm_mode,
            batch_norm_eps=batch_norm_eps,
            batch_norm_momentum=batch_norm_momentum,
        )


class SingleMeanVariationalLinear(SingleMeanVariationalBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Optional[Union[nn.Module, List[nn.Module]]] = None,
        activation_mode: Union[
            Literal["std"],
            Literal["end"],
            Literal["std+end"],
            Literal["std+end"],
        ] = "std",
        use_batch_norm: bool = False,
        batch_norm_mode: Union[
            Literal["mean"],
            Literal["std"],
            Literal["mean+std"],
            Literal["end"],
            Literal["mean+end"],
            Literal["std+end"],
            Literal["mean+std+end"],
        ] = "std",
        batch_norm_eps: float = 1e-3,
        batch_norm_momentum: float = 0.01,
        bias=True,
        **kwargs,
    ) -> None:

        super().__init__()

        if use_batch_norm:
            bias = False

        stds = nn.Linear(in_features, out_features, bias=bias, **kwargs)

        super().build(
            stds,
            nn.BatchNorm1d,
            out_features,
            activation=activation,
            activation_mode=activation_mode,
            use_batch_norm=use_batch_norm,
            batch_norm_mode=batch_norm_mode,
            batch_norm_eps=batch_norm_eps,
            batch_norm_momentum=batch_norm_momentum,
        )
