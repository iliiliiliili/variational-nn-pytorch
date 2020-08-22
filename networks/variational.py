from typing import Any, List, Optional, Literal, Tuple, Union
import torch
from torch import nn


class VariationalBase(nn.Module):

    GLOBAL_STD: float = 0

    def __init__(self) -> None:
        super().__init__()

    def build(
        self, means: nn.Module, stds: Any,
        batch_norm_module: Any,
        batch_norm_size: int,
        activation: Optional[Union[nn.Module, List[nn.Module]]] = None,
        activation_mode: Union[
            Literal['mean'], Literal['std'],
            Literal['mean+std'], Literal['end'],
            Literal['mean+end'], Literal['std+end'],
            Literal['mean+std+end']
        ] = 'mean',
        use_batch_norm: bool = False,
        batch_norm_mode: Union[
            Literal['mean'], Literal['std'],
            Literal['mean+std'], Literal['end'],
            Literal['mean+end'], Literal['std+end'],
            Literal['mean+std+end']
        ] = 'mean',
        batch_norm_eps: float = 1e-3,
        batch_norm_momentum: float = 0.01,
        global_std_mode: Union[
            Literal['none'], Literal['replace'],
            Literal['multiply']
        ] = 'none'
    ) -> None:

        super().__init__()

        self.end_activation = None
        self.end_batch_norm = None

        self.means = means
        self.stds = stds

        self.global_std_mode = global_std_mode

        if use_batch_norm:

            batch_norm_targets = batch_norm_mode.split('+')

            for i, target in enumerate(batch_norm_targets):

                if target == 'mean':
                    self.means = nn.Sequential(
                        self.means,
                        batch_norm_module(
                            batch_norm_size,
                            eps=batch_norm_eps,
                            momentum=batch_norm_momentum,
                        )
                    )
                elif target == 'std':
                    if self.stds is not None:
                        self.stds = nn.Sequential(
                            self.stds,
                            batch_norm_module(
                                batch_norm_size,
                                eps=batch_norm_eps,
                                momentum=batch_norm_momentum,
                            )
                        )
                elif target == 'end':
                    self.end_batch_norm = batch_norm_module(
                        batch_norm_size,
                        eps=batch_norm_eps,
                        momentum=batch_norm_momentum,
                    )
                else:
                    raise ValueError('Unknown batch norm target: ' + target)

        if activation is not None:

            activation_targets = activation_mode.split('+')

            for i, target in enumerate(activation_targets):

                if len(activation_targets) == 1:
                    current_activation: nn.Module = activation  # type: ignore
                else:
                    current_activation: nn.Module = \
                        activation[i]  # type: ignore

                if target == 'mean':
                    self.means = nn.Sequential(
                        self.means,
                        current_activation,
                    )
                elif target == 'std':
                    if self.stds is not None:
                        self.stds = nn.Sequential(
                            self.stds,
                            current_activation,
                        )
                elif target == 'end':
                    self.end_activation = current_activation
                else:
                    raise ValueError('Unknown activation target: ' + target)

    def forward(self, x):

        means = self.means(x)

        if self.stds:
            stds = self.stds(x)
        else:
            stds = 0

        if self.global_std_mode == 'replace':
            stds = VariationalBase.GLOBAL_STD
        elif self.global_std_mode == 'multiply':
            stds = VariationalBase.GLOBAL_STD * stds

        result = torch.distributions.Normal(means, stds).rsample()

        if self.end_batch_norm is not None:
            result = self.end_batch_norm(result)

        if self.end_activation is not None:
            result = self.end_activation(result)

        return result


class VariationalConvolution(VariationalBase):

    def __init__(
        self, in_channels: int, out_channels: int,
        kernel_size: int, stride: Union[Tuple, int] = 1,
        activation: Optional[Union[nn.Module, List[nn.Module]]] = None,
        activation_mode: Union[
            Literal['mean'], Literal['std'],
            Literal['mean+std'], Literal['end'],
            Literal['mean+end'], Literal['std+end'],
            Literal['mean+std+end']
        ] = 'mean',
        use_batch_norm: bool = False,
        batch_norm_mode: Union[
            Literal['mean'], Literal['std'],
            Literal['mean+std'], Literal['end'],
            Literal['mean+end'], Literal['std+end'],
            Literal['mean+std+end']
        ] = 'end',
        batch_norm_eps: float = 1e-3,
        batch_norm_momentum: float = 0.01,
        global_std_mode: Union[
            Literal['none'], Literal['replace'],
            Literal['multiply']
        ] = 'none',
        bias=True,
        **kwargs,
    ) -> None:

        super().__init__()

        if use_batch_norm:
            bias = False

        means = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
            **kwargs
        )

        if global_std_mode == 'replace':
            stds = None
        else:
            stds = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=bias,
                **kwargs
            )

        super().build(
            means, stds,
            nn.BatchNorm2d,
            out_channels,
            activation=activation,
            activation_mode=activation_mode,
            use_batch_norm=use_batch_norm,
            batch_norm_mode=batch_norm_mode,
            batch_norm_eps=batch_norm_eps,
            batch_norm_momentum=batch_norm_momentum,
            global_std_mode=global_std_mode,
        )


class VariationalLinear(VariationalBase):

    def __init__(
        self, in_features: int, out_features: int,
        activation: Optional[Union[nn.Module, List[nn.Module]]] = None,
        activation_mode: Union[
            Literal['mean'], Literal['std'],
            Literal['mean+std'], Literal['end'],
            Literal['mean+end'], Literal['std+end'],
            Literal['mean+std+end']
        ] = 'mean',
        use_batch_norm: bool = False,
        batch_norm_mode: Union[
            Literal['mean'], Literal['std'],
            Literal['mean+std'], Literal['end'],
            Literal['mean+end'], Literal['std+end'],
            Literal['mean+std+end']
        ] = 'mean',
        batch_norm_eps: float = 1e-3,
        batch_norm_momentum: float = 0.01,
        global_std_mode: Union[
            Literal['none'], Literal['replace'],
            Literal['multiply']
        ] = 'none',
        bias=True,
        **kwargs,
    ) -> None:

        super().__init__()

        if use_batch_norm:
            bias = False

        means = nn.Linear(
            in_features, out_features,
            bias=bias,
            **kwargs
        )

        if global_std_mode == 'replace':
            stds = None
        else:
            stds = nn.Linear(
                in_features, out_features,
                bias=bias,
                **kwargs
            )

        super().build(
            means, stds,
            nn.BatchNorm1d,
            out_features,
            activation=activation,
            activation_mode=activation_mode,
            use_batch_norm=use_batch_norm,
            batch_norm_mode=batch_norm_mode,
            batch_norm_eps=batch_norm_eps,
            batch_norm_momentum=batch_norm_momentum,
            global_std_mode=global_std_mode,
        )


class VariationalConvolutionTranspose(VariationalBase):

    def __init__(
        self, in_channels: int, out_channels: int,
        kernel_size: int, stride: Union[Tuple, int] = 1,
        activation: Optional[Union[nn.Module, List[nn.Module]]] = None,
        activation_mode: Union[
            Literal['mean'], Literal['std'],
            Literal['mean+std'], Literal['end'],
            Literal['mean+end'], Literal['std+end'],
            Literal['mean+std+end']
        ] = 'mean',
        use_batch_norm: bool = False,
        batch_norm_mode: Union[
            Literal['mean'], Literal['std'],
            Literal['mean+std'], Literal['end'],
            Literal['mean+end'], Literal['std+end'],
            Literal['mean+std+end']
        ] = 'end',
        batch_norm_eps: float = 1e-3,
        batch_norm_momentum: float = 0.01,
        global_std_mode: Union[
            Literal['none'], Literal['replace'],
            Literal['multiply']
        ] = 'none',
        bias=True,
        **kwargs,
    ) -> None:

        super().__init__()

        if use_batch_norm:
            bias = False

        means = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
            **kwargs
        )

        if global_std_mode == 'replace':
            stds = None
        else:
            stds = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=bias,
                **kwargs
            )

        super().build(
            means, stds,
            nn.BatchNorm2d,
            out_channels,
            activation=activation,
            activation_mode=activation_mode,
            use_batch_norm=use_batch_norm,
            batch_norm_mode=batch_norm_mode,
            batch_norm_eps=batch_norm_eps,
            batch_norm_momentum=batch_norm_momentum,
            global_std_mode=global_std_mode,
        )
