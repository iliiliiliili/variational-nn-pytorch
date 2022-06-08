from typing import Optional, Tuple, Union
from torch import nn
import torch
from networks.functional import FunctionalConvolution, FunctionalConvolutionTranspose, FunctionalLinear
from networks.network import Network
import numpy as np


def get_shapes(network: Network):
    shapes = [a.parameter_shapes for a in network.modules() if hasattr(a, "parameter_shapes")]
    return shapes


class HypermodelNetwork(Network):
    def __init__(self, base, hypertorso_creator, index_dim, index_scale=0, scale_weights=False, shape_index=False) -> None:
        super().__init__()
        self.base = base

        self.intialize_shapes()
        self.index_dim = self.total_size if index_dim is None else index_dim
        self.index_scale = index_scale
        self.scale_weights = scale_weights
        self.shape_index = shape_index

        self.hypertorso = hypertorso_creator(index_dim, self.total_size, self.shapes)

    def intialize_shapes(self):

        shapes = get_shapes(self.base)
        flat_shapes = [[np.prod(x) for x in sub_shapes] for sub_shapes in shapes]
        block_sizes = [np.sum(x) for x in flat_shapes]
        total_size = np.sum(block_sizes)

        self.total_size = total_size
        self.shapes = shapes
        self.flat_shapes = flat_shapes

    def forward(self, x):

        weights = self.create_hyperweights(x.device)
        for weight, module in zip(weights, filter(lambda a: hasattr(a, "parameter_shapes"), self.base.modules())):
            module.apply_weight(weight)

        result = self.base(x)
        return result
    
    def create_hyperweights(self, device):
        def singe_prior_index(shape, index_scale=0):

            if shape[0] == 0:
                return torch.empty([], device=device)

            result = torch.normal(0, torch.ones(shape, device=device))

            if index_scale > 0:
                result *= index_scale / np.sqrt(np.prod(shape))
            
            return result

        def single_scale(w):
            return w / np.sqrt(np.prod(w.shape))

        if self.shape_index:
            index = [[singe_prior_index(x, self.index_scale) for x in sub_shapes] for sub_shapes in self.shapes]
        else:
            index = singe_prior_index([self.index_dim])

        weights = self.hypertorso(index)

        if self.scale_weights:
            weights = [[single_scale(w) if i == 0 else w for i, w in enumerate(sub_weights)] for sub_weights in weights]

        return weights


def create_hypermodel(network_creator, hypertorso_creator, index_dim, index_scale, scale_weights, shape_index, **kwargs):
    base = network_creator(**kwargs)

    result = HypermodelNetwork(base, hypertorso_creator, index_dim, index_scale, scale_weights, shape_index)

    return result


def create_linear_hypermodel(network_creator, index_dim, **kwargs):
    hypertorso_creator = lambda index_dim, total_parameters, shapes: LinearHypertorso(index_dim, shapes)

    result = create_hypermodel(network_creator, hypertorso_creator, index_dim, scale_weights=False, shape_index=False, **kwargs)

    return result


def create_bbb_hypermodel(network_creator, index_scale=1, **kwargs):
    hypertorso_creator = lambda _, total_parameters, shapes: DiagonalLinearHypertorso(total_parameters, shapes)

    result = create_hypermodel(network_creator, hypertorso_creator, None, index_scale=index_scale, scale_weights=False, shape_index=True, **kwargs)

    return result


class HypermodelConvolution(FunctionalConvolution):

    def __init__(
        self,
        in_channels: int, out_channels: int,
        kernel_size: int, stride: Union[Tuple, int] = 1,
        activation: Optional[nn.Module] = None,
        use_batch_norm: bool = False,
        batch_norm_eps: float = 1e-3,
        batch_norm_momentum: float = 0.01,
        bias=True,
        groups=1,
        **kwargs,
    ) -> None:

        super().__init__(
            activation=activation,
            stride=stride,
            use_batch_norm=use_batch_norm,
            batch_norm_eps=batch_norm_eps,
            batch_norm_size=out_channels,
            batch_norm_momentum=batch_norm_momentum,
            groups=groups,
            **kwargs,
        )
        
        if use_batch_norm:
            bias = False

        if isinstance(kernel_size, int):
            kernel = (kernel_size, kernel_size)
        else:
            kernel = kernel_size

        weights_shape = [out_channels, in_channels // groups, kernel[0], kernel[1]]

        if bias:
            bias_shape = [out_channels]
        else:
            bias_shape = [0]
        
        self.parameter_shapes = [weights_shape, bias_shape]
        self.weigths = None
    
    def forward(self, x, weights=None):
        
        if weights is None:
            weights = self.weigths
        
        result = super().forward(x, weights)
        return result
    
    def apply_weight(self, weights):
        self.weigths = weights


class HypermodelLinear(FunctionalLinear):

    def __init__(
        self,
        in_features: int, out_features: int,
        activation: Optional[nn.Module] = None,
        use_batch_norm: bool = False,
        batch_norm_eps: float = 1e-3,
        batch_norm_momentum: float = 0.01,
        bias=True,
        **kwargs,
    ) -> None:

        super().__init__(
            activation=activation,
            use_batch_norm=use_batch_norm,
            batch_norm_eps=batch_norm_eps,
            batch_norm_size=out_features,
            batch_norm_momentum=batch_norm_momentum,
            **kwargs,
        )
        
        if use_batch_norm:
            bias = False

        weights_shape = [out_features, in_features]

        if bias:
            bias_shape = [out_features]
        else:
            bias_shape = [0]
        
        self.parameter_shapes = [weights_shape, bias_shape]

    def forward(self, x, weights=None):
        
        if weights is None:
            weights = self.weigths
        
        result = super().forward(x, weights)
        return result
    
    def apply_weight(self, weights):
        self.weigths = weights


class HypermodelConvolutionTranspose(FunctionalConvolutionTranspose):

    def __init__(
        self,
        in_channels: int, out_channels: int,
        kernel_size: int, stride: Union[Tuple, int] = 1,
        activation: Optional[nn.Module] = None,
        use_batch_norm: bool = False,
        batch_norm_eps: float = 1e-3,
        batch_norm_momentum: float = 0.01,
        groups=1,
        bias=True,
            **kwargs,
    ) -> None:

        super().__init__(
            activation=activation,
            stride=stride,
            use_batch_norm=use_batch_norm,
            batch_norm_eps=batch_norm_eps,
            batch_norm_size=out_channels,
            batch_norm_momentum=batch_norm_momentum,
            groups=groups,
            **kwargs,
        )

        if use_batch_norm:
            bias = False

        if isinstance(kernel_size, int):
            kernel = (kernel_size, kernel_size)
        else:
            kernel = kernel_size

        weights_shape = [in_channels, out_channels // groups, kernel[0], kernel[1]]

        if bias:
            bias_shape = [out_channels]
        else:
            bias_shape = [0]
            
        self.parameter_shapes = [weights_shape, bias_shape]

    def forward(self, x, weights=None):
        
        if weights is None:
            weights = self.weigths
        
        result = super().forward(x, weights)
        return result
    
    def apply_weight(self, weights):
        self.weigths = weights


class DiagonalLinearHypertorso(nn.Module):
    def __init__(self, input_size, shapes, use_bias=True):
        super().__init__()

        ws = nn.ModuleList([nn.ParameterList([nn.parameter.Parameter(torch.empty(x)) for x in sub_shapes]) for sub_shapes in shapes])
        stds = [[1.0 / np.sqrt(np.prod(x)) for x in sub_shapes] for sub_shapes in shapes]

        for sub_stds, sub_ws in zip(stds, ws):
            for std, w in zip(sub_stds, sub_ws):
                torch.nn.init.trunc_normal_(w, std=std)

        self.ws = ws
        
        if use_bias:
            bs = nn.ModuleList([nn.ParameterList([nn.parameter.Parameter(torch.zeros(x)) for x in sub_shapes]) for sub_shapes in shapes])
            self.bs = bs
        
        self.use_bias = use_bias
        self.shapes = shapes
        self.flat_shapes = [[np.prod(x) for x in sub_shapes] for sub_shapes in shapes]

    def forward(self, shaped_x):

        # shaped_x = split_by_arrays(x, self.flat_shapes)
        # result = [[w for x, w in zip(layer_xs, layer_ws)] for layer_xs, layer_ws in zip(shaped_x, self.ws)]
        # result = [[w * x.reshape(w.shape) for x, w in zip(layer_xs, layer_ws)] for layer_xs, layer_ws in zip(shaped_x, self.ws)]
        result = [[torch.log(1 + torch.exp(w)) * x.reshape(w.shape if w.shape[0] != 0 else x.shape) for x, w in zip(layer_xs, layer_ws)] for layer_xs, layer_ws in zip(shaped_x, self.ws)]

        if self.use_bias:
            result = [[h + b for h, b in zip(layer_result, layer_bs)] for layer_result, layer_bs in zip(result, self.bs)]

        return result


class LinearHypertorso(nn.Module):
    def __init__(self, index_dim, shapes, use_bias=True):
        super().__init__()

        self.hyper_layer = nn.Linear(index_dim, index_dim)
        self.layers = nn.ModuleList([nn.ModuleList([nn.Linear(index_dim, np.prod(x)) for x in sub_shapes]) for sub_shapes in shapes])
        self.shapes = shapes
    
    def forward(self, x):
        hyper_index = self.hyper_layer(x)

        # hyper_index = torch.ones_like(x)

        result = [[l(hyper_index).reshape(s) for s, l in zip(layer_shapes, layer_layers)] for layer_shapes, layer_layers in zip(self.shapes, self.layers)]

        return result