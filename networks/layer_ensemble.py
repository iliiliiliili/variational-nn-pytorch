from typing import Any, List, Optional, Literal, Tuple, Union
from torch import nn
import torch
import numpy as np
from core import Flatten
from time import time

from networks.network import Network


class LayerEnsembleBase(nn.Module):

    __COLLECTION = []

    def __init__(self) -> None:
        super().__init__()
        LayerEnsembleBase.__COLLECTION.append(self)

    def build(
        self,
        ensembles: nn.Module,
        batch_norm_module: Any,
        batch_norm_size: int,
        activation: Optional[nn.Module] = None,
        use_batch_norm: bool = False,
        batch_norm_eps: float = 1e-3,
        batch_norm_momentum: float = 0.01,
    ) -> None:

        super().__init__()

        self.ensembles = ensembles
        self.num_ensemble = len(ensembles)
        self.suffix = nn.Identity()
        self.sample = None

        if use_batch_norm:
            self.suffix = nn.Sequential(
                self.suffix,
                batch_norm_module(
                    batch_norm_size,
                    eps=batch_norm_eps,
                    momentum=batch_norm_momentum,
                ),
            )

        if activation is not None:
            self.suffix = nn.Sequential(self.suffix, activation)

    def forward(self, input, sample=None):

        if sample is None:
            sample = self.sample

        if sample is None:
            raise Exception("No sample provided")

        result = self.ensembles[sample](input)
        result = self.suffix(result)

        self.sample = None
        return result

    @staticmethod
    def collect():
        result = LayerEnsembleBase.__COLLECTION
        LayerEnsembleBase.__COLLECTION = []

        return result

class LayerEnsembleConvolution(LayerEnsembleBase):
    def __init__(
        self,
        num_ensemble: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: Union[Tuple, int] = 1,
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

        modules = nn.ModuleList([nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
            **kwargs,
        ) for _ in range(num_ensemble)])

        super().build(
            modules,
            nn.BatchNorm2d,
            out_channels,
            activation=activation,
            use_batch_norm=use_batch_norm,
            batch_norm_eps=batch_norm_eps,
            batch_norm_momentum=batch_norm_momentum,
        )


class LayerEnsembleLinear(LayerEnsembleBase):
    def __init__(
        self,
        num_ensemble: int,
        in_features: int,
        out_features: int,
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

        modules = nn.ModuleList([nn.Linear(in_features, out_features, bias=bias, **kwargs) for _ in range(num_ensemble)])

        super().build(
            modules,
            nn.BatchNorm1d,
            out_features,
            activation=activation,
            use_batch_norm=use_batch_norm,
            batch_norm_eps=batch_norm_eps,
            batch_norm_momentum=batch_norm_momentum,
        )


class LayerEnsembleConvolutionTranspose(LayerEnsembleBase):
    def __init__(
        self,
        num_ensemble: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: Union[Tuple, int] = 1,
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

        modules = nn.ModuleList([nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
            **kwargs,
        ) for _ in range(num_ensemble)])

        super().build(
            modules,
            nn.BatchNorm2d,
            out_channels,
            activation=activation,
            use_batch_norm=use_batch_norm,
            batch_norm_eps=batch_norm_eps,
            batch_norm_momentum=batch_norm_momentum,
        )


class LayerEnsembleNetwork(Network):
    def __init__(self, network_creator, prior_scale, **kwargs) -> None:
        super().__init__()
        self.prior_scale = prior_scale

        empty = LayerEnsembleBase.collect()
        assert len(empty) == 0

        self.network = network_creator(**kwargs)
        self.network_collection = LayerEnsembleBase.collect()

        self.prior = network_creator(**kwargs)
        self.prior.requires_grad_(False)
        self.prior_collection = LayerEnsembleBase.collect()

        self.num_ensembles = [a.num_ensemble for a in self.network_collection]

    def forward(self, x, samples=10):
        sampled_models = self.sampler(samples)
        result = self.batched(x, sampled_models)

        return result

    def batched(self, x, samples):

        def output_with_prior(network, prior, sample):
            self.select_sample(self.network_collection, sample)
            result = network(x)
            if self.prior_scale > 0:
                self.select_sample(self.prior_collection, sample)
                return result + prior(x) * self.prior_scale

            return result

        result = torch.mean(torch.stack([output_with_prior(self.network, self.prior, sample) for sample in samples]), dim=0)
        return result

    def select_sample(self, collection, sample):
        assert len(sample) == len(self.num_ensembles)

        for i, layer in enumerate(collection):
            layer_sample = sample[i]
            assert layer_sample < layer.num_ensemble

            layer.sample = layer_sample

    def sampler(self, num_samples=None):

        def create_all_samples(i, num_ensembles, prefix):
            result = []
            for q in range(num_ensembles[i]):
                value = (*prefix, q)

                if i + 1 < len(num_ensembles):
                    result += create_all_samples(i + 1, num_ensembles, value)
                else:
                    result.append(value)

            return result

        all_samples = np.array(create_all_samples(0, self.num_ensembles, []))

        if num_samples is None:
            sorted_results = all_samples
        else:
            indices = np.random.choice(len(all_samples), num_samples, replace=False)
            results = all_samples[indices]

            lex_results = [results[:, results.shape[-1] - 1 - i] for i in range(results.shape[-1])]
            sorted_results = results[np.lexsort(lex_results)]

        return sorted_results


def create_layer_ensemble_network(network_creator, num_ensemble=2, prior_scale=1, **kwargs):

    creator = network_creator(
        specific_lens_layer(num_ensemble, LayerEnsembleConvolution),
        specific_lens_layer(num_ensemble, LayerEnsembleLinear)
    )
    result = LayerEnsembleNetwork(creator, prior_scale, **kwargs)

    return result


def specific_lens_layer(num_ensemble, Layer: LayerEnsembleBase):
    return lambda *args, **kwargs: Layer(num_ensemble, *args, **kwargs)

class SimpleLayerEnsembleNetwork(Network):
    def __init__(self, num_ensemble, optimized, **kwargs) -> None:
        super().__init__()

        num_ensembles = [num_ensemble, num_ensemble, num_ensemble, 0, num_ensemble]

        layers = nn.ModuleList([
            LayerEnsembleConvolution(num_ensemble, 1, 256, 9, 1, **kwargs),
            LayerEnsembleConvolution(num_ensemble, 256, 256, 9, 2, **kwargs),
            LayerEnsembleConvolution(num_ensemble, 256, 16, 4, 1, **kwargs),
            Flatten(start_dim=1),  # type: ignore
            LayerEnsembleLinear(
                num_ensemble,
                3 * 3 * 16,
                10,
                **kwargs
            ),
        ])

        self.num_ensembles = num_ensembles
        self.layers = layers
        self.optimized = optimized
        self.times = []

    def forward(self, x):

        t = time()

        samples = self.sampler()

        if self.optimized:
            result = self.optimized_forward(x, samples)
        else:
            result = self.unoptimized_forward(x, samples)

        t = time() - t
        self.times.append(t)

        return result

    def optimized_forward(self, x, samples):

        num_layers = len(self.num_ensembles)

        def ole(i, x, sub_samples):
            results = []

            if i == num_layers:
                return [x]

            if self.num_ensembles[i] == 0:
                out = self.layers[i](x)
                new_sub_samples = [s[1:] for s in sub_samples]
                return ole(i + 1, out, new_sub_samples)

            layer_indices = [s[0] for s in sub_samples]
            last_index = None
            last_out = None
            last_sub_samples = []
            all_last_sub_samples = []

            for q, index in enumerate(layer_indices):
                if last_index is None:
                    last_index = index
                    last_out = self.layers[i].ensembles[last_index](x)
                elif index != last_index:
                    results += ole(i + 1, last_out, last_sub_samples)
                    last_index = index
                    last_out = self.layers[i].ensembles[last_index](x)

                    all_last_sub_samples.append(last_sub_samples)
                    last_sub_samples = []

                last_sub_samples.append(sub_samples[q][1:])

            if last_index is not None:
                results += ole(i + 1, last_out, last_sub_samples)
                all_last_sub_samples.append(last_sub_samples)
            
            assert len(results) == len(sub_samples)
            return results

        outs = ole(0, x, samples)
        assert len(outs) == len(samples)
        result = torch.mean(torch.stack(outs), dim=0)
        return result

    def unoptimized_forward(self, x, samples):

        outs = []

        for sample in samples:
            val = x

            for i, q in enumerate(sample):

                if self.num_ensembles[i] == 0:
                    val = self.layers[i](val)
                else:
                    val = self.layers[i].ensembles[q](val)
                
            outs.append(val)

        assert len(outs) == len(samples)
        result = torch.mean(torch.stack(outs), dim=0)
        return result

    def sampler(self):

        def create_all_samples(i, num_ensembles, prefix):
            result = []
            for q in range(max(1, num_ensembles[i])):
                value = (*prefix, q)

                if i + 1 < len(num_ensembles):
                    result += create_all_samples(i + 1, num_ensembles, value)
                else:
                    result.append(value)

            return result

        all_samples = np.array(create_all_samples(0, self.num_ensembles, []))

        # indices = np.random.choice(len(all_samples), num_samples, replace=False)
        # results = all_samples[indices]
        # lex_results = [results[:, results.shape[-1] - 1 - i] for i in range(results.shape[-1])]
        # sorted_results = results[np.lexsort(lex_results)]

        return all_samples


def simple_mnist_layer_ensemble(num_ensemble, **kwargs):
    network = SimpleLayerEnsembleNetwork(num_ensemble, **kwargs)
    return network
