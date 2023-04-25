# Variational Neural Networks Pytorch

This repository contains a Pytorch implementation of Variational Neural Networks (VNNs) and image classification experiments for [Variational Neural Networks paper](https://arxiv.org/abs/2207.01524).

The corresponding package contains layer implementations for VNNs and other used architectures. It can be installed using `pip install vnn`.

Bayesian Neural Networks (BNNs) provide a tool to estimate the uncertainty of a neural network by considering a distribution over weights and sampling different models for each input. In this paper, we propose a method for uncertainty estimation in neural networks called Variational Neural Network that, instead of considering a distribution over weights, generates parameters for the output distribution of a layer by transforming its inputs with learnable sub-layers. In uncertainty quality estimation experiments, we show that VNNs achieve better uncertainty quality than Monte Carlo Dropout or Bayes By Backpropagation methods.

## Run

Use `run_example.sh` to train and evaluate a single model on MNIST.
The corresponding reproducible capsule is available at [CodeOcean](https://codeocean.com/capsule/9585164/tree).

## Package
Use `pip install vnn` or `python3 -m pip install vnn` to install the package.
The package includes only the layer implementations of VNNs, as well as dropout, functional and classic layers.
These layers are implemented with the same interface, making it easy to implement different versions of your desired network by changing the class names.

An example of a simple convolutional network:
```python
import torch
from vnn import VariationalConvolution, VariationalLinear

class Based(torch.nn.Module):

    def __init__(self, **kwargs) -> None:

        super().__init__()

        self.model = nn.Sequential(
            VariationalConvolution(1, 256, 9, 1, **kwargs),
            VariationalConvolution(256, 256, 9, 2, **kwargs),
            VariationalConvolution(256, 16, 4, 1, **kwargs),
            torch.nn.Flatten(start_dim=1),
            VariationalLinear(3 * 3 * 16, 10, **kwargs),
        )

    def forward(self, x):

        return self.model(x)
```

The same classic network:

```python
import torch
from vnn.classic import ClassicConvolution, ClassicLinear

class Based(torch.nn.Module):

    def __init__(self, **kwargs) -> None:

        super().__init__()

        self.model = nn.Sequential(
            ClassicConvolution(1, 256, 9, 1, **kwargs),
            ClassicConvolution(256, 256, 9, 2, **kwargs),
            ClassicConvolution(256, 16, 4, 1, **kwargs),
            torch.nn.Flatten(start_dim=1),
            ClassicLinear(3 * 3 * 16, 10, **kwargs),
        )

    def forward(self, x):

        return self.model(x)
```

Or a generalized network class:
```python
import torch
from vnn import VariationalConvolution, VariationalLinear
from vnn.classic import ClassicConvolution, ClassicLinear
from vnn.dropout import DropoutConvolution, DropoutLinear
from vnn.functional import FunctionalConvolution, FunctionalLinear

def create_based(Convolution, Linear):
    class Based(torch.nn.Module):

        def __init__(self, **kwargs) -> None:

            super().__init__()

            self.model = nn.Sequential(
                Convolution(1, 256, 9, 1, **kwargs),
                Convolution(256, 256, 9, 2, **kwargs),
                Convolution(256, 16, 4, 1, **kwargs),
                torch.nn.Flatten(start_dim=1),
                Linear(3 * 3 * 16, 10, **kwargs),
            )

        def forward(self, x):

            return self.model(x)

based_vnn = create_based(VariationalConvolution, VariationalLinear)
based_classic = create_based(ClassicConvolution, ClassicLinear)
based_dropout = create_based(DropoutConvolution, DropoutLinear)
based_functional = create_based(FunctionalConvolution, FunctionalLinear) # see hypermodels on how to use functional layers

```

## Citation

If you use this work for your research, you can cite it as:
### Library:
```
@article{oleksiienko2022vnntorchjax,
    title = {Variational Neural Networks implementation in Pytorch and JAX},
    author = {Oleksiienko, Illia and Tran, Dat Thanh and Iosifidis, Alexandros},
    journal = {Software Impacts},
    volume = {14},
    pages = {100431},
    year = {2022},
}
```
### Paper:
```
@article{oleksiienko2023vnn,
  title={Variational Neural Networks}, 
  author = {Oleksiienko, Illia and Tran, Dat Thanh and Iosifidis, Alexandros},
  journal={arxiv:2207.01524}, 
  year={2023},
}
```
