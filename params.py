from losses import bbb_loss, gaussian_regression_loss
from networks.const_mean_variational import (
    OneMeanVariationalConvolution,
    OneMeanVariationalLinear,
    ZeroMeanVariationalConvolution,
    ZeroMeanVariationalLinear,
)
from networks.ensemble import create_ensemble
from networks.hypermodel import HypermodelConvolution, HypermodelLinear, create_bbb_hypermodel, create_hypermodel, create_linear_hypermodel
from networks.single_mean_variational import (
    SingleMeanVariationalConvolution,
    SingleMeanVariationalLinear,
)

from networks.single_std_variational import (
    SingleStdVariationalConvolution, SingleStdVariationalLinear,
)

from networks.architectures.cifar10_mini_base import createCifar10MiniBase
from networks.architectures.mnist_mlp import createMnistMlp
from networks.architectures.mnist_conv_max import createMnistConvMax
from networks.architectures.mnist_mini2_base import createMnistMini2Base
from networks.architectures.mnist_mini_base import createMnistMiniBase
from networks.architectures.mnist_auto_encoder_base import createMnistAutoEncoderBase
from networks.architectures.vgg import createVGG
from networks.variational import (
    VariationalConvolution,
    VariationalLinear,
)
from networks.classic import ClassicConvolution, ClassicLinear
from networks.dropout import DropoutConvolution, DropoutLinear
import torch
from torchvision import datasets
from torchvision import transforms

from networks.architectures.mnist_base import createMnistBase
from networks.architectures.cifar10_base import createCifar10Base
from networks.architectures.resnet import createResnet

from networks.architectures import densenet_pure
from networks.architectures import densenet1
from networks.architectures import densenet2

from networks.architectures import resnet_pure
from networks.architectures import vgg_pure


SEED = 2605

attacked_dataset_params = {

    "mnist": {
        "dataset": datasets.MNIST,
        "path": "./datasets/",
        "train_size": 59000,
        "validation_size": 1000,
        "transform": {
            "all": transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
        },
        "mean": (0.1307,),
        "std": (0.3081,),
    },
    "cifar10": {
        "dataset": datasets.CIFAR10,
        "path": "./datasets/",
        "train_size": 40000,
        "validation_size": 10000,
        "transform": {
            "all": transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
        },
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
    },
    "cifar10_n": {
        "dataset": datasets.CIFAR10,
        "path": "./datasets/",
        "train_size": 40000,
        "validation_size": 10000,
        "transform": {
            "all": transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
        },
        "mean": [x / 255 for x in [125.3, 123.0, 113.9]],
        "std": [x / 255 for x in [63.0, 62.1, 66.7]],
    },
    "cifar10_n2": {
        "dataset": datasets.CIFAR10,
        "path": "./datasets/",
        "train_size": 40000,
        "validation_size": 10000,
        "transform": {
            "train": transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                ]
            ),
            "test": transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            ),
        },
        "mean": [x / 255 for x in [125.3, 123.0, 113.9]],
        "std": [x / 255 for x in [63.0, 62.1, 66.7]],
    },
}

perturbed_dataset_params = {

    "mnist": lambda noise: {
        "dataset": datasets.MNIST,
        "path": "./datasets/",
        "train_size": 59000,
        "validation_size": 1000,
        "transform": {
            "all": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Lambda(noise),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )
        },
    },
    "cifar10": lambda noise: {
        "dataset": datasets.CIFAR10,
        "path": "./datasets/",
        "train_size": 40000,
        "validation_size": 10000,
        "transform": {
            "all": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Lambda(noise),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        },
    },
    "cifar10_n": lambda noise: {
        "dataset": datasets.CIFAR10,
        "path": "./datasets/",
        "train_size": 40000,
        "validation_size": 10000,
        "transform": {
            "all": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Lambda(noise),
                    transforms.Normalize(
                        [x / 255 for x in [125.3, 123.0, 113.9]],
                        [x / 255 for x in [63.0, 62.1, 66.7]],
                    ),
                ]
            )
        },
    },
    "cifar10_n2": lambda noise: {
        "dataset": datasets.CIFAR10,
        "path": "./datasets/",
        "train_size": 40000,
        "validation_size": 10000,
        "transform": {
            "train": transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    transforms.Lambda(noise),
                    transforms.Normalize(
                        [x / 255 for x in [125.3, 123.0, 113.9]],
                        [x / 255 for x in [63.0, 62.1, 66.7]],
                    ),
                ]
            ),
            "test": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Lambda(noise),
                    transforms.Normalize(
                        [x / 255 for x in [125.3, 123.0, 113.9]],
                        [x / 255 for x in [63.0, 62.1, 66.7]],
                    ),
                ]
            ),
        },
    },
}

dataset_params = {
    "mnist": {
        "dataset": datasets.MNIST,
        "path": "./datasets/",
        "train_size": 50000,
        "validation_size": 10000,
        "transform": {
            "all": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )
        },
    },
    "mnist_0_1": {
        "dataset": datasets.MNIST,
        "path": "./datasets/",
        "train_size": 50000,
        "validation_size": 10000,
        "transform": {
            "all": transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0,), (1,))]
            )
        },
    },
    "cifar10": {
        "dataset": datasets.CIFAR10,
        "path": "./datasets/",
        "train_size": 40000,
        "validation_size": 10000,
        "transform": {
            "all": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        },
    },
    "cifar10_n": {
        "dataset": datasets.CIFAR10,
        "path": "./datasets/",
        "train_size": 40000,
        "validation_size": 10000,
        "transform": {
            "all": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [x / 255 for x in [125.3, 123.0, 113.9]],
                        [x / 255 for x in [63.0, 62.1, 66.7]],
                    ),
                ]
            )
        },
    },
    "cifar10_n2": {
        "dataset": datasets.CIFAR10,
        "path": "./datasets/",
        "train_size": 40000,
        "validation_size": 10000,
        "transform": {
            "train": transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [x / 255 for x in [125.3, 123.0, 113.9]],
                        [x / 255 for x in [63.0, 62.1, 66.7]],
                    ),
                ]
            ),
            "test": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [x / 255 for x in [125.3, 123.0, 113.9]],
                        [x / 255 for x in [63.0, 62.1, 66.7]],
                    ),
                ]
            ),
        },
    },
}


def create_network(network_name, network_type):

    def model_generator(network_creator, network_type):
        if network_type == "vnn":
            return network_creator(VariationalConvolution, VariationalLinear)
        if network_type == "classic":
            return network_creator(ClassicConvolution, ClassicLinear)
        elif network_type == "dropout":
            return network_creator(DropoutConvolution, DropoutLinear)
        elif network_type == "bbb":
            return lambda *args, **kwargs: create_bbb_hypermodel(network_creator(HypermodelConvolution, HypermodelLinear), *args, **kwargs)
        elif network_type == "hypermodel":
            return lambda *args, **kwargs: create_linear_hypermodel(network_creator(HypermodelConvolution, HypermodelLinear), *args, **kwargs)
        elif network_type == "ensemble":
            return lambda *args, **kwargs: create_ensemble(model_generator(network_creator, "classic"), *args, **kwargs)
        elif network_type == "ensemble_vnn":
            return lambda *args, **kwargs: create_ensemble(model_generator(network_creator, "vnn"), *args, **kwargs)
        elif network_type == "ensemble_dropout":
            return lambda *args, **kwargs: create_ensemble(model_generator(network_creator, "dropout"), *args, **kwargs)
        elif network_type == "ensemble_bbb":
            return lambda *args, **kwargs: create_ensemble(model_generator(network_creator, "bbb"), *args, **kwargs)
        elif network_type == "ensemble_hypermodel":
            return lambda *args, **kwargs: create_ensemble(model_generator(network_creator, "hypermodel"), *args, **kwargs)
        elif network_type == "ensemble_ensemble":
            return lambda *args, **kwargs: create_ensemble(model_generator(network_creator, "ensemble"), *args, **kwargs)

    creators = {
        "mnist_base": createMnistBase,
        "mnist_mini_base": createMnistMiniBase,
        "mnist_mini2_base": createMnistMini2Base,
        "mnist_conv_max": createMnistConvMax,
        "mnist_mlp": createMnistMlp,
        "mnist_auto_encoder_base": createMnistAutoEncoderBase,
        "cifar10_base": createCifar10Base,
        "cifar10_mini_base": createCifar10MiniBase,
        "resnet_18": lambda *args: createResnet(*args)["ResNet18"],
        "resnet_34": lambda *args: createResnet(*args)["ResNet34"],
        "resnet_50": lambda *args: createResnet(*args)["ResNet50"],
        "resnet_101": lambda *args: createResnet(*args)["ResNet101"],
        "resnet_152": lambda *args: createResnet(*args)["ResNet152"],
        "vgg_11": lambda *args: createVGG(*args)["VGG11"],
        "vgg_13": lambda *args: createVGG(*args)["VGG13"],
        "vgg_16": lambda *args: createVGG(*args)["VGG16"],
        "vgg_19": lambda *args: createVGG(*args)["VGG19"],
        "densenet1": densenet1.createDenseNet,
        "densenet2": densenet2.createDenseNet2,
    }

    result = model_generator(creators[network_name], network_type)

    return result


loss_functions = {
    "cross_entropy": torch.nn.CrossEntropyLoss,
    "gaussian_regression_loss": gaussian_regression_loss,
    "mse": torch.nn.MSELoss,
    "bce": torch.nn.BCELoss,
    "bbb": bbb_loss,
}

loss_params = {
    "cross_entropy": [],
    "gaussian_regression_loss": ["noise_std"],
    "mse": [],
    "bce": [],
    "bbb": ["sigma_0"],
}


loss_functions_that_use_network = ["bbb"]

activations = {
    "relu": torch.nn.ReLU,
    "relu6": torch.nn.ReLU6,
    "sigmoid": torch.nn.Sigmoid,
    "tanh": torch.nn.Tanh,
    "lrelu": torch.nn.LeakyReLU,
    "none": lambda **kwargs: None,
}

activation_params = {
    "relu": ["inplace"],
    "relu6": ["inplace"],
    "sigmoid": [],
    "tanh": [],
    "lrelu": ["negative_slope", "inplace"],
    "none": [],
}

optimizers = {
    "SGD": torch.optim.SGD,
    "Adam": torch.optim.Adam,
}

optimizer_params = {
    "SGD": ["lr", "momentum", "dampening", "weight_decay", "nesterov"],
    "Adam": ["lr", "betas", "eps", "weight_decay", "amsgrad"],
}
