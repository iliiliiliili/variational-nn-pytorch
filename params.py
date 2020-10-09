from networks.cifar10_mini_base import createCifar10MiniBase
from networks.mnist_mlp import createMnistMlp
from networks.mnist_double_linear import createMnistDoubleLinear
from networks.mnist_mini2_base import createMnistMini2Base
from networks.mnist_mini_base import createMnistMiniBase
from networks.mnist_auto_encoder_base import createMnistAutoEncoderBase
from networks.vgg import createVGG
from networks.variational import (
    VariationalConvolution,
    VariationalLinear,
)
from networks.classic import ClassicConvolution, ClassicLinear
from networks.dropout import DropoutConvolution, DropoutLinear
import torch
from torchvision import datasets
from torchvision import transforms

from networks.mnist_base import createMnistBase
from networks.cifar10_base import createCifar10Base
from networks.resnet import createResnet

from networks import densenet_pure
from networks import densenet1
from networks import densenet2

from networks import resnet_pure
from networks import vgg_pure

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
                [transforms.ToTensor(), transforms.Normalize((0,), (1,)),]
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


networks = {
    "mnist_base_vnn": createMnistBase(
        VariationalConvolution, VariationalLinear
    ),
    "mnist_base_classic": createMnistBase(ClassicConvolution, ClassicLinear),
    "mnist_base_dropout": createMnistBase(DropoutConvolution, DropoutLinear),
    "mnist_mini_base_vnn": createMnistMiniBase(
        VariationalConvolution, VariationalLinear
    ),
    "mnist_mini_base_classic": createMnistMiniBase(
        ClassicConvolution, ClassicLinear
    ),
    "mnist_mini_base_dropout": createMnistMiniBase(
        DropoutConvolution, DropoutLinear
    ),
    "mnist_mini2_base_vnn": createMnistMini2Base(
        VariationalConvolution, VariationalLinear
    ),
    "mnist_mini2_base_classic": createMnistMini2Base(
        ClassicConvolution, ClassicLinear
    ),
    "mnist_mini2_base_dropout": createMnistMini2Base(
        DropoutConvolution, DropoutLinear
    ),
    "mnist_double_linear_vnn": createMnistDoubleLinear(
        VariationalConvolution, VariationalLinear
    ),
    "mnist_double_linear_classic": createMnistDoubleLinear(
        ClassicConvolution, ClassicLinear
    ),
    "mnist_double_linear_dropout": createMnistDoubleLinear(
        DropoutConvolution, DropoutLinear
    ),
    "mnist_mlp_vnn": createMnistMlp(VariationalConvolution, VariationalLinear),
    "mnist_mlp_classic": createMnistMlp(ClassicConvolution, ClassicLinear),
    "mnist_mlp_dropout": createMnistMlp(DropoutConvolution, DropoutLinear),
    "mnist_auto_encoder_base_vnn": createMnistAutoEncoderBase(
        VariationalConvolution, VariationalLinear
    ),
    "cifar10_base_vnn": createCifar10Base(
        VariationalConvolution, VariationalLinear
    ),
    "cifar10_base_classic": createCifar10Base(
        ClassicConvolution, ClassicLinear
    ),
    "cifar10_base_dropout": createCifar10Base(
        DropoutConvolution, DropoutLinear
    ),
    "cifar10_mini_base_vnn": createCifar10MiniBase(
        VariationalConvolution, VariationalLinear
    ),
    "cifar10_mini_base_classic": createCifar10MiniBase(
        ClassicConvolution, ClassicLinear
    ),
    "cifar10_mini_base_dropout": createCifar10MiniBase(
        DropoutConvolution, DropoutLinear
    ),
    "resnet_vnn_18": createResnet(VariationalConvolution, VariationalLinear)[
        "ResNet18"
    ],
    "resnet_vnn_34": createResnet(VariationalConvolution, VariationalLinear)[
        "ResNet34"
    ],
    "resnet_vnn_50": createResnet(VariationalConvolution, VariationalLinear)[
        "ResNet50"
    ],
    "resnet_vnn_101": createResnet(VariationalConvolution, VariationalLinear)[
        "ResNet101"
    ],
    "resnet_vnn_152": createResnet(VariationalConvolution, VariationalLinear)[
        "ResNet152"
    ],
    "resnet_classic_18": createResnet(ClassicConvolution, ClassicLinear)[
        "ResNet18"
    ],
    "resnet_classic_34": createResnet(ClassicConvolution, ClassicLinear)[
        "ResNet34"
    ],
    "resnet_classic_50": createResnet(ClassicConvolution, ClassicLinear)[
        "ResNet50"
    ],
    "resnet_classic_101": createResnet(ClassicConvolution, ClassicLinear)[
        "ResNet101"
    ],
    "resnet_classic_152": createResnet(ClassicConvolution, ClassicLinear)[
        "ResNet152"
    ],
    "resnet_dropout_18": createResnet(DropoutConvolution, DropoutLinear)[
        "ResNet18"
    ],
    "resnet_dropout_34": createResnet(DropoutConvolution, DropoutLinear)[
        "ResNet34"
    ],
    "resnet_dropout_50": createResnet(DropoutConvolution, DropoutLinear)[
        "ResNet50"
    ],
    "resnet_dropout_101": createResnet(DropoutConvolution, DropoutLinear)[
        "ResNet101"
    ],
    "resnet_dropout_152": createResnet(DropoutConvolution, DropoutLinear)[
        "ResNet152"
    ],
    "vgg_vnn_11": createVGG(VariationalConvolution, VariationalLinear)[
        "VGG11"
    ],
    "vgg_vnn_13": createVGG(VariationalConvolution, VariationalLinear)[
        "VGG13"
    ],
    "vgg_vnn_16": createVGG(VariationalConvolution, VariationalLinear)[
        "VGG16"
    ],
    "vgg_vnn_19": createVGG(VariationalConvolution, VariationalLinear)[
        "VGG19"
    ],
    "vgg_classic_11": createVGG(ClassicConvolution, ClassicLinear)["VGG11"],
    "vgg_classic_13": createVGG(ClassicConvolution, ClassicLinear)["VGG13"],
    "vgg_classic_16": createVGG(ClassicConvolution, ClassicLinear)["VGG16"],
    "vgg_classic_19": createVGG(ClassicConvolution, ClassicLinear)["VGG19"],
    "vgg_dropout_11": createVGG(DropoutConvolution, DropoutLinear)["VGG11"],
    "vgg_dropout_13": createVGG(DropoutConvolution, DropoutLinear)["VGG13"],
    "vgg_dropout_16": createVGG(DropoutConvolution, DropoutLinear)["VGG16"],
    "vgg_dropout_19": createVGG(DropoutConvolution, DropoutLinear)["VGG19"],
    "densenet_pure": densenet_pure.DenseNet121,
    "densenet1_classic": densenet1.createDenseNet(
        ClassicConvolution, ClassicLinear
    ),
    "densenet1_vnn": densenet1.createDenseNet(
        VariationalConvolution, VariationalLinear
    ),
    "densenet2_classic": densenet2.createDenseNet2(
        ClassicConvolution, ClassicLinear
    ),
    "densenet2_vnn": densenet2.createDenseNet2(
        VariationalConvolution, VariationalLinear
    ),
    "densenet2_dropout": densenet2.createDenseNet2(
        DropoutConvolution, DropoutLinear
    ),
    "resnet_pure_50": resnet_pure.Resnet50,
    "vgg_pure_16": vgg_pure.Vgg16,
    "vgg_pure_19": vgg_pure.Vgg19,
}

loss_functions = {
    "cross_entropy": torch.nn.CrossEntropyLoss(),
    "mse": torch.nn.MSELoss(),
    "bce": torch.nn.BCELoss(),
}

activations = {
    "relu": torch.nn.ReLU,
    "relu6": torch.nn.ReLU6,
    "sigmoid": torch.nn.Sigmoid,
    "tanh": torch.nn.Tanh,
    "leacky_relu": torch.nn.LeakyReLU,
    "noact": lambda **kwargs: None,
}

activation_params = {
    "relu": ["inplace"],
    "relu6": ["inplace"],
    "sigmoid": [],
    "tanh": [],
    "leacky_relu": ["negative_slope", "inplace"],
    "noact": [],
}

optimizers = {
    "SGD": torch.optim.SGD,
    "Adam": torch.optim.Adam,
}

optimizer_params = {
    "SGD": ["lr", "momentum", "dampening", "weight_decay", "nesterov"],
    "Adam": ["lr", "betas", "eps", "weight_decay", "amsgrad"],
}
