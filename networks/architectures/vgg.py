from networks.network import Network
import torch
import torch.nn as nn


def createVGG(Convolution, Linear):
    class VGG(Network):
        def __init__(
            self, features, num_classes=10, init_weights=True, **kwargs
        ):
            super(VGG, self).__init__()
            self.features = features
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            self.classifier = nn.Sequential(
                Linear(512 * 7 * 7, 512, **kwargs),
                Linear(512, 512, **kwargs),
                Linear(512, 10, **kwargs),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x

    def make_layers(cfg, in_channels=3, **kwargs):
        layers = []
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = Convolution(
                    in_channels, v, kernel_size=3, padding=1, **kwargs
                )
                layers += [conv2d]
                in_channels = v
        return nn.Sequential(*layers)

    cfgs = {
        "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
        "B": [
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            "M",
            512,
            512,
            "M",
            512,
            512,
            "M",
        ],
        "D": [
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            256,
            "M",
            512,
            512,
            512,
            "M",
            512,
            512,
            512,
            "M",
        ],
        "E": [
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            256,
            256,
            "M",
            512,
            512,
            512,
            512,
            "M",
            512,
            512,
            512,
            512,
            "M",
        ],
    }

    def make_vgg(cfg, in_channels=3, **kwargs):
        return VGG(make_layers(cfgs[cfg], in_channels, **kwargs), **kwargs)

    def VGG11(in_channels=3, **kwargs):
        return make_vgg("A", in_channels, **kwargs)

    def VGG13(in_channels=3, **kwargs):
        return make_vgg("B", in_channels, **kwargs)

    def VGG16(in_channels=3, **kwargs):
        return make_vgg("D", in_channels, **kwargs)

    def VGG19(in_channels=3, **kwargs):
        return make_vgg("E", in_channels, **kwargs)

    return {
        "VGG11": VGG11,
        "VGG13": VGG13,
        "VGG16": VGG16,
        "VGG19": VGG19,
    }
