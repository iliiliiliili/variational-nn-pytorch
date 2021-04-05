from torch import nn
from networks.network import Network
from networks.classic import ClassicConvolution, ClassicLinear
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, **kwargs):

        super().__init__()

        self.block = nn.Sequential(
            ClassicConvolution(
                in_channels, out_channels, kernel_size=3,
                stride=stride, padding=1, bias=False,
                use_batch_norm=True,
                **kwargs
            ),
            ClassicConvolution(
                out_channels, out_channels, kernel_size=3,
                stride=1, padding=1, bias=False,
                use_batch_norm=True, **kwargs
            )
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                ClassicConvolution(
                    in_channels, self.expansion * out_channels,
                    kernel_size=1, stride=stride, bias=False,
                    use_batch_norm=True,
                    **kwargs
                ),
            )

    def forward(self, x):
        out = self.block(x)
        out += self.shortcut(x)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, **kwargs):

        super().__init__()

        self.block = nn.Sequential(
            ClassicConvolution(
                in_channels, out_channels, kernel_size=1,
                bias=False,
                use_batch_norm=True,
                **kwargs
            ),
            ClassicConvolution(
                out_channels, out_channels, kernel_size=3,
                stride=stride, padding=1, bias=False,
                use_batch_norm=True, **kwargs
            ),
            ClassicConvolution(
                out_channels, self.expansion * out_channels, kernel_size=1,
                bias=False,
                use_batch_norm=True, **kwargs
            ),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                ClassicConvolution(
                    in_channels, self.expansion * out_channels,
                    kernel_size=1, stride=stride, bias=False,
                    use_batch_norm=True,
                    **kwargs
                ),
            )

    def forward(self, x):
        out = self.block(x)
        out += self.shortcut(x)
        return out


class ResNetClassic(Network):
    def __init__(
        self, block, num_blocks, num_classes=10,
        input_channels=3, **kwargs
    ):

        super().__init__()

        self.in_channels = 64

        self.conv1 = ClassicConvolution(
            input_channels, 64, kernel_size=3,
            stride=1, padding=1, bias=False,
            use_batch_norm=True,
            **kwargs
        )
        self.layer1 = self._make_layer(
            block, 64, num_blocks[0], stride=1, **kwargs
        )
        self.layer2 = self._make_layer(
            block, 128, num_blocks[1], stride=2, **kwargs
        )
        self.layer3 = self._make_layer(
            block, 256, num_blocks[2], stride=2, **kwargs
        )
        self.layer4 = self._make_layer(
            block, 512, num_blocks[3], stride=2, **kwargs
        )
        self.linear = ClassicLinear(
            512 * block.expansion, num_classes, **kwargs
        )

    def _make_layer(self, block, out_channels, num_blocks, stride, **kwargs):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(self.in_channels, out_channels, stride, **kwargs)
            )
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNetClassic18(num_classes=10, input_channels=3, **kwargs):
    return ResNetClassic(
        BasicBlock, [2, 2, 2, 2], num_classes=num_classes,
        input_channels=input_channels, **kwargs
    )


def ResNetClassic34(num_classes=10, input_channels=3, **kwargs):
    return ResNetClassic(
        BasicBlock, [3, 4, 6, 3], num_classes=num_classes,
        input_channels=input_channels, **kwargs
    )


def ResNetClassic50(num_classes=10, input_channels=3, **kwargs):
    return ResNetClassic(
        Bottleneck, [3, 4, 6, 3], num_classes=num_classes,
        input_channels=input_channels, **kwargs
    )


def ResNetClassic101(num_classes=10, input_channels=3, **kwargs):
    return ResNetClassic(
        Bottleneck, [3, 4, 23, 3], num_classes=num_classes,
        input_channels=input_channels, **kwargs
    )


def ResNetClassic152(num_classes=10, input_channels=3, **kwargs):
    return ResNetClassic(
        Bottleneck, [3, 8, 36, 3], num_classes=num_classes,
        input_channels=input_channels, **kwargs
    )
