import math
import torch
import torch.nn as nn
from networks.network import Network
import torch.nn.functional as F


def createDenseNet2(Convolution, Linear):
    class Bottleneck(nn.Module):
        def __init__(self, in_planes, growth_rate, **kwargs):
            super(Bottleneck, self).__init__()
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.conv1 = Convolution(
                in_planes, 4 * growth_rate, kernel_size=1, **kwargs,
            )
            self.bn2 = nn.BatchNorm2d(4 * growth_rate)
            self.conv2 = Convolution(
                4 * growth_rate,
                growth_rate,
                kernel_size=3,
                padding=1,
                **kwargs,
            )

        def forward(self, x):
            out = self.conv1(self.bn1(x))
            out = self.conv2(self.bn2(out))
            out = torch.cat([out, x], 1)
            return out

    class Transition(nn.Module):
        def __init__(self, in_planes, out_planes, **kwargs):
            super(Transition, self).__init__()
            self.bn = nn.BatchNorm2d(in_planes)
            self.conv = Convolution(
                in_planes, out_planes, kernel_size=1, **kwargs,
            )

        def forward(self, x):
            out = self.conv(self.bn(x))
            out = F.avg_pool2d(out, 2)
            return out

    class DenseNet(Network):
        def __init__(
            self,
            block,
            nblocks,
            growth_rate=12,
            reduction=0.5,
            nb_class=10,
            **kwargs
        ):
            super(DenseNet, self).__init__()
            self.growth_rate = growth_rate

            num_planes = 2 * growth_rate
            self.conv1 = Convolution(
                3, num_planes, kernel_size=3, padding=1, **kwargs,
            )

            self.dense1 = self._make_dense_layers(
                block, num_planes, nblocks[0], **kwargs
            )
            num_planes += nblocks[0] * growth_rate
            out_planes = int(math.floor(num_planes * reduction))
            self.trans1 = Transition(num_planes, out_planes)
            num_planes = out_planes

            self.dense2 = self._make_dense_layers(
                block, num_planes, nblocks[1], **kwargs
            )
            num_planes += nblocks[1] * growth_rate
            out_planes = int(math.floor(num_planes * reduction))
            self.trans2 = Transition(num_planes, out_planes)
            num_planes = out_planes

            self.dense3 = self._make_dense_layers(
                block, num_planes, nblocks[2], **kwargs
            )
            num_planes += nblocks[2] * growth_rate
            out_planes = int(math.floor(num_planes * reduction))
            self.trans3 = Transition(num_planes, out_planes)
            num_planes = out_planes

            self.dense4 = self._make_dense_layers(
                block, num_planes, nblocks[3], **kwargs
            )
            num_planes += nblocks[3] * growth_rate

            self.bn = nn.BatchNorm2d(num_planes)
            self.linear = Linear(
                num_planes,
                nb_class,
                activation=torch.nn.ReLU(),
            )

            self.initialize()

        def _make_dense_layers(self, block, in_planes, nblock, **kwargs):
            layers = []
            for i in range(nblock):
                layers.append(block(in_planes, self.growth_rate, **kwargs))
                in_planes += self.growth_rate
            return nn.Sequential(*layers)

        def forward(self, x):
            out = self.conv1(x)
            # print(out.size())
            out = self.trans1(self.dense1(out))
            # print(out.size())
            out = self.trans2(self.dense2(out))
            # print(out.size())
            out = self.trans3(self.dense3(out))
            # print(out.size())
            out = self.dense4(out)
            # print(out.size())
            out = F.avg_pool2d(self.bn(out), 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out

        def initialize(self,):
            for layer in self.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight)
                    nn.init.normal_(layer.bias)
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight)
                    nn.init.normal_(layer.bias)

                if isinstance(layer, nn.BatchNorm2d):
                    nn.init.constant_(layer.weight, 1.0)
                    nn.init.constant_(layer.bias, 0.0)

    def DenseNet121(**kwargs):
        return DenseNet(
            Bottleneck, [6, 12, 24, 16], growth_rate=32, nb_class=10, **kwargs
        )

    return DenseNet121
