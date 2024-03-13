"""
Based on https://github.com/wielandbrendel/bag-of-local-features-models
"""

import math
from typing import Type

import torch
import torch.nn as nn
from torch.utils import model_zoo

from core.processors.cg_image_classification.dataset import ImgDataset
from core.processors.cg_image_classification.paths import get_cg_image_classification_tb_log_dir

__all__ = ["bagnet9", "bagnet17", "bagnet33"]

from util.logger import Logger

model_urls = {
    "bagnet9": "https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/249e8fa82c0913623a807d9d35eeab9da7dcc2a8/bagnet8-34f4ccd2.pth.tar",
    "bagnet17": "https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/249e8fa82c0913623a807d9d35eeab9da7dcc2a8/bagnet16-105524de.pth.tar",
    "bagnet33": "https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/249e8fa82c0913623a807d9d35eeab9da7dcc2a8/bagnet32-2ddd53ed.pth.tar",
}


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, kernel_size=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, **kwargs):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if residual.size(-1) != out.size(-1):
            diff = residual.size(-1) - out.size(-1)
            residual = residual[:, :, :-diff, :-diff]

        out += residual
        out = self.relu(out)

        return out


class BagNet(nn.Module):
    def __init__(self, dataset: ImgDataset, block: Type[Bottleneck], layers, strides=[1, 2, 2, 2], patch_size: int = 3,
                 kernel3=[0, 0, 0, 0], avg_pool=True):
        super(BagNet, self).__init__()

        self.inplanes = 64
        self.dataset = dataset
        self.num_classes = dataset.num_classes
        self.patch_size = patch_size

        self.conv1 = nn.Conv2d(dataset.img_color_channels, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.001)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], kernel3=kernel3[0], prefix="layer1")
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], kernel3=kernel3[1], prefix="layer2")
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], kernel3=kernel3[2], prefix="layer3")
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], kernel3=kernel3[3], prefix="layer4")
        self.avgpool = nn.AvgPool2d(1, stride=1)

        self.fc = nn.Linear(512 * block.expansion, self.num_classes)
        self.avg_pool = avg_pool
        self.block = block

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block: Type[Bottleneck], planes, blocks, stride=1, kernel3=0, prefix=""):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        kernel = 1 if kernel3 == 0 else 3
        layers.append(block(self.inplanes, planes, stride, downsample, kernel_size=kernel))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            kernel = 1 if kernel3 <= i else 3
            layers.append(block(self.inplanes, planes, kernel_size=kernel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.avg_pool:
            x = nn.AvgPool2d((x.size()[2], x.size()[3]), stride=1)(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        else:
            x = x.permute(0, 2, 3, 1)
            x = self.fc(x)

        return x


def weight_init(layer: nn.Linear):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.constant_(layer.bias, 0.)


def create_bagnet_model(dataset: ImgDataset, patch_size: int, kernel3, strides=None, pretrained=False,
                        pretrained_model_name=None,
                        **kwargs) -> BagNet:
    """
    Constructs a BagNet model.
    Args:
        pretrained (bool): If True, returns a model pretrained on ImageNet
    """
    if not strides:
        strides = [2, 2, 2, 1]
    model = BagNet(dataset, Bottleneck, [3, 4, 6, 3], strides=strides, patch_size=patch_size, kernel3=kernel3, **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(
            model_urls[pretrained_model_name],
            model_dir=get_cg_image_classification_tb_log_dir(),
            map_location=torch.device("cpu") if not torch.cuda.is_available() else torch.device(
                "cuda:0"),
            check_hash=True)

        Logger.info("Dropping the last fc layer weight&bias of the pretrained BagNet model")
        pretrained_state_dict.pop("fc.weight")
        pretrained_state_dict.pop("fc.bias")

        if model.conv1.weight.size(1) == 1:
            Logger.info(
                "Converting the pretrained BagNet model's first conv layer weight to greyscale by summing up the channels")
            conv1_weight = pretrained_state_dict.pop("conv1.weight")
            conv1_weight = torch.sum(conv1_weight, dim=1, keepdim=True)
            pretrained_state_dict["conv1.weight"] = conv1_weight

        model.load_state_dict(pretrained_state_dict, strict=False)
        weight_init(model.fc)

    return model


def bagnet9(dataset: ImgDataset, strides=None, pretrained=False, **kwargs) -> BagNet:
    return create_bagnet_model(dataset, 9, kernel3=[1, 1, 0, 0], strides=strides, pretrained=pretrained,
                               pretrained_model_name="bagnet9", **kwargs)


def bagnet17(dataset: ImgDataset, strides=None, pretrained=False, **kwargs) -> BagNet:
    return create_bagnet_model(dataset, 17, kernel3=[1, 1, 1, 0], strides=strides, pretrained=pretrained,
                               pretrained_model_name="bagnet17", **kwargs)


def bagnet33(dataset: ImgDataset, strides=None, pretrained=False, **kwargs) -> BagNet:
    return create_bagnet_model(dataset, 33, kernel3=[1, 1, 1, 1], strides=strides, pretrained=pretrained,
                               pretrained_model_name="bagnet33", **kwargs)
