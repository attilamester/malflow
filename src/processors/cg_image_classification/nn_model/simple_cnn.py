import torch
import torch.nn as nn

from core.processors.cg_image_classification.dataset import ImgDataset


class SimpleCNN(torch.nn.Module):

    def __init__(self, dataset: ImgDataset, size_hid: int, dropout: float):
        super().__init__()

        layers = []
        layers.append(
            nn.Conv2d(dataset.img_color_channels, size_hid, kernel_size=1, stride=1, padding=0, bias=True))
        layers.append(
            nn.Conv2d(size_hid, size_hid, kernel_size=1, stride=1, padding=0, bias=True))

        layers.append(nn.BatchNorm2d(size_hid))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.AvgPool2d(1, stride=1))

        self.stack = nn.Sequential(*layers)
        self.fc = nn.Linear(size_hid * dataset.img_shape[0] * dataset.img_shape[1], dataset.num_classes)

    def forward(self, x):
        x = self.stack(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
