import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple

import cv2
import numpy as np
import pandas as pd

from util import config
from util.logger import Logger
from util.validators import Validator

config.load_env("core/processors/bagnet/dataset.env")


@dataclass
class ImgDataset:
    ground_truth_path: str
    img_dir_path: str
    img_shape: Tuple[int, int]
    img_color_channels: int

    augm: bool

    train_batch_size: int
    test_batch_size: int

    _mean: Tuple[float, float, float] = field(default=None)
    _std: Tuple[float, float, float] = field(default=None)

    @property
    def mean(self):
        if self._mean is None:
            self.calculate_mean_std()
        return self._mean

    @mean.setter
    def mean(self, value):
        Logger.warning(
            f"Manually setting the `mean` value of the dataset ({self.img_dir_path}) from {self._mean} to {value}.")
        self._mean = value

    @property
    def std(self):
        if self._std is None:
            self.calculate_mean_std()
        return self._std

    @std.setter
    def std(self, value):
        Logger.warning(
            f"Manually setting the `std` value of the dataset ({self.img_dir_path}) from {self._std} to {value}.")
        self._std = value

    def read_ground_truth(self) -> pd.DataFrame:
        return pd.read_csv(self.ground_truth_path, delimiter=",")

    def calculate_mean_std(self):
        channels_mean = [[], [], []]
        channels_std = [[], [], []]
        for filename in os.listdir(self.img_dir_path):
            if filename.endswith(".png"):
                file_path = os.path.join(self.img_dir_path, filename)

                image = cv2.imread(file_path, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                for i, channel in enumerate(cv2.split(image)):
                    channels_mean[i].append(np.mean(channel))
                    channels_std[i].append(np.std(channel))

        self._mean = np.mean(channels_mean, axis=1) / 256  # 8-bit images
        self._std = np.mean(channels_std, axis=1) / 256


class Datasets(Enum):
    BODMAS = ImgDataset(
        ground_truth_path=os.environ["DATASETS_BODMAS_GROUND_TRUTH_PATH"],
        img_dir_path=os.environ["DATASETS_BODMAS_IMG_DIR_PATH"],
        img_shape=Validator.validate_shape(os.environ["DATASETS_BODMAS_IMG_SHAPE"]),
        img_color_channels=Validator.validate_int(os.environ["DATASETS_BODMAS_IMG_COLOR_CHANNELS"]),
        augm=Validator.validate_bool(os.environ["DATASETS_BODMAS_IMG_AUGM"]),
        train_batch_size=Validator.validate_int(os.environ["DATASETS_BODMAS_TRAIN_BATCH_SIZE"]),
        test_batch_size=Validator.validate_int(os.environ["DATASETS_BODMAS_TEST_BATCH_SIZE"])
    )
