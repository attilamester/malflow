import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, Dict

import cv2
import numpy as np
import pandas as pd

from core.processors.cg_image_classification.dataset.preprocess import df_filter_having_at_column_min_occurencies, \
    list_to_dict_keys
from util.logger import Logger
from util.validators import Validator


@dataclass
class ImgDataset:
    ground_truth_path: str
    img_dir_path: str
    img_shape: Tuple[int, int]
    img_color_channels: int

    augm: bool

    _mean: Tuple[float, float, float] = field(default=None)
    _std: Tuple[float, float, float] = field(default=None)
    _num_classes: int = field(default=None)

    initialized: bool = field(default=False)
    data_df_gt: pd.DataFrame = field(default=None)
    data_df_gt_filtered: pd.DataFrame = field(default=None)
    data_class2index: Dict[str, int] = field(default=None)
    data_index2class: Dict[int, str] = field(default=None)

    @property
    def num_classes(self):
        if self._num_classes is None:
            raise Exception("num_classes is not set yet")
        return self._num_classes

    @num_classes.setter
    def num_classes(self, value):
        self._num_classes = value

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

    def get_ground_truth(self) -> pd.DataFrame:
        if self.data_df_gt is None:
            self.data_df_gt = pd.read_csv(self.ground_truth_path, delimiter=",")
        return self.data_df_gt

    def filter_ground_truth(self, items_per_class: int) -> pd.DataFrame:
        Logger.info(f"Filtering ImgDataset with items_per_class:{items_per_class}")
        df = self.get_ground_truth()
        df_filtered = df_filter_having_at_column_min_occurencies(df, "family", items_per_class)
        family_index = list_to_dict_keys(list(df_filtered["family"].unique()))

        self.num_classes = len(family_index)
        self.data_df_gt_filtered = df_filtered
        self.data_class2index = family_index
        self.data_index2class = {v: k for k, v in family_index.items()}
        self.initialized = True
        return df_filtered

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
    )
