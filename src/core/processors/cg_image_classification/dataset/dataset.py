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

    initialized: bool = field(default=False)
    _data_num_classes: int = field(default=None)
    _data_num_items: int = field(default=None)
    _data_mean: Tuple[float, float, float] = field(default=None)
    _data_std: Tuple[float, float, float] = field(default=None)
    _data_df_gt: pd.DataFrame = field(default=None)
    _data_df_gt_filtered: pd.DataFrame = field(default=None)
    _data_class2index: Dict[str, int] = field(default=None)
    _data_index2class: Dict[int, str] = field(default=None)

    @property
    def num_classes(self):
        if self._data_num_classes is None:
            raise Exception("Dataset is not initialized")
        return self._data_num_classes

    @property
    def size(self):
        if self._data_num_items is None:
            raise Exception("Dataset is not initialized")
        return self._data_num_items

    @property
    def mean(self):
        if self._data_mean is None:
            raise Exception("Dataset is not initialized")
        return self._data_mean

    @property
    def std(self):
        if self._data_std is None:
            raise Exception("Dataset is not initialized")
        return self._data_std

    def read_image(self, filename):
        image = cv2.imread(os.path.join(self.img_dir_path, filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    # =================
    # Below, methods are domain-specific. Maybe they should be moved to a subclass
    # =================

    def get_ground_truth(self) -> pd.DataFrame:
        if self._data_df_gt is None:
            self._data_df_gt = pd.read_csv(self.ground_truth_path, delimiter=",")
        return self._data_df_gt

    def get_row_id(self, row: pd.Series) -> str:
        if pd.notna(row["unpacked-md5"]):
            return row["unpacked-md5"]
        else:
            return row["md5"]

    def get_filename(self, md5):
        return f"{md5}_{self.img_shape[0]}x{self.img_shape[1]}_True_True.png"

    def filter_ground_truth(self, items_per_class: int) -> pd.DataFrame:
        df = self.get_ground_truth()
        df_filtered = df_filter_having_at_column_min_occurencies(df, "family", items_per_class)
        family_index = list_to_dict_keys(list(df_filtered["family"].unique()))

        self._data_num_classes = len(family_index)
        self._data_df_gt_filtered = df_filtered
        self._data_class2index = family_index
        self._data_index2class = {v: k for k, v in family_index.items()}
        self.initialized = True

        Logger.info(
            f"[Dataset] Filtered ImgDataset with items_per_class:{items_per_class}. "
            f"Samples: {len(df_filtered)} | Classes: {self.num_classes}")

        self._calculate_mean_std()

        return df_filtered

    def _calculate_mean_std(self):
        if not self.initialized:
            raise Exception("Dataset is not initialized")

        channels_mean = [[], [], []]
        channels_std = [[], [], []]
        for i, row in self._data_df_gt_filtered.iterrows():
            md5 = self.get_row_id(row)
            filename = self.get_filename(md5)
            filepath = os.path.join(self.img_dir_path, filename)

            image = cv2.imread(filepath, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            for i, channel in enumerate(cv2.split(image)):
                channels_mean[i].append(np.mean(channel))
                channels_std[i].append(np.std(channel))

        self._data_mean = np.mean(np.array(channels_mean), axis=1) / 256  # 8-bit images
        self._data_std = np.mean(np.array(channels_std), axis=1) / 256

        Logger.info(f"[Dataset] Calculated mean&std on {len(channels_mean[0])} items from {self.img_dir_path}:\n"
                    f"\tmean: {self._data_mean}\n"
                    f"\t std: {self._data_std}")


class Datasets(Enum):
    BODMAS = ImgDataset(
        ground_truth_path=os.environ["DATASETS_BODMAS_GROUND_TRUTH_PATH"],
        img_dir_path=os.environ["DATASETS_BODMAS_IMG_DIR_PATH"],
        img_shape=Validator.validate_shape(os.environ["DATASETS_BODMAS_IMG_SHAPE"]),
        img_color_channels=Validator.validate_int(os.environ["DATASETS_BODMAS_IMG_COLOR_CHANNELS"]),
        augm=Validator.validate_bool(os.environ["DATASETS_BODMAS_IMG_AUGM"]),
    )
