from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple


@dataclass
class ImgDataset:
    img_shape: Tuple[int, int]
    num_classes: int
    color_channels: 3
    data_dir: str
    metadata_file: str
    images_dir: str
    # "mean": (0.485, 0.456, 0.406),
    # "std": (0.229, 0.224, 0.225),
    augm: bool
    train_batch_size: int = field(default=8)
    test_batch_size: int = field(default=8)


class Datasets(Enum):
    BODMAS = ImgDataset(**{
        "img_shape": (300, 300),
        "num_classes": 56,
        "color_channels": 3,
        "data_dir": None,
        "metadata_file": "BODMAS_ground_truth.csv",
        "images_dir": "BODMAS_images_512_512_False_False",
        # "mean": (0.485, 0.456, 0.406),
        # "std": (0.229, 0.224, 0.225),
        "augm": False
    })

BODMAS_LABEL_MAPPINGS = None
