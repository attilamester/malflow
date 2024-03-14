"""
This is the entry point of the import process -- imported from main.py
Ensure the envs are loaded.
"""
from core.processors.cg_image_classification.paths import get_cg_image_classification_env
from util import config

config.load_env(get_cg_image_classification_env())

import logging
import os
from typing import Dict, Union

import torch.utils.data
import torchvision

from core.processors.cg_image_classification import paths
from core.processors.cg_image_classification.dataset import Datasets, ImgDataset
from core.processors.cg_image_classification.dataset.dataloader import create_bodmas_train_val_loader
from core.processors.cg_image_classification.hparams import HPARAMS, get_hparam_value
from core.processors.cg_image_classification.nn_model import bagnet17, bagnet9, bagnet33
from util.logger import Logger


# =======================================================
# \\\\\\\\\\ Required functions for main.py /////////////
# see: https://github.com/attilamester/pytorch-examples/blob/feature/83/imagenet/main.py
# =======================================================

def get_model() -> torch.nn.Module:
    """depends on: DATASET"""

    global MODEL, DATASET

    if MODEL is not None:
        return MODEL

    if DATASET is None:
        get_dataset()

    hp_model = get_hparam_value(HPARAMS.MODEL)
    hp_model_pretrained = get_hparam_value(HPARAMS.MODEL_PRETRAINED)

    if hp_model.startswith("bagnet"):
        if hp_model == "bagnet9":
            MODEL = bagnet9(DATASET, pretrained=hp_model_pretrained)
        elif hp_model == "bagnet17":
            MODEL = bagnet17(DATASET, pretrained=hp_model_pretrained)
        elif hp_model == "bagnet33":
            MODEL = bagnet33(DATASET, pretrained=hp_model_pretrained)
    elif hp_model.startswith("resnet"):
        if hp_model == "resnet18":
            MODEL = torchvision.models.resnet18(pretrained=hp_model_pretrained)
            MODEL.fc = torch.nn.Linear(512 * 1, DATASET.num_classes)
        elif hp_model == "resnet50":
            MODEL = torchvision.models.resnet50(pretrained=hp_model_pretrained)
            MODEL.fc = torch.nn.Linear(512 * 4, DATASET.num_classes)

    if MODEL is None:
        raise Exception(f"Unknown model: {hp_model}")

    return MODEL


def get_model_info() -> str:
    hp_model = get_hparam_value(HPARAMS.MODEL)
    hp_model_pretrained = get_hparam_value(HPARAMS.MODEL_PRETRAINED)
    return f"{hp_model}:{hp_model_pretrained}"


def get_dataset_info() -> str:
    if DATASET is None:
        get_dataset()
    ds = DATASET
    items_per_class = get_hparam_value(HPARAMS.DATA_MIN_ITEM_PER_CLASS)
    return f"Bodmas-{ds.img_shape[0]}x{ds.img_shape[1]}x{ds.img_color_channels}:{BATCH_SIZE}:{items_per_class}"


def get_train_dataset() -> torch.utils.data.Dataset:
    if not DATALOADER_LOADED:
        init_train_valid_loader()
    return TRAIN_DS


def get_val_dataset() -> torch.utils.data.Dataset:
    if not DATALOADER_LOADED:
        init_train_valid_loader()
    return VALID_DS


def get_train_loader() -> torch.utils.data.DataLoader:
    if not DATALOADER_LOADED:
        init_train_valid_loader()
    return TRAIN_LOADER


def get_val_loader() -> torch.utils.data.DataLoader:
    if not DATALOADER_LOADED:
        init_train_valid_loader()
    return VALID_LOADER


def get_logger() -> logging.Logger:
    l = Logger.get_logger('default')
    _info = l.info
    l.info = lambda msg: _info(f"[Main] {msg}")
    return l


def get_hparams() -> Dict[str, Union[int, float, bool, str]]:
    return {
        hp.name: get_hparam_value(hp) for hp in HPARAMS
    }


def target_class_translations() -> Dict[int, str]:
    if DATASET is None:
        get_dataset()
    return DATASET._data_index2class


# =======================================================
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# =======================================================


MODEL: torch.nn.Module = None
DATASET: ImgDataset = None
BATCH_SIZE: int = None

DATALOADER_LOADED = False
(TRAIN_DS, TRAIN_LOADER,
 VALID_DS, VALID_LOADER) = [None] * 4


# =======================================================
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# =======================================================

def get_dataset() -> ImgDataset:
    """depends on: -"""

    global DATASET, BATCH_SIZE
    if DATASET is not None:
        return DATASET

    DATASET = Datasets.BODMAS.value
    DATASET.filter_ground_truth(get_hparam_value(HPARAMS.DATA_MIN_ITEM_PER_CLASS))

    BATCH_SIZE = get_batch_size(DATASET)

    return DATASET


def get_batch_size(ds: ImgDataset):
    """depends on: -"""

    # model = get_model()
    # get_batch_size(model, torch.device("cpu"), (3, 300, 300), (Datasets.BODMAS.value.num_classes,), 1000, 8, 128)
    batch_size = get_hparam_value(HPARAMS.DATA_BATCH_SIZE)
    if ds.img_shape == (300, 300):
        if batch_size > 16:
            Logger.warning("[TrainDef] Batch size is too high for 300x300 image size")
            return 16
    elif ds.img_shape == (224, 224):
        if batch_size > 32:
            Logger.warning("[TrainDef] Batch size is too high for 224x224 image size")
            return 32
    elif ds.img_shape == (100, 100):
        if batch_size > 64:
            Logger.warning("[TrainDef] Batch size is too high for 30x30 image size")
            return 64

    return batch_size


def init_train_valid_loader():
    """depends on: DATASET, BATCH_SIZE"""

    global DATALOADER_LOADED
    global \
        TRAIN_DS, TRAIN_LOADER, \
        VALID_DS, VALID_LOADER

    if DATALOADER_LOADED:
        return

    if DATASET is None:
        get_dataset()

    files = os.listdir(paths.get_cg_image_classification_tb_log_dir())
    for file in files:
        if os.path.isdir(file) and get_model_info() in file and get_dataset_info() in file:
            Logger.warning(f"[TrainDef] Model and dataset already trained: {file}")

    (TRAIN_DS, TRAIN_LOADER,
     VALID_DS, VALID_LOADER) = \
        create_bodmas_train_val_loader(DATASET, batch_size=BATCH_SIZE)

    DATALOADER_LOADED = True
