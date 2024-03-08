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

from core.processors.cg_image_classification import paths
from core.processors.cg_image_classification.dataset import Datasets
from core.processors.cg_image_classification.dataset.dataloader import get_train_valid_dataset_sampler_loader
from core.processors.cg_image_classification.hparams import HPARAMS, get_hparam_value
from core.processors.cg_image_classification.nn_model import bagnet17, bagnet9, bagnet33
from util.logger import Logger


# =======================================================
# \\\\\\\\\\ Required functions for main.py /////////////
# see: https://github.com/attilamester/pytorch-examples/blob/feature/83/imagenet/main.py
# =======================================================

def get_model() -> torch.nn.Module:
    if not dataset_loaded:
        init_train_valid_loader()

    hp_model_bagnet = get_hparam_value(HPARAMS.MODEL_BAGNET)
    hp_model_pretrained = get_hparam_value(HPARAMS.MODEL_PRETRAINED)

    if hp_model_bagnet == 9:
        return bagnet9(Datasets.BODMAS.value, pretrained=hp_model_pretrained)
    elif hp_model_bagnet == 17:
        return bagnet17(Datasets.BODMAS.value, pretrained=hp_model_pretrained)
    elif hp_model_bagnet == 33:
        return bagnet33(Datasets.BODMAS.value, pretrained=hp_model_pretrained)
    else:
        raise Exception(f"Unknown model: {hp_model_bagnet}")


def get_model_info() -> str:
    hp_model_bagnet = get_hparam_value(HPARAMS.MODEL_BAGNET)
    hp_model_pretrained = get_hparam_value(HPARAMS.MODEL_PRETRAINED)
    return f"Bagnet-{hp_model_bagnet}:{hp_model_pretrained}"


def get_dataset_info() -> str:
    ds = Datasets.BODMAS.value
    items_per_class = get_hparam_value(HPARAMS.DATA_MIN_ITEM_PER_CLASS)
    batch_size = get_hparam_value(HPARAMS.DATA_BATCH_SIZE)
    return f"Bodmas-{ds.img_shape[0]}x{ds.img_shape[1]}x{ds.img_color_channels}:{batch_size}:{items_per_class}"


def get_train_dataset() -> torch.utils.data.Dataset:
    if not dataset_loaded:
        init_train_valid_loader()
    return train_dataset


def get_val_dataset() -> torch.utils.data.Dataset:
    if not dataset_loaded:
        init_train_valid_loader()
    return valid_dataset


def get_train_loader() -> torch.utils.data.DataLoader:
    if not dataset_loaded:
        init_train_valid_loader()
    return train_loader


def get_val_loader() -> torch.utils.data.DataLoader:
    if not dataset_loaded:
        init_train_valid_loader()
    return valid_loader


def get_logger() -> logging.Logger:
    return Logger.get_logger('default')


def get_hparams() -> Dict[str, Union[int, float, bool, str]]:
    return {
        hp.name: get_hparam_value(hp) for hp in HPARAMS
    }


def target_class_translations() -> Dict[int, str]:
    return train_dataset.index_family


# =======================================================
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# =======================================================

dataset_loaded = False
(train_dataset, train_sampler, train_loader,
 valid_dataset, valid_sampler, valid_loader) = [None] * 6


def init_train_valid_loader():
    global dataset_loaded
    global \
        train_dataset, train_sampler, train_loader, \
        valid_dataset, valid_sampler, valid_loader

    if dataset_loaded:
        return

    files = os.listdir(paths.get_cg_image_classification_tb_log_dir())
    for file in files:
        if get_model_info() in file and get_dataset_info() in file:
            Logger.warning(f"Exiting | Model and dataset already trained: {file}")
            exit(0)
    (train_dataset, train_loader,
     valid_dataset, valid_loader) = \
        get_train_valid_dataset_sampler_loader(Datasets.BODMAS.value,
                                               items_per_class=get_hparam_value(HPARAMS.DATA_MIN_ITEM_PER_CLASS),
                                               batch_size=get_hparam_value(HPARAMS.DATA_BATCH_SIZE))

    dataset_loaded = True
