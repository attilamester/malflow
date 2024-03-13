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
from core.processors.cg_image_classification.dataset import Datasets
from core.processors.cg_image_classification.dataset.dataloader import create_bodmas_train_val_loader
from core.processors.cg_image_classification.hparams import HPARAMS, get_hparam_value
from core.processors.cg_image_classification.nn_model import bagnet17, bagnet9, bagnet33
from util.logger import Logger


# =======================================================
# \\\\\\\\\\ Required functions for main.py /////////////
# see: https://github.com/attilamester/pytorch-examples/blob/feature/83/imagenet/main.py
# =======================================================

def get_model() -> torch.nn.Module:
    if not Datasets.BODMAS.value.initialized:
        init_train_valid_loader()

    hp_model = get_hparam_value(HPARAMS.MODEL)
    hp_model_pretrained = get_hparam_value(HPARAMS.MODEL_PRETRAINED)

    if hp_model.startswith("bagnet"):
        if hp_model == "bagnet9":
            return bagnet9(Datasets.BODMAS.value, pretrained=hp_model_pretrained)
        elif hp_model == "bagnet17":
            return bagnet17(Datasets.BODMAS.value, pretrained=hp_model_pretrained)
        elif hp_model == "bagnet33":
            return bagnet33(Datasets.BODMAS.value, pretrained=hp_model_pretrained)
    elif hp_model.startswith("resnet"):
        if hp_model == "resnet18":
            m = torchvision.models.resnet18(pretrained=hp_model_pretrained)
            m.fc = torch.nn.Linear(512 * 1, Datasets.BODMAS.value.num_classes)
            return m
        elif hp_model == "resnet50":
            m = torchvision.models.resnet50(pretrained=hp_model_pretrained)
            m.fc = torch.nn.Linear(512 * 4, Datasets.BODMAS.value.num_classes)
            return m

    raise Exception(f"Unknown model: {hp_model}")


def get_model_info() -> str:
    hp_model = get_hparam_value(HPARAMS.MODEL)
    hp_model_pretrained = get_hparam_value(HPARAMS.MODEL_PRETRAINED)
    return f"{hp_model}:{hp_model_pretrained}"


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
    if not Datasets.BODMAS.value.initialized:
        init_train_valid_loader()
    return train_dataset.dataset.data_index2class


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
        if os.path.isdir(file) and get_model_info() in file and get_dataset_info() in file:
            Logger.warning(f"Model and dataset already trained: {file}")

    (train_dataset, train_loader,
     valid_dataset, valid_loader) = \
        create_bodmas_train_val_loader(Datasets.BODMAS.value,
                                       items_per_class=get_hparam_value(HPARAMS.DATA_MIN_ITEM_PER_CLASS),
                                       batch_size=get_hparam_value(HPARAMS.DATA_BATCH_SIZE))

    dataset_loaded = True
