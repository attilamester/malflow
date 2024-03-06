import logging
from typing import Dict, Union

import torch.utils.data

from core.processors.bagnet.dataset import Datasets
from core.processors.bagnet.dataset.dataloader import get_train_valid_dataset_sampler_loader
from core.processors.bagnet.hparams import HPARAMS, get_hparam_value
from core.processors.bagnet.nn_model import bagnet17, bagnet9, bagnet33
from util.logger import Logger


# =======================================================
# \\\\\\\\\\ Required functions for main.py /////////////
# see: https://github.com/attilamester/pytorch-examples/blob/feature/83/imagenet/main.py
# =======================================================

def get_model() -> torch.nn.Module:
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
    return f"Bagnet-{hp_model_bagnet}{hp_model_pretrained}"


def get_train_dataset() -> torch.utils.data.Dataset:
    return train_dataset


def get_val_dataset() -> torch.utils.data.Dataset:
    return valid_dataset


def get_train_loader() -> torch.utils.data.DataLoader:
    return train_loader


def get_val_loader() -> torch.utils.data.DataLoader:
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

(train_dataset, train_sampler, train_loader,
 valid_dataset, valid_sampler, valid_loader) = \
    get_train_valid_dataset_sampler_loader(Datasets.BODMAS.value, get_hparam_value(HPARAMS.DATA_MIN_ITEM_PER_CLASS))
