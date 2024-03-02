import torch.utils.data

from core.processors.bagnet.dataset import Datasets
from core.processors.bagnet.dataset.dataloader import get_train_valid_dataset_sampler_loader
from core.processors.bagnet.nn_model import bagnet17


# =======================================================
# \\\\\\\\\\ Required functions for main.py /////////////
# see: https://github.com/attilamester/pytorch-examples/blob/feature/83/imagenet/main.py
# =======================================================

def get_model() -> torch.nn.Module:
    return bagnet17(Datasets.BODMAS.value)


def get_train_dataset() -> torch.utils.data.Dataset:
    return train_dataset


def get_val_dataset() -> torch.utils.data.Dataset:
    return valid_dataset


def get_train_loader() -> torch.utils.data.DataLoader:
    return train_loader


def get_val_loader() -> torch.utils.data.DataLoader:
    return valid_loader


# =======================================================
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# =======================================================

(train_dataset, train_sampler, train_loader,
 valid_dataset, valid_sampler, valid_loader) = get_train_valid_dataset_sampler_loader(Datasets.BODMAS.value)
