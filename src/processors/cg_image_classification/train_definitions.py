"""
This is the entry point of the import process -- imported from main.py
Ensure the envs are loaded.
"""
from core.processors.cg_image_classification.paths import get_cg_image_classification_env
from util import config
from util.validators import Validator

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
from core.processors.cg_image_classification.nn_model import bagnet17, bagnet9, bagnet33, SimpleCNN
from core.processors.cg_image_classification.nn_model.resnet1d import ResNet1D
from util.logger import Logger


# =======================================================
# \\\\\\\\\\ Required functions for main.py /////////////
# see: https://github.com/attilamester/pytorch-examples/blob/feature/83/imagenet/main.py
# =======================================================

def get_model(debug: bool = False) -> torch.nn.Module:
    """depends on: DATASET"""

    global MODEL, DATASET

    if MODEL is not None:
        return MODEL

    if DATASET is None:
        get_dataset()

    hp_model = get_hparam_value(HPARAMS.MODEL)
    hp_model_pretrained = get_hparam_value(HPARAMS.MODEL_PRETRAINED)
    hp_model_kernel_rowwise = get_hparam_value(HPARAMS.MODEL_KERNEL_ROWWISE)

    if hp_model.startswith("bagnet"):
        if hp_model == "bagnet9":
            MODEL = bagnet9(DATASET, pretrained=hp_model_pretrained, debug=debug,
                            kernel_only_rowwise=hp_model_kernel_rowwise)
        elif hp_model == "bagnet17":
            MODEL = bagnet17(DATASET, pretrained=hp_model_pretrained, debug=debug,
                             kernel_only_rowwise=hp_model_kernel_rowwise)
        elif hp_model == "bagnet33":
            MODEL = bagnet33(DATASET, pretrained=hp_model_pretrained, debug=debug,
                             kernel_only_rowwise=hp_model_kernel_rowwise)
    elif hp_model.startswith("resnet"):
        if hp_model == "resnet18":
            weights = None if not hp_model_pretrained else torchvision.models.ResNet18_Weights.IMAGENET1K_V1
            MODEL = torchvision.models.resnet18(weights=weights)
            MODEL.fc = torch.nn.Linear(512 * 1, DATASET.num_classes)
        elif hp_model == "resnet50":
            weights = None if not hp_model_pretrained else torchvision.models.ResNet50_Weights.IMAGENET1K_V2
            MODEL = torchvision.models.resnet50(weights=weights)
            MODEL.fc = torch.nn.Linear(512 * 4, DATASET.num_classes)
        elif hp_model == "resnet1d":
            MODEL = ResNet1D(3, 64, 9, 1, 1, 4, DATASET.num_classes)
    elif hp_model.startswith("alexnet"):
        weights = None if not hp_model_pretrained else torchvision.models.AlexNet_Weights.IMAGENET1K_V1
        MODEL = torchvision.models.alexnet(weights=weights)
        MODEL.classifier[-1] = torch.nn.Linear(MODEL.classifier[-1].in_features, DATASET.num_classes)
    elif hp_model.startswith("densenet"):
        if hp_model == "densenet121":
            weights = None if not hp_model_pretrained else torchvision.models.DenseNet121_Weights.IMAGENET1K_V1
            MODEL = torchvision.models.densenet121(weights=weights)
            MODEL.classifier = torch.nn.Linear(MODEL.classifier.in_features, DATASET.num_classes)
        elif hp_model == "densenet161":
            weights = None if not hp_model_pretrained else torchvision.models.DenseNet161_Weights.IMAGENET1K_V1
            MODEL = torchvision.models.densenet161(weights=weights)
            MODEL.classifier = torch.nn.Linear(MODEL.classifier.in_features, DATASET.num_classes)
    elif hp_model.startswith("efficientnet"):
        weights = None if not hp_model_pretrained else torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        MODEL = torchvision.models.efficientnet_v2_s(weights=weights)
        MODEL.classifier[1] = torch.nn.Linear(MODEL.classifier[1].in_features, DATASET.num_classes)
    elif hp_model.startswith("googlenet"):
        weights = None if not hp_model_pretrained else torchvision.models.GoogLeNet_Weights.IMAGENET1K_V1
        MODEL = torchvision.models.googlenet(weights=weights)
        MODEL.fc = torch.nn.Linear(MODEL.fc.in_features, DATASET.num_classes)
    # elif hp_model.startswith("inception"):
    #     weights = None if not hp_model_pretrained else torchvision.models.Inception_V3_Weights.IMAGENET1K_V1
    #     MODEL = torchvision.models.inception_v3(weights=weights)
    #     MODEL.AuxLogits = InceptionAux(768, DATASET.num_classes)
    #     MODEL.fc = torch.nn.Linear(2048, DATASET.num_classes)
    elif hp_model.startswith("mobilenet"):
        weights = None if not hp_model_pretrained else torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        MODEL = torchvision.models.mobilenet_v3_small(weights=weights)
        MODEL.classifier[-1] = torch.nn.Linear(MODEL.classifier[-1].in_features, DATASET.num_classes)
    elif hp_model.startswith("vgg"):
        weights = None if not hp_model_pretrained else torchvision.models.VGG11_Weights.IMAGENET1K_V1
        MODEL = torchvision.models.vgg11(weights=weights)
        MODEL.classifier[-1] = torch.nn.Linear(4096, DATASET.num_classes)

    elif hp_model.startswith("simplecnn"):
        MODEL = SimpleCNN(DATASET, 32, dropout=0.5)

    if MODEL is None:
        raise Exception(f"Unknown model: {hp_model}")

    return MODEL


def get_model_info() -> str:
    hp_model = get_hparam_value(HPARAMS.MODEL)
    hp_model_pretrained = get_hparam_value(HPARAMS.MODEL_PRETRAINED)
    pretrained = "pretr" if hp_model_pretrained else "nopre"
    return f"{hp_model}:{pretrained}"


def get_dataset_info() -> str:
    if DATASET is None:
        get_dataset()
    ds = DATASET
    items_per_class = get_hparam_value(HPARAMS.DATA_MIN_ITEM_PER_CLASS)
    augm = "augmen" if get_hparam_value(HPARAMS.DATA_AUGM) else "noaugm"

    return f"Bodmas-{ds.get_img_dir_name(0, 8)}:{ds.img_shape[0]}x{ds.img_shape[1]}x{ds.img_color_channels}:{BATCH_SIZE}:{items_per_class}:{augm}"


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
    if DATASET is None:
        get_dataset()

    return {
        **{hp.name: get_hparam_value(hp) for hp in HPARAMS},
        "DATA_IMG_SHAPE": f"{DATASET.img_shape[0]}x{DATASET.img_shape[1]}",
        "DATA_NUM_CLASSES": get_dataset().num_classes
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


def get_model_requirements() -> Dict:
    """
    https://pytorch.org/vision/main/models.html
    """
    req = {}
    hp_model = get_hparam_value(HPARAMS.MODEL)
    if "alexnet" in hp_model:
        req["min_size"] = (63, 63)
    elif "vgg" in hp_model:
        req["min_size"] = (32, 32)

    return req


def get_dataset() -> ImgDataset:
    """depends on: -"""

    global DATASET, BATCH_SIZE
    if DATASET is not None:
        return DATASET

    DATASET = Datasets.BODMAS.value
    DATASET.filter_ground_truth(get_hparam_value(HPARAMS.DATA_MIN_ITEM_PER_CLASS),
                                get_hparam_value(HPARAMS.DATA_AUGM),
                                get_hparam_value(HPARAMS.DATA_MAX_ITEM_PER_CLASS))

    BATCH_SIZE = get_batch_size(DATASET)

    return DATASET


def get_batch_size(ds: ImgDataset):
    """depends on: -"""

    # model = get_model()
    # get_batch_size(model, torch.device("cpu"), (3, 300, 300), (Datasets.BODMAS.value.num_classes,), 1000, 8, 128)
    batch_size = get_hparam_value(HPARAMS.DATA_BATCH_SIZE)
    for shape, max_batch in [((300, 300), 16), ((224, 224), 32), ((100, 100), 64)]:
        if ds.img_shape == shape and batch_size > max_batch:
            Logger.warning(
                f"[TrainDef] HPARAM batch_size ({batch_size}) is too large for {shape} images. Using {max_batch}.")
            return max_batch

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
            if Validator.validate_bool(os.environ.get("TRAIN_SKIP_EXISTING", "t")):
                Logger.warning(f"[TrainDef] Skipping training")
                exit(0)

    (TRAIN_DS, TRAIN_LOADER,
     VALID_DS, VALID_LOADER) = \
        create_bodmas_train_val_loader(DATASET, batch_size=BATCH_SIZE, model_requirements=get_model_requirements())

    DATALOADER_LOADED = True


def debug_bagnet():
    import numpy as np
    init_train_valid_loader()
    loader = get_train_loader()

    for model in [
        bagnet9(DATASET, pretrained=False, debug=True),
        # bagnet17(DATASET, pretrained=False, debug=True),
        # bagnet33(DATASET, pretrained=False, debug=True)
    ]:
        np_arr = np.zeros((1, 3, 100), dtype=np.float32)
        np_arr[0, :, 0, 0] = 1
        images = torch.tensor(np_arr)
        # (images, target) = next(iter(loader))
        output = model(images)
        # print(output)
        # print(model)
