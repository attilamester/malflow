import os

from util.misc import get_project_root


def get_cg_image_classification_folder():
    return os.path.join(get_project_root(), "src/core/processors/cg_image_classification")


def get_cg_image_classification_env():
    return os.path.join(get_cg_image_classification_folder(), ".env")


def get_cg_image_classification_env_hparams():
    return os.path.join(get_cg_image_classification_folder(), "hparams.env")


def get_cg_image_classification_tb_log_dir():
    return os.path.join(get_cg_image_classification_folder(), "tensorboard/log_dir")
