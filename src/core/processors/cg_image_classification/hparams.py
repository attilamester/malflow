import os
from dataclasses import dataclass
from enum import Enum
from typing import Type, List

from core.processors.cg_image_classification.paths import get_cg_image_classification_env, \
    get_cg_image_classification_folder
from util import config
from util.logger import Logger
from util.validators import Validator

config.load_env(get_cg_image_classification_env())


@dataclass
class HParamSpace:
    type_: Type
    values: List


class HPARAMS(Enum):
    """
    First item is the default value
    """
    MODEL_BAGNET = HParamSpace(int, [9, 17, 33])
    MODEL_PRETRAINED = HParamSpace(bool, [False])
    DATA_MIN_ITEM_PER_CLASS = HParamSpace(
        int, Validator.validate_list(os.environ["HPARAM_SPACE_DATA_MIN_ITEM_PER_CLASS"], int))
    DATA_BATCH_SIZE = HParamSpace(
        int, Validator.validate_list(os.environ["HPARAM_SPACE_DATA_BATCH_SIZE"], int))


def get_hparam_name(hparam: HPARAMS):
    return f"HPARAM_{hparam.name}"


def get_hparam_value(hparam: HPARAMS, custom_env_name: str = None):
    if hparam not in __hparams:
        config.load_env(os.path.join(get_cg_image_classification_folder(), "hparams.env"))
        env_name = get_hparam_name(hparam) if custom_env_name is None else custom_env_name
        if hparam.value.type_ == int:
            validator = Validator.validate_int
        elif hparam.value.type_ == bool:
            validator = Validator.validate_bool
        else:
            raise Exception(f"Unknown hparam type: {hparam}")

        env_value = validator(os.environ[env_name])
        Logger.info(f"Loaded hparam {env_name}={env_value}")

        __hparams[hparam] = env_value

    return __hparams[hparam]


__hparams = {}
