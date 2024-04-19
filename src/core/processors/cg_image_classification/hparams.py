import os
from enum import Enum
from typing import Type, List

from core.processors.cg_image_classification.paths import get_cg_image_classification_env, \
    get_cg_image_classification_env_hparams
from util import config
from util.logger import Logger
from util.validators import Validator

config.load_env(get_cg_image_classification_env())


class HParamSpace:
    type_: Type
    values: List

    def __init__(self, type_, values):
        self.type_ = type_
        self.values = values


class HPARAMS(Enum):
    """
    First item is the default value
    """
    MODEL = HParamSpace(
        str, Validator.validate_list(os.environ["HPARAM_SPACE_MODEL"], str))
    MODEL_PRETRAINED = HParamSpace(
        bool, Validator.validate_list(os.environ["HPARAM_SPACE_MODEL_PRETRAINED"], Validator.validate_bool))
    DATA_MIN_ITEM_PER_CLASS = HParamSpace(
        int, Validator.validate_list(os.environ["HPARAM_SPACE_DATA_MIN_ITEM_PER_CLASS"], int))
    DATA_MAX_ITEM_PER_CLASS = HParamSpace(
        int, Validator.validate_list(os.environ["HPARAM_SPACE_DATA_MAX_ITEM_PER_CLASS"], int))
    DATA_BATCH_SIZE = HParamSpace(
        int, Validator.validate_list(os.environ["HPARAM_SPACE_DATA_BATCH_SIZE"], int))
    DATA_AUGM = HParamSpace(
        bool, Validator.validate_list(os.environ["HPARAM_SPACE_DATA_AUGM"], Validator.validate_bool))


def get_hparam_name(hparam: HPARAMS):
    return f"HPARAM_{hparam.name}"


def get_hparam_value(hparam: HPARAMS, custom_env_name: str = None):
    if hparam not in __hparams:
        env_name = get_hparam_name(hparam) if custom_env_name is None else custom_env_name
        if env_name not in os.environ:
            config.load_env(get_cg_image_classification_env_hparams())

        env_value = os.environ[env_name]

        if hparam.value.type_ == int:
            validator = Validator.validate_int
        elif hparam.value.type_ == bool:
            validator = Validator.validate_bool
        elif hparam.value.type_ == str:
            validator = str
        else:
            raise Exception(f"Unknown hparam type: {hparam}")

        env_value = validator(env_value)
        Logger.info(f"Loaded hparam {env_name}={env_value}")

        __hparams[hparam] = env_value

    return __hparams[hparam]


__hparams = {}
