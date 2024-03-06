import os
from dataclasses import dataclass
from enum import Enum
from typing import Type, List

from util import config
from util.logger import Logger
from util.misc import get_bagnet_folder
from util.validators import Validator

config.load_env(os.path.join(get_bagnet_folder(), "hparams.env"))


@dataclass
class Hyperparamer:
    type_: Type
    values: List


class HPARAMS(Enum):
    """
    First item is the default value
    """
    MODEL_BAGNET = Hyperparamer(int, [9, 17, 33])
    MODEL_PRETRAINED = Hyperparamer(bool, [False])
    DATA_MIN_ITEM_PER_CLASS = Hyperparamer(int, [100, 50, 200])
    DATA_BATCH_SIZE = Hyperparamer(int, [8, 16, 32])


def get_hparam_value(hparam: HPARAMS, custom_env_name: str = None):
    env_name = f"HPARAM_{hparam.name}" if custom_env_name is None else custom_env_name
    if hparam.value.type_ == int:
        validator = Validator.validate_int
    elif hparam.value.type_ == bool:
        validator = Validator.validate_bool
    else:
        raise Exception(f"Unknown hparam type: {hparam}")

    env_value = validator(os.environ.get(env_name, hparam.value.values[0]))
    Logger.info(f"Loaded hparam {hparam.name}={env_value}")
    return env_value
