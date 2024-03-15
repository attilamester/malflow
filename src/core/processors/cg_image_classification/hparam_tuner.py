import itertools
import os
import random
import subprocess
from typing import Dict

import pandas as pd

from core.processors.cg_image_classification.hparams import HPARAMS, get_hparam_name
from core.processors.cg_image_classification.paths import get_cg_image_classification_folder, \
    get_cg_image_classification_tb_log_dir
from util.logger import Logger


def run_train_dockerless_gpu(hparam: Dict[HPARAMS, int]):
    hparam_envs = {}
    for hp, value in hparam.items():
        hparam_envs[get_hparam_name(hp)] = str(value)

    Logger.info(f"[HParam Tuner] Running train with HPARAMS: \n\t{hparam_envs}")

    # --multiprocessing-distributed --rank=0 --world-size=1 --dist-url='tcp://localhost:29500'
    subprocess.run(["/bin/sh", "train_dockerless.sh", get_cg_image_classification_tb_log_dir()],
                   cwd=get_cg_image_classification_folder(), env={**dict(os.environ), **hparam_envs})


def hparam_tuner():
    hparams = [hparam for hparam in HPARAMS]
    hparam_space = list(itertools.product(*[hparam.value.values for hparam in HPARAMS]))
    df_hparam_space = pd.DataFrame(hparam_space, columns=[hparam.name for hparam in hparams])
    random.shuffle(hparam_space)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    Logger.info(f"[HParam Tuner] Hparam space: \n{df_hparam_space}")

    for item in hparam_space:
        hparam = {hp: value for hp, value in zip(hparams, item)}
        try:
            run_train_dockerless_gpu(hparam)
        except Exception as e:
            Logger.warning(f"[HParam Tuner] Could not run train with hparam: {hparam} [{e}]")


if __name__ == "__main__":
    hparam_tuner()
