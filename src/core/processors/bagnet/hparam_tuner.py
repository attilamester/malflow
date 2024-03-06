import itertools
import os
import random
import subprocess
from typing import Dict

from core.processors.bagnet.hparams import HPARAMS
from util.logger import Logger
from util.misc import get_bagnet_folder


def run_train_dockerless_gpu(hparam: Dict[str, int]):
    buff = ""
    for name, value in hparam.items():
        buff += f"{name}={value}\n"
    buff += "\n"
    with open(os.path.join(get_bagnet_folder(), "hparams.env"), "w") as f:
        f.write(buff)
        f.flush()
    Logger.info(f"Running train with HPARAMS: {hparam}")

    subprocess.run(["/bin/sh", "train_dockerless_gpu.sh", "tensorboard/log_dir"],
                   cwd=get_bagnet_folder())


def hparam_tuner():
    hparams_names = [hparam.name for hparam in HPARAMS]
    hparam_space = list(itertools.product(*[hparam.value.values for hparam in HPARAMS]))
    random.shuffle(hparam_space)
    Logger.info(f"Hparam space for {hparams_names}: \n<<\n{hparam_space}>>\n")
    for item in hparam_space:
        hparam = {name: value for name, value in zip(hparams_names, item)}
        try:
            run_train_dockerless_gpu(hparam)
        except Exception as e:
            Logger.warning(f"Could not run train with hparam: {hparam} [{e}]")


if __name__ == "__main__":
    hparam_tuner()
