import itertools
import os
import subprocess
from typing import Dict

from core.processors.bagnet.hparams import HPARAMS
from util.logger import Logger
from util.misc import get_bagnet_folder


def run_train_native(hparam: Dict[str, int]):
    buff = ""
    for name, value in hparam.items():
        buff += f"{name}={value}\n"
    buff += "\n"
    with open(os.path.join(get_bagnet_folder(), "hparams.env"), "w") as f:
        f.write(buff)
        f.flush()
    Logger.info(f"Running train with HPARAMS: \n<<\n{buff}>>")

    subprocess.run(["/bin/sh", "train_dockerless_gpu.sh", "tensorboard/log_dir"],
                   cwd=get_bagnet_folder())


def hparam_tuner():
    hparams_names = [hparam.name for hparam in HPARAMS]
    for item in itertools.product(*[hparam.value for hparam in HPARAMS]):
        hparam = {name: value for name, value in zip(hparams_names, item)}
        try:
            run_train_native(hparam)
        except Exception as e:
            Logger.warning(f"Could not run train with hparam: {hparam} [{e}]")


if __name__ == "__main__":
    hparam_tuner()
