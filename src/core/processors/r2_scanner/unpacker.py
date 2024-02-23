import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Type

from core.data import DatasetProvider
from core.data.bodmas import Bodmas, BodmasUnpacked
from core.model import CallGraph
from core.model.sample import Sample
from core.processors.r2_scanner.create_dfs import create_callgraph_dfs
from core.processors.r2_scanner.paths import get_path_image
from core.processors.r2_scanner.scan_samples import scan_sample
from core.processors.util import process_samples, decorator_callgraph_processor, decorator_sample_processor
from util import config
from util.logger import Logger


@decorator_callgraph_processor(BodmasUnpacked, skip_load_if=lambda dset, md5: os.path.isfile(
    get_path_image(dset, md5, (300, 300), True, True)))
def create_dfs(dset: Type[DatasetProvider], cg: CallGraph):
    create_callgraph_dfs(dset, cg, img_dims=[(30, 30), (100, 100), (224, 224), (300, 300)])


def unpack_sample(sample: Sample):
    dset = Bodmas
    dset_unpacked = BodmasUnpacked
    die_info_path = os.path.join(dset.get_dir_info(), "die", os.path.basename(sample.filepath) + ".csv")

    if not os.path.isfile(die_info_path):
        Logger.info(f"No die info found for sample {sample.md5} {sample.sha256}")
        return

    packed = False
    with open(die_info_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Packer;"):
                tokens = line.split(";")
                packer_name = tokens[1].strip()
                packer_info = tokens[4].replace("Packer:", "").strip()
                packed = True
    if not packed:
        Logger.info(f"Sample not packed: {sample.md5} {sample.sha256}")
        return

    try:
        subprocess.run(["unipacker", "--dest", dset_unpacked.get_dir_samples(), sample.filepath],
                       capture_output=True)
    except Exception as e:
        Logger.error(f"Error unpacking {sample.filepath}: {e}")


@decorator_sample_processor(BodmasUnpacked)
def scan_unpacked_bodmas_sample(dset: Type[DatasetProvider], sample: Sample):
    scan_sample(dset, sample)


if __name__ == "__main__":
    config.load_env()
    Logger.set_file_logging("unpacker")
    process_samples(Bodmas, unpack_sample, batch_size=1000, max_batches=None, pool=ThreadPoolExecutor(max_workers=8))
    process_samples(BodmasUnpacked, scan_unpacked_bodmas_sample, batch_size=1000, max_batches=None,
                    pool=ThreadPoolExecutor(max_workers=8))
