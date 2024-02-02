import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

from core.data.bodmas import Bodmas
from core.model import CallGraph, CallGraphCompressed
from core.model.sample import Sample
from util import config
from util.logger import Logger


def scan(cg: CallGraph):
    cg.scan()
    compressed = CallGraphCompressed(cg)
    compressed.dump_compressed(Bodmas.get_dir_r2_scans())
    Logger.info(f">> Dumped cg of {cg.md5} to {Bodmas.get_dir_r2_scans()}")


def scan_sample(filepath: str, rescan=False):
    cg: CallGraph
    cg_compressed: CallGraphCompressed

    cg = CallGraph(filepath, scan=False, verbose=False)
    md5 = cg.md5

    if rescan:
        scan(cg)
    else:
        compressed_path = CallGraphCompressed.get_compressed_path(Bodmas.get_dir_r2_scans(), md5)
        if not os.path.isfile(compressed_path):
            Logger.error(f">> No r2 found on disk: {md5}")
            scan(cg)
        else:
            Logger.info(f">> Already existing r2 found on disk: {md5}")


def process_sample_batch(batch: List[Sample], batch_number: int):
    Logger.info(f"[Batch {batch_number}] {len(batch)} samples")
    i = 0
    ts = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as executor:
        for res in executor.map(scan_sample, [sample.filepath for sample in batch]):
            i += 1
            dt = time.perf_counter() - ts
            eta = round(len(batch) * dt / i - dt, 2)
            Logger.info(f"[Batch {batch_number}] {i} samples done, ETA: {eta}s")
        executor.shutdown(wait=True)


def process_bodmas_samples():
    sample: Sample
    b = 0
    batch = []
    for sample in Bodmas.get_samples():
        batch.append(sample)
        if len(batch) == 1000:
            b += 1
            process_sample_batch(batch, b)
            batch = []

    if batch:
        b += 1
        process_sample_batch(batch, b)


if __name__ == "__main__":
    config.load_env()
    process_bodmas_samples()
