import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import List, Callable, Type, Union

from core.data import DatasetProvider
from core.data.bodmas import Bodmas
from core.model import CallGraph, CallGraphCompressed
from core.model.sample import Sample
from util import config
from util.logger import Logger
from util.misc import dict_key_add


# ======================
# General methods
# ======================

def process_samples(dset: Type[DatasetProvider], fn: Callable[[Sample], None], batch_size: int = 1000,
                    max_batches: int = None, pool: Union[ThreadPoolExecutor, ProcessPoolExecutor] = None):
    sample: Sample
    b = 0
    batch = []
    for sample in dset.get_samples():
        batch.append(sample)
        if len(batch) == batch_size:
            b += 1
            process_sample_batch(batch, b, fn, pool)
            batch = []

            if max_batches and b == max_batches:
                Logger.info(f"Max batches {max_batches} reached, stopping.")
                return

    if batch:
        b += 1
        process_sample_batch(batch, b, fn, pool)
    if pool:
        pool.shutdown(wait=True)


def process_sample_batch(batch: List[Sample], batch_number: int, fn: Callable,
                         pool: Union[ThreadPoolExecutor, ProcessPoolExecutor] = None):
    Logger.info(f"[Batch {batch_number}] {len(batch)} samples")
    i = 0
    ts = time.perf_counter()

    def log_eta(start_ts, i):
        dt = time.perf_counter() - start_ts
        eta = round((len(batch) * dt) / i - dt, 2)
        Logger.info(f"[Batch {batch_number}] {i} samples done, ETA: {eta}s")

    if pool is None:
        for sample in batch:
            fn(sample)
            i += 1
            log_eta(ts, i)

    else:
        for res in pool.map(fn, batch):
            i += 1
            log_eta(ts, i)


# ======================
# Scanning the binaries
# ======================

def scan(cg: CallGraph):
    cg.scan()
    compressed = CallGraphCompressed(cg)
    compressed.dump_compressed(Bodmas.get_dir_r2_scans())
    Logger.info(f">> Dumped cg of {cg.md5} to {Bodmas.get_dir_r2_scans()}")


def scan_sample(sample: Sample, rescan=False):
    cg = CallGraph(sample.filepath, scan=False, verbose=False)
    md5 = cg.md5

    if rescan:
        scan(cg)
    else:
        compressed_path = CallGraphCompressed.get_compressed_path(Bodmas.get_dir_r2_scans(), md5)
        if not os.path.isfile(compressed_path):
            Logger.info(f">> No r2 found on disk: {md5}")
            scan(cg)
        else:
            Logger.info(f">> Already existing r2 found on disk: {md5}")


# ======================
# Processing the callgraphs
# ======================


def extract_sample_instructions(sample: Sample):
    cg = CallGraph(sample.filepath, scan=False, verbose=False)
    md5 = cg.md5

    instructions_path = os.path.join(Bodmas.get_dir_r2_scans(), f"{md5}.instructions.json")
    if os.path.isfile(instructions_path):
        return

    compressed_path = CallGraphCompressed.get_compressed_path(Bodmas.get_dir_r2_scans(), md5)
    if not os.path.isfile(compressed_path):
        raise Exception(f"No r2 found on disk: {md5} for {sample.filepath}")

    try:
        cg = CallGraphCompressed.load(compressed_path, verbose=True).decompress()
    except Exception as e:
        Logger.error(f"Could not load compressed callgraph: {e} [{md5} {sample.filepath}]")

    instructions = {}
    for node in cg.nodes.values():
        for i in node.instructions:
            key = i.get_fmt()
            dict_key_add(instructions, key)
    with open(instructions_path, "w") as f:
        json.dump(instructions, f)


if __name__ == "__main__":
    config.load_env()
    process_samples(Bodmas, extract_sample_instructions, batch_size=1000, max_batches=None,
                    pool=ProcessPoolExecutor(max_workers=12))
