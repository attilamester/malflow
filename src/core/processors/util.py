import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import wraps
from typing import List, Callable, Type, Union

from core.data import DatasetProvider
from core.model import CallGraph, CallGraphCompressed
from core.model.sample import Sample
from util.logger import Logger


def process_samples(dset: Type[DatasetProvider], fn: Callable[[Type[DatasetProvider], Sample], None],
                    batch_size: int = 1000,
                    max_batches: int = None, pool: Union[ThreadPoolExecutor, ProcessPoolExecutor] = None):
    sample: Sample
    b = 0
    batch = []
    for sample in dset.get_samples():
        batch.append(sample)
        if len(batch) == batch_size:
            b += 1
            process_sample_batch(dset, batch, b, fn, pool)
            batch = []

            if max_batches and b == max_batches:
                Logger.info(f"Max batches {max_batches} reached, stopping.")
                return

    if batch:
        b += 1
        process_sample_batch(dset, batch, b, fn, pool)
    if pool:
        pool.shutdown(wait=True)


def process_sample_batch(dset: Type[DatasetProvider], batch: List[Sample], batch_number: int, fn: Callable,
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
            fn(dset, sample)
            i += 1
            log_eta(ts, i)

    else:
        for res in pool.map(fn, [dset] * len(batch), batch):
            i += 1
            log_eta(ts, i)


def decorator_callgraph_processor(dset: Type[DatasetProvider] = None,
                                  skip_load_if: Callable[[Type[DatasetProvider], str], bool] = lambda x: False) \
        -> Callable[[Type[DatasetProvider]], Callable[[Sample], None]]:
    """
    Will load the callgraph from disk and pass it to the processor.
    :param dset:
    :return: Callable[[Sample], None]
    """

    def decorator(processor) -> Callable[[Sample], None]:
        @wraps(processor)
        def wrapped(sample: Sample):
            cg = CallGraph(sample.filepath, scan=False, verbose=False)
            md5 = cg.md5

            if skip_load_if and skip_load_if(dset, md5) is True:
                Logger.error(f"Skipping `{processor.__name__}` for {md5}")
                return

            compressed_path = CallGraphCompressed.get_compressed_path(dset.get_dir_callgraphs(), md5)
            if not os.path.isfile(compressed_path):
                raise Exception(f"No r2 found on disk: {md5} for {sample.filepath}")

            try:
                cg = CallGraphCompressed.load(compressed_path, verbose=True).decompress()
            except Exception as e:
                Logger.error(f"Could not load compressed callgraph: {e} [{md5} {sample.filepath}]")

            try:
                processor(dset, cg)
            except Exception as e:
                Logger.error(f"Could not execute processor: {e} [{md5} {sample.filepath}]")

        return wrapped

    return decorator
