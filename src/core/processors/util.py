import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import List, Callable, Type, Union

from core.data import DatasetProvider
from core.model.sample import Sample
from util.logger import Logger


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