import os
from concurrent.futures import ProcessPoolExecutor

from core.data.bodmas import Bodmas
from core.model import CallGraph, CallGraphCompressed
from core.model.sample import Sample
from core.processors.r2_scanner.scan_samples import create_callgraph_dfs
from core.processors.util import process_samples
from util import config
from util.logger import Logger


def create_dfs(sample: Sample):
    cg = CallGraph(sample.filepath, scan=False, verbose=False)
    md5 = cg.md5

    compressed_path = CallGraphCompressed.get_compressed_path(Bodmas.get_dir_r2_scans(), md5)
    if not os.path.isfile(compressed_path):
        raise Exception(f"No r2 found on disk: {md5} for {sample.filepath}")

    try:
        cg = CallGraphCompressed.load(compressed_path, verbose=True).decompress()
        create_callgraph_dfs(cg)
    except Exception as e:
        Logger.error(f"Could not load compressed callgraph: {e} [{md5} {sample.filepath}]")


if __name__ == "__main__":
    config.load_env()
    process_samples(Bodmas, create_dfs, batch_size=1000, max_batches=None, pool=ProcessPoolExecutor(max_workers=8))
