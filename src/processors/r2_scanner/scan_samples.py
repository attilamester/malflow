import os
from concurrent.futures import ThreadPoolExecutor
from typing import Type

from malflow.core.data import DatasetProvider
from malflow.core.data.bodmas import Bodmas
from malflow.core.model import CallGraph, CallGraphCompressed
from malflow.core.model.sample import Sample
from malflow.util import config
from malflow.util.logger import Logger
from processors.r2_scanner.create_dfs import create_callgraph_dfs, create_callgraph_function_blocks
from processors.r2_scanner.scan_instructions import extract_callgraph_instructions_stats
from processors.util import process_samples


def scan(dset: Type[DatasetProvider], cg: CallGraph):
    cg.scan()
    compressed = CallGraphCompressed(cg)
    compressed.dump_compressed(dset.get_dir_callgraphs())
    Logger.info(f">> Dumped cg: {cg.file_path}|{cg.md5}")


def scan_sample(dset: Type[DatasetProvider], sample: Sample):
    cg = CallGraph(sample.filepath, scan=False, verbose=False)
    md5 = cg.md5

    compressed_path = CallGraphCompressed.get_compressed_path(dset.get_dir_callgraphs(), md5)
    if not os.path.isfile(compressed_path):
        Logger.info(f">> No r2 found on disk: {md5}")
        scan(dset, cg)
        extract_callgraph_instructions_stats(dset, cg)
        create_callgraph_dfs(dset, cg, img_dims=[(30, 30), (100, 100), (224, 224), (300, 300)])
        create_callgraph_function_blocks(dset, cg, img_dims=[(30, 30), (100, 100), (224, 224), (300, 300)])
    else:
        Logger.info(f">> Already existing r2 found on disk: {md5}")


if __name__ == "__main__":
    config.load_env()
    process_samples(Bodmas, scan_sample, batch_size=1000, max_batches=None, pool=ThreadPoolExecutor(max_workers=8))
