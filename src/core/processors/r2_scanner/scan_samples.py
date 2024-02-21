import os
from concurrent.futures import ThreadPoolExecutor

from core.data.bodmas import Bodmas
from core.model import CallGraph, CallGraphCompressed
from core.model.sample import Sample
from core.processors.r2_scanner.create_images import create_callgraph_image
from core.processors.r2_scanner.scan_instructions import extract_callgraph_instructions
from core.processors.util import process_samples
from util import config
from util.logger import Logger


def scan(cg: CallGraph):
    cg.scan()
    compressed = CallGraphCompressed(cg)
    compressed.dump_compressed(Bodmas.get_dir_callgraphs())
    Logger.info(f">> Dumped cg of {cg.md5} to {Bodmas.get_dir_callgraphs()}")


def scan_sample(sample: Sample, rescan=False):
    cg = CallGraph(sample.filepath, scan=False, verbose=False)
    md5 = cg.md5

    if rescan:
        scan(cg)
    else:
        compressed_path = CallGraphCompressed.get_compressed_path(Bodmas.get_dir_callgraphs(), md5)
        if not os.path.isfile(compressed_path):
            Logger.info(f">> No r2 found on disk: {md5}")
            scan(cg)
            extract_callgraph_instructions(Bodmas, cg)
            create_callgraph_image(Bodmas, cg)
        else:
            Logger.info(f">> Already existing r2 found on disk: {md5}")


if __name__ == "__main__":
    config.load_env()
    process_samples(Bodmas, scan_sample, batch_size=1000, max_batches=None,
                    pool=ThreadPoolExecutor(max_workers=8))
