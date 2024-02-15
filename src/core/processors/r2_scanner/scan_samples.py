import json
import os
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from core.data.bodmas import Bodmas
from core.model import CallGraph, CallGraphCompressed
from core.model.call_graph_image import CallGraphImage
from core.model.sample import Sample
from core.processors.util import process_samples
from util import config
from util.logger import Logger
from util.misc import dict_key_add


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
            extract_callgraph_instructions(cg)
            create_callgraph_image(cg)
        else:
            Logger.info(f">> Already existing r2 found on disk: {md5}")


def extract_callgraph_instructions(cg: CallGraph):
    instructions_path = os.path.join(Bodmas.get_dir_r2_scans(), f"{cg.md5}.instructions.json")
    instructions = {}
    for node in cg.nodes.values():
        for i in node.instructions:
            key = i.get_fmt()
            dict_key_add(instructions, key)

    with open(instructions_path, "w") as f:
        json.dump(instructions, f)


def create_callgraph_image(cg: CallGraph):
    image_path = os.path.join(Bodmas.get_dir_r2_scans(), f"{cg.md5}.png")
    cg_img = CallGraphImage(cg)
    np_pixels = cg_img.get_image()
    pil_image = Image.fromarray(np_pixels)
    pil_image.save(image_path)


if __name__ == "__main__":
    config.load_env()
    process_samples(Bodmas, scan_sample, batch_size=1000, max_batches=None,
                    pool=ThreadPoolExecutor(max_workers=8))
