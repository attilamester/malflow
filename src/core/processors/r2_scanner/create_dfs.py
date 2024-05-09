import os
import pickle
from typing import Type

from PIL import Image

from core.data import DatasetProvider
from core.model import CallGraph
from core.model.call_graph_image import CallGraphImage
from core.processors.r2_scanner.paths import get_path_image, get_path_instructions_dfs, \
    get_path_instructions_function_blocks
from core.processors.util import decorator_callgraph_processor
from util import config
from util.compression import BrotliCompressor

COMPRESSOR = BrotliCompressor(4)


def create_callgraph_dfs(dset: Type[DatasetProvider], cg: CallGraph, img_dims=None):
    if not img_dims:
        img_dims = [(30, 30)]
    else:
        img_dims = sorted(img_dims, key=lambda x: x[0] * x[1])

    for allow_multiple_visits in [True]:
        for store_call in [True]:
            instructions_path = get_path_instructions_dfs(dset, cg.md5, allow_multiple_visits, store_call)
            instructions = cg.DFS_instructions(max_instructions=img_dims[-1][0] * img_dims[-1][1],
                                               allow_multiple_visits=allow_multiple_visits,
                                               store_call=store_call)
            with open(instructions_path, "wb") as f:
                f.write(COMPRESSOR.compress(pickle.dumps([i.compress() for i in instructions])))

            if img_dims:
                pixels = [CallGraphImage.encode_instruction_rgb(i) for i in instructions]
                for dim in img_dims:
                    image_path = get_path_image(dset, cg.md5, dim, allow_multiple_visits, store_call)
                    np_pixels = CallGraphImage.get_image_from_pixels(dim, pixels)
                    pil_image = Image.fromarray(np_pixels)
                    pil_image.save(image_path)


def create_callgraph_function_blocks(dset: Type[DatasetProvider], cg: CallGraph, img_dims=None):
    if not img_dims:
        img_dims = [(30, 30)]
    else:
        img_dims = sorted(img_dims, key=lambda x: x[0] * x[1])

    instructions = [node.instructions for node in cg.DFS()]
    instructions_to_pickle = [[i.compress() for i in node] for node in instructions]
    instructions_path = get_path_instructions_function_blocks(dset, cg.md5)
    with open(instructions_path, "wb") as f:
        f.write(COMPRESSOR.compress(pickle.dumps(instructions_to_pickle)))

    instructions = [i for node in instructions for i in node]
    pixels = [CallGraphImage.encode_instruction_rgb(i) for i in instructions]
    for dim in img_dims:
        image_path = get_path_image(dset, cg.md5, dim, True, True, "_function_blocks")
        np_pixels = CallGraphImage.get_image_from_pixels(dim, pixels)
        pil_image = Image.fromarray(np_pixels)
        pil_image.save(image_path)


@decorator_callgraph_processor(skip_load_if=lambda dset, md5: os.path.isfile(
    get_path_image(dset, md5, (300, 300), False, False)))
def create_dfs(dset: Type[DatasetProvider], cg: CallGraph):
    create_callgraph_dfs(dset, cg, img_dims=[(30, 30), (100, 100), (224, 224), (300, 300)])


@decorator_callgraph_processor(skip_load_if=lambda dset, md5: os.path.isfile(
    get_path_image(dset, md5, (224, 224), False, False)))
def create_function_blocks(dset: Type[DatasetProvider], cg: CallGraph):
    create_callgraph_function_blocks(dset, cg, img_dims=[(30, 30), (100, 100), (224, 224)])


if __name__ == "__main__":
    config.load_env()
    # process_samples(Bodmas, create_dfs, batch_size=1000, max_batches=None, pool=ProcessPoolExecutor(max_workers=8))
    # process_samples(Bodmas, create_function_blocks, batch_size=1000, max_batches=None,
    #                 pool=ProcessPoolExecutor(max_workers=8))
