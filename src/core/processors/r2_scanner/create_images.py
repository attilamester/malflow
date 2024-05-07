import json
import os
from typing import Tuple, Type, List

import numpy as np
from PIL import Image

from core.data import DatasetProvider
from core.data.bodmas import Bodmas
from core.model import CallGraph
from core.model.call_graph_image import CallGraphImage, InstructionEncoder, InstructionEncoderMnemonicPrefixBnd
from core.model.sample import Sample
from core.processors.r2_scanner.paths import get_path_imageinfo, get_path_image, get_path_instructions_dfs
from core.processors.util import decorator_callgraph_processor
from helpers.readme.load_instruction import load_instruction_pickle
from util import config
from util.logger import Logger
from util.misc import dict_key_inc, dict_key_add, list_stats


def create_callgraph_image(dset: Type[DatasetProvider], cg: CallGraph, dim: Tuple[int, int] = (512, 512)):
    info_path = get_path_imageinfo(dset, cg.md5)
    cg_img = CallGraphImage(cg)

    info = {"configs": {}}
    for allow_multiple_visits in [True]:
        for store_call in [True]:
            image_path = get_path_image(dset, cg.md5, dim, allow_multiple_visits, store_call)
            np_pixels, original_size = cg_img.get_image(dim, allow_multiple_visits=allow_multiple_visits,
                                                        store_call=store_call)
            pil_image = Image.fromarray(np_pixels)
            pil_image.save(image_path)
            info["configs"][f"{allow_multiple_visits}_{store_call}"] = original_size

    with open(info_path, "w") as f:
        json.dump(info, f)


def create_callgraph_image_on_dfs_file(dset: Type[DatasetProvider], sample: Sample, img_dims: List[Tuple[int, int]],
                                       instruction_encoder: Type[InstructionEncoder]):
    dfs_path = get_path_instructions_dfs(dset, sample.md5)
    if not os.path.isfile(dfs_path):
        Logger.error(f"No DFS file found for {sample.filepath}")
        return

    should_scan = False
    for dim in img_dims:
        image_path = get_path_image(dset, sample.md5, dim, subdir=f"_{instruction_encoder.__name__}")
        if not os.path.isfile(image_path):
            should_scan = True
            break

    if not should_scan:
        return

    instructions = load_instruction_pickle(dfs_path)
    pixels = [instruction_encoder.encode(i) for i in instructions]
    for dim in img_dims:
        image_path = get_path_image(dset, sample.md5, dim, subdir=f"_{instruction_encoder.__name__}")
        np_pixels = CallGraphImage.get_image_from_pixels(dim, pixels)
        pil_image = Image.fromarray(np_pixels)
        pil_image.save(image_path)


@decorator_callgraph_processor(Bodmas, skip_load_if=lambda dset, md5: os.path.isfile(get_path_imageinfo(dset, md5)))
def create_image(dset: Type[DatasetProvider], cg: CallGraph):
    create_callgraph_image(dset, cg, dim=(512, 512))


def create_image_on_dfs_files(dset: Type[DatasetProvider], sample: Sample):
    create_callgraph_image_on_dfs_file(dset, sample, img_dims=[(30, 30), (100, 100), (224, 224)],
                                       instruction_encoder=InstructionEncoderMnemonicPrefixBnd)


def create_image_on_simple_hexdump(dset: Type[DatasetProvider], sample: Sample):
    dim = (100, 100)
    image_path = get_path_image(dset, sample.md5, dim, subdir=f"_hexdump")
    if os.path.isfile(image_path):
        return

    with open(sample.filepath, "rb") as f:
        content = f.read()
        content = content[:dim[0] * dim[1]]

    image = CallGraphImage.get_image_from_hexdump(dim, content)
    pil_image = Image.fromarray(image)
    pil_image.save(image_path)


def tmp_count_rcalls(cg: CallGraph):
    calls = {}
    calls_path = os.path.join(Bodmas.get_dir_info(), f"{cg.md5}.calls.json")
    for node in cg.nodes.values():
        for i in node.instructions:
            if "call" in i.mnemonic or "jmp" in i.mnemonic:
                dict_key_add(calls, i.mnemonic)
                if not i.refs or len(i.refs) == 0:
                    Logger.info(f"No refs: <{i.get_fmt()}> <{i.disasm} | {i}> {cg.md5} {cg.file_path} {node}")
                    dict_key_inc(calls, f"{i.mnemonic}_no_ref")

    with open(calls_path, "w") as f:
        json.dump(calls, f)


def tmp_call_jmp_stats(instructions_path: str):
    with open(instructions_path, "r") as f:
        instructions = json.load(f)

    calls = {}
    jmps = {}

    for key, value in instructions.items():
        if "call" in key:
            calls[key] = value
        elif "jmp" in key:
            jmps[key] = value
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 10))
    y_pos = np.arange(len(calls))
    plt.barh(y_pos, calls.values(), label="Calls")
    y_pos2 = len(calls) + np.arange(len(jmps))
    plt.barh(y_pos2, jmps.values(), label="Jumps")
    plt.yticks(ticks=list(y_pos) + list(y_pos2), labels=list(calls.keys()) + list(jmps.keys()))
    plt.legend()
    plt.savefig("./calls_jumps.png")


IMAGE_INFO = {"configs": {}}


def tmp_collect_image_stats(sample: Sample):
    """
    Run this in main:
    ```
    process_samples(Bodmas, tmp_collect_image_stats, batch_size=1000, max_batches=None)

    with open("BODMAS_image_info.json", "w") as f:
        json.dump(IMAGE_INFO, f)
    ```
    """
    cg = CallGraph(sample.filepath, scan=False, verbose=False)
    info_path = os.path.join(Bodmas.get_dir_info(), f"{cg.md5}_imageinfo.json")
    if not os.path.isfile(info_path):
        return

    with open(info_path, "r") as f:
        info = json.load(f)

    for key, value in info["configs"].items():
        dict_key_add(IMAGE_INFO["configs"], key, value, collect_as_list=True)


def tmp_display_image_stats():
    with open("BODMAS_image_info.json", "r") as f:
        info = json.load(f)

    print("Image stats on DFS-INSTRUCTION-LENGTH(the original, before cropping to 512x512) for DFS kwargs")
    for key, value in info["configs"].items():
        stats = list_stats(value)
        print(key, stats)


if __name__ == "__main__":
    config.load_env()
    # process_samples(Bodmas, create_image, batch_size=1000, max_batches=None, pool=ProcessPoolExecutor(max_workers=8))
    # process_samples(Bodmas, create_image_on_dfs_files, batch_size=1000, max_batches=None,
    #                 pool=ProcessPoolExecutor(max_workers=8))
    # process_samples(Bodmas, create_image_on_simple_hexdump, batch_size=1000, max_batches=None,
    #                 pool=ProcessPoolExecutor(max_workers=8))
