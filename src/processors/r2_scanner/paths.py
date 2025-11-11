import os
from typing import Type

from malflow.core.data import DatasetProvider
from malflow.util.misc import ensure_dir


def get_path_imageinfo(dset: Type[DatasetProvider], md5: str):
    img_info_dir = os.path.join(dset.get_dir_info(), f"imageinfo")
    ensure_dir(img_info_dir)
    return os.path.join(img_info_dir, f"{md5}_imageinfo.json")


def get_path_image(dset: Type[DatasetProvider], md5: str, dim, allow_multiple_visits: bool = True,
                   store_call: bool = True, subdir: str = None):
    img_dir = os.path.join(dset.get_dir_images(), f"images_{dim[0]}x{dim[1]}{subdir if subdir else ''}")
    ensure_dir(img_dir)
    return os.path.join(img_dir, f"{md5}_{dim[0]}x{dim[1]}_{allow_multiple_visits}_{store_call}.png")


def get_path_instructions_dfs(dset: Type[DatasetProvider], md5: str, allow_multiple_visits: bool = True,
                              store_call: bool = True):
    ins_dfs = os.path.join(dset.get_dir_instructions(), "dfs")
    ensure_dir(ins_dfs)
    return os.path.join(ins_dfs, f"{md5}.instructions_{allow_multiple_visits}_{store_call}.pickle")


def get_path_instructions_function_blocks(dset: Type[DatasetProvider], md5: str):
    ins = os.path.join(dset.get_dir_instructions(), "function_blocks")
    ensure_dir(ins)
    return os.path.join(ins, f"{md5}.instructions.pickle")


def get_path_instructions_stats(dset: Type[DatasetProvider], md5: str):
    ins_stats = os.path.join(dset.get_dir_instructions(), "stats")
    ensure_dir(ins_stats)
    return os.path.join(ins_stats, f"{md5}.instructions.json")
