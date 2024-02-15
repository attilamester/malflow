import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from core.data.bodmas import Bodmas
from core.model import CallGraph, CallGraphCompressed
from core.model.sample import Sample
from core.processors.r2_scanner.scan_samples import create_callgraph_image
from core.processors.util import process_samples
from util import config
from util.logger import Logger
from util.misc import dict_key_inc, dict_key_add


def create_image(sample: Sample):
    cg = CallGraph(sample.filepath, scan=False, verbose=False)
    md5 = cg.md5

    compressed_path = CallGraphCompressed.get_compressed_path(Bodmas.get_dir_r2_scans(), md5)
    if not os.path.isfile(compressed_path):
        raise Exception(f"No r2 found on disk: {md5} for {sample.filepath}")

    try:
        cg = CallGraphCompressed.load(compressed_path, verbose=True).decompress()
        create_callgraph_image(cg)
    except Exception as e:
        Logger.error(f"Could not load compressed callgraph: {e} [{md5} {sample.filepath}]")

    # count_rcalls(cg)


def count_rcalls(cg: CallGraph):
    calls = {}
    calls_path = os.path.join(Bodmas.get_dir_r2_scans(), f"{cg.md5}.calls.json")
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


if __name__ == "__main__":
    config.load_env()
    process_samples(Bodmas, create_image, batch_size=1000, max_batches=None, pool=ThreadPoolExecutor(max_workers=8))
