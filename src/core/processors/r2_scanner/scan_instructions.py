import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from core.data.bodmas import Bodmas
from core.model import CallGraph, CallGraphCompressed
from core.model.sample import Sample
from core.processors.r2_scanner.scan_samples import extract_sample_callgraph_instructions
from core.processors.util import process_samples
from util import config
from util.logger import Logger
from util.misc import list_stats, dict_key_inc


def extract_sample_instructions(sample: Sample):
    cg = CallGraph(sample.filepath, scan=False, verbose=False)
    md5 = cg.md5

    instructions_path = os.path.join(Bodmas.get_dir_r2_scans(), f"{md5}.instructions.json")
    if os.path.isfile(instructions_path):
        return

    compressed_path = CallGraphCompressed.get_compressed_path(Bodmas.get_dir_r2_scans(), md5)
    if not os.path.isfile(compressed_path):
        raise Exception(f"No r2 found on disk: {md5} for {sample.filepath}")

    try:
        cg = CallGraphCompressed.load(compressed_path, verbose=True).decompress()
        extract_sample_callgraph_instructions(cg)
    except Exception as e:
        Logger.error(f"Could not load compressed callgraph: {e} [{md5} {sample.filepath}]")


INSTRUCTIONS = Counter()
INSTRUCTIONS_PATH = "BODMAS_instructions_all.json"


def sum_up_instructions(sample: Sample):
    global INSTRUCTIONS
    cg = CallGraph(sample.filepath, scan=False, verbose=False)
    md5 = cg.md5

    instructions_path = os.path.join(Bodmas.get_dir_r2_scans(), f"{md5}.instructions.json")
    if not os.path.isfile(instructions_path):
        Logger.error(f"No instructions found for {sample.filepath}")
        return

    with open(instructions_path, "r") as f:
        instructions = json.load(f)

    INSTRUCTIONS += instructions


# ==============================
# Stats on the INSTRUCTIONS_PATH
# ==============================

def get_param_counts(instructions):
    param_counts = {}
    for instr, count in instructions.items():
        mnemonic_end = instr.rindex("]")
        param_count = 0
        params = instr[mnemonic_end + 1:].strip()
        if params:
            param_count = len(params.split(" "))
        dict_key_inc(param_counts, param_count, count)
    return param_counts


def display_instruction_stats():
    if not os.path.isfile(INSTRUCTIONS_PATH):
        Logger.error(f"No instructions found at {INSTRUCTIONS_PATH}")
        return
    with open(INSTRUCTIONS_PATH, "r") as f:
        instructions = json.load(f)

    param_counts = get_param_counts(instructions)

    sorted_values = [(k, v) for k, v in sorted(instructions.items(), key=lambda item: item[1])]
    buff = f"Unique instructions: {len(instructions)}\n"
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.title(buff)
    plt.yscale("log")
    plt.ylabel("Count of such instructions")
    plt.xlabel("With frequency")
    plt.hist(instructions.values(), bins=50, label="Instruction occurrence histogram")
    plt.legend()
    plt.savefig("BODMAS_instructions_all.png")

    plt.subplot(2, 1, 2)
    plt.title(f"Param counts: {param_counts}")
    plt.ylabel("Count of such instructions")
    plt.xlabel("Param count")
    plt.bar(param_counts.keys(), param_counts.values(), label="Param counts")
    plt.legend()
    plt.savefig("BODMAS_instructions_all.png")

    with open("BODMAS_instructions_all.txt", "w") as f:
        buff += f"Instruction occurrence stats: {list_stats(list(instructions.values()), True)} \n"
        top = '\t' + '\n\t'.join([f"{v}" for v in sorted_values[-100:][::-1]])
        low = '\t' + '\n\t'.join([f"{v}" for v in sorted_values[:100]])
        buff += f"Top: \n{top} \n"
        buff += f"Low: \n{low}"
        f.write(buff)


if __name__ == "__main__":
    config.load_env()
    process_samples(Bodmas, extract_sample_instructions, batch_size=1000, max_batches=None,
                    pool=ProcessPoolExecutor(max_workers=12))

    process_samples(Bodmas, sum_up_instructions, batch_size=1000)
    with open(INSTRUCTIONS_PATH, "w") as f:
        json.dump(INSTRUCTIONS, f, sort_keys=True)

    display_instruction_stats()
