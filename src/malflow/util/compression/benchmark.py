import os
import pickle
import random
import time
from typing import List, Tuple

from malflow.util.compression.compression import BrotliCompressor, ZLibCompressor, Compressor, Bzip2Compressor, LzmaCompressor
from malflow.util.misc import list_avg, get_project_root, list_stats, display_size


def get_corpus():
    proj_root = get_project_root()
    corpus = ""
    for root, dirs, files in os.walk(proj_root):
        for file in files:
            if file.endswith(".py"):
                corpus += open(os.path.join(root, file), "r").read()
    return corpus


def get_random_string(corpus, n):
    i = random.randint(0, len(corpus) - n)
    return corpus[i: i + n]


def get_docs_to_compress(num_docs, tokens, max_token_length):
    corpus = get_corpus()
    docs = []
    for i in range(num_docs):
        doc = ""
        for i in range(random.randint(tokens / 2, tokens)):
            r = get_random_string(corpus, max_token_length)
            doc += r
        docs.append(doc)
    pickle_docs = [pickle.dumps(d) for d in docs]
    original_sizes = [len(d) for d in pickle_docs]
    return docs, pickle_docs, original_sizes


def data_provider_text(num_docs, tokens, max_token_length) -> Tuple[List, List[bytes], List[int]]:
    return get_docs_to_compress(num_docs, tokens, max_token_length)


def data_provider_callgraph() -> Tuple[List, List[bytes], List[int]]:
    from core.data.malware_bazaar import MalwareBazaar
    from core.model import CallGraph, CallGraphCompressed
    from cases.r2_scanner_data import R2_SCANNER_DATA, R2ScannerData
    from util import config
    config.load_env()

    docs = []
    test_sample: R2ScannerData
    for key, test_sample in R2_SCANNER_DATA.items():
        sample = MalwareBazaar.get_sample(sha256=test_sample.sha256)
        cg = CallGraph(file_path=sample.filepath, scan=True, verbose=False)
        docs.append(CallGraphCompressed(cg))

    pickle_docs = [pickle.dumps(d) for d in docs]
    original_sizes = [len(d) for d in pickle_docs]
    return docs, pickle_docs, original_sizes


def compression_ratio_tester(data: Tuple[List, List[bytes], List[int]], compressors: List[Compressor]):
    docs, pickle_docs, original_sizes = data

    original_sizes_stats = list_stats(original_sizes)
    original_sizes_stats.pop("length", None)
    size_info = '\n'.join([f"{k:<10}{display_size(v):>20}" for k, v in original_sizes_stats.items()])
    print(f"Testing compression ratio: {len(docs)} docs, size: \n"
          f"{size_info}")

    for compressor in compressors:
        ts = time.perf_counter()
        compressed_docs = [compressor.compress(d) for d in pickle_docs]
        dt1 = time.perf_counter() - ts
        decompressed_docs = [compressor.decompress(d) for d in compressed_docs]
        dt2 = time.perf_counter() - dt1 - ts
        compressed_sizes = [len(d) for d in compressed_docs]
        assert pickle_docs == decompressed_docs
        ratio = round(list_avg([o / c for o, c in zip(original_sizes, compressed_sizes)]), 3)
        compression_speed = original_sizes_stats["all"] / dt1
        decompression_speed = original_sizes_stats["all"] / dt2
        print(f"{str(compressor):<25} CR {ratio:<5} "
              f"| compression   {display_size(compression_speed):>10}/s "
              f"| decompression {display_size(decompression_speed):>10}/s "
              f"| compressed avg: {display_size(list_avg(compressed_sizes))}")


if __name__ == "__main__":
    compression_ratio_tester(data_provider_text(1000, 50, 1000), [
        BrotliCompressor(1),
        BrotliCompressor(2),
        BrotliCompressor(3),
        BrotliCompressor(4),
        BrotliCompressor(6),
        BrotliCompressor(8),
        BrotliCompressor(11),
        ZLibCompressor(),
        Bzip2Compressor(),
        LzmaCompressor()
    ])
    compression_ratio_tester(data_provider_callgraph(), [
        BrotliCompressor(1),
        BrotliCompressor(2),
        BrotliCompressor(3),
        BrotliCompressor(4),
        BrotliCompressor(6),
        BrotliCompressor(8),
        BrotliCompressor(11),
        BrotliCompressor(11),
        ZLibCompressor(),
        Bzip2Compressor(),
        LzmaCompressor()
    ])
