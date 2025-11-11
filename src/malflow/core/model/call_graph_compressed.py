import os
import pickle
import time
from typing import Dict, List, Union

from malflow.core.model.call_graph import CallGraph
from malflow.core.model.function import CGNode, FunctionType
from malflow.core.model.instruction import Instruction
from malflow.util.compression import BrotliCompressor, Compressor
from malflow.util.logger import Logger
from malflow.util.misc import display_size


class CallGraphCompressed:
    COMPRESSED_EXTENSION = ".cgcompressed.pickle"

    md5: str
    file_path: str
    nodes: Dict[str, List]
    scan_time: Union[float, None]

    def __init__(self, cg: CallGraph):
        self.init(cg)

    def init(self, cg: CallGraph):
        self.md5 = cg.md5
        self.file_path = cg.file_path
        self.scan_time = cg.scan_time
        self.nodes = {}
        for str, node in cg.nodes.items():
            self.nodes[str] = [node.label, node.rva.value, node.type.value,
                               [instr.compress() for instr in node.instructions],
                               [n.label for n in node.get_calls()]]

    def to_dict(self):
        return {
            "md5": self.md5,
            "scan_time": self.scan_time,
            "nodes": self.nodes
        }

    def decompress(self) -> "CallGraph":
        cg = CallGraph(None, scan=False, verbose=False, decompressed=True)
        cg.md5 = self.md5
        cg.scan_time = self.scan_time
        cg.file_path = self.file_path
        for label, data in self.nodes.items():
            node_label, node_rva, node_type, instructions, call_labels = data
            node = CGNode(node_label, node_rva)
            node.type = FunctionType(node_type)
            node.instructions = [Instruction.decompress(i) for i in instructions]
            cg.add_node(node)
            if label.startswith("entry"):
                cg.entrypoints.append(node)
        for label, data in self.nodes.items():
            node_label, node_rva, node_type, instructions, call_labels = data
            node = cg.get_node_by_label(node_label)
            for call_label in call_labels:
                # TODO: this should not be here -- fix is already applied in add_node;
                # keeping this for old scans
                if call_label == "eip":
                    call_label = "entry0"
                other_node = cg.get_node_by_label(call_label)
                node.add_call_to(other_node)
        return cg

    def get_compressed(self, compressor: Compressor = BrotliCompressor(1)) -> bytes:
        return compressor.compress(pickle.dumps(self))

    def dump_compressed(self, dir_path, compressor: Compressor = BrotliCompressor(1)) -> str:
        file_path = CallGraphCompressed.get_compressed_path(dir_path, self.md5)
        with open(file_path, "wb") as f:
            f.write(self.get_compressed(compressor))
        return file_path

    @staticmethod
    def load(path: str, verbose=False) -> "CallGraphCompressed":
        ts = time.perf_counter()
        with open(path, "rb") as f:
            cg_compressed: CallGraphCompressed
            content = f.read()
            decompressor = Compressor.get_decompressor(content)
            decompressed = decompressor.decompress(content)
            cg_compressed = pickle.loads(decompressed)
            if verbose:
                dt = time.perf_counter() - ts
                Logger.info(
                    f"Loaded compressed callgraph in {dt:.1f}s"
                    f"[md5={cg_compressed.md5}, pickle={display_size(len(content))}, "
                    f"decompressed={display_size(len(decompressed))}] from {path}")
            return cg_compressed

    @staticmethod
    def get_compressed_name(md5: str):
        return f"{md5}{CallGraphCompressed.COMPRESSED_EXTENSION}"

    @staticmethod
    def get_compressed_path(dir_path, md5):
        return os.path.join(dir_path, CallGraphCompressed.get_compressed_name(md5))

    def __eq__(self, other):
        if isinstance(other, CallGraphCompressed):
            return (self.md5 == other.md5 and
                    self.scan_time == other.scan_time and
                    self.file_path == other.file_path and
                    self.nodes == other.nodes)
        return False
