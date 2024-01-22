import os
import pickle
from typing import Dict, List, Union

from core.model.call_graph import CallGraph
from core.model.function import CGNode, FunctionType
from core.model.instruction import Instruction


class CallGraphCompressed:
    md5: str
    file_path: str
    nodes: Dict[str, List]
    scan_time: Union[float, None]

    def __init__(self, cg: "CallGraph"):
        self.init(cg)

    def init(self, cg: "CallGraph"):
        self.md5 = cg.md5
        self.file_path = cg.file_path
        self.scan_time = cg.scan_time
        self.nodes = {}
        for str, node in cg.nodes.items():
            self.nodes[str] = [node.label, node.rva.value, node.type.value,
                               [[instr.disasm, instr.opcode] for instr in node.instructions],
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
            node.instructions = [Instruction(i[0], i[1]) for i in instructions]
            cg.add_node(node)
            if label.startswith("entry"):
                cg.entrypoints.append(node)
        for label, data in self.nodes.items():
            node_label, node_rva, node_type, mnemonics, call_labels = data
            node = cg.get_node_by_label(node_label)
            for call_label in call_labels:
                other_node = cg.get_node_by_label(call_label)
                node.add_call_to(other_node)
        return cg

    def dump_compressed(self, dir_path):
        file_path = CallGraphCompressed.get_compressed_path(dir_path, self.md5)
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
        return file_path

    @staticmethod
    def load(path: str) -> "CallGraphCompressed":
        with open(path, "rb") as f:
            cg_compressed: CallGraphCompressed
            cg_compressed = pickle.load(f)
            return cg_compressed

    @staticmethod
    def get_compressed_path(dir_path, md5):
        return os.path.join(dir_path, CallGraph.get_compressed_file_name(md5))

    def __eq__(self, other):
        if isinstance(other, CallGraphCompressed):
            return (self.md5 == other.md5 and
                    self.scan_time == other.scan_time and
                    self.file_path == other.file_path and
                    self.nodes == other.nodes)
        return False
