import hashlib
import json
import os.path
import time
from typing import List, Dict, Set, Union, Tuple, Callable, Optional

import networkx as nx
import pygraphviz
import r2pipe
from networkx import MultiDiGraph

from core.model.function import CGNode, FunctionType
from core.model.instruction import Instruction, InstructionParameter
from core.model.radare2_definitions.sanitizer import sanitize_r2_bugs
from util.logger import Logger


class CallGraph:
    md5: str
    file_path: str
    entrypoints: List[CGNode]
    nodes: Dict[str, CGNode]
    addresses: Dict[str, CGNode]
    scan_time: Union[float, None]

    __nx: MultiDiGraph

    def __init__(self, file_path: str = None, scan=True, verbose=False, decompressed=False):

        if not decompressed:
            with open(file_path, "rb") as f:
                self.md5 = hashlib.md5(f.read()).hexdigest()
        self.file_path = file_path
        self.entrypoints = []
        self.nodes = {}
        self.addresses = {}
        self.scan_time = None

        if scan:
            try:
                self.scan(verbose)
            except Exception as e:
                Logger.error(f"Could not scan {self.md5}: {e}")
                raise Exception(e)

    def get_node_by_label(self, label) -> Optional[CGNode]:
        return self.nodes.get(label, None)

    def get_node_by_rva(self, rva: str) -> Optional[CGNode]:
        return self.addresses.get(rva, None)

    def add_node(self, node: CGNode) -> CGNode:
        if node.label == "eip":
            # experiments prove that `agCd` may have duplicate addresses with both labels `entry0` and `eip`
            for ep in self.entrypoints:
                if node.rva.value == ep.rva.value:
                    Logger.warning(f"Skipping adding EIP node {node} [{self.md5} {self.file_path}]")
                    return ep

        if node.label in self.nodes:
            if node.rva.value != self.nodes[node.label].rva.value:
                if node.label.startswith("entry"):
                    Logger.warning(
                        f"Incorrect entrypoint address provided by ie ({self.nodes[node.label]}. Fixing to the corrected value {node} "
                        f"[{self.md5} {self.file_path}]")
                    self.addresses.pop(self.nodes[node.label].rva.value)
                else:
                    raise Exception(f"Conflict while adding node {node} ; existing {self.nodes[node.label]} "
                                    f"[{self.md5} {self.file_path}]")
            else:
                return node
        self.nodes[node.label] = node
        self.addresses[node.rva.value] = node
        return node

    @staticmethod
    def get_compressed_file_name(md5: str):
        return f"{md5}.cgcompressed.pickle"

    @staticmethod
    def basepath(path):
        dirname = os.path.dirname(path)
        basename = os.path.basename(path)
        return os.path.join(dirname, basename.split(".")[0])

    def scan(self, verbose=False):
        Logger.info(f"Scanning {self.file_path}")

        ts = time.time()

        r2 = r2pipe.open(self.file_path, flags=["-2"])
        r2.cmd("aaa")

        entrypoint_info = r2.cmd("ie").split("\n")
        entries = -1
        for entry in entrypoint_info[1:]:
            if not entry:
                break
            entries += 1

            entry = entry.split(" ")
            rva = entry[0].split("=")[1]
            node = CGNode(f"entry{entries}", rva)
            self.add_node(node)
            self.entrypoints.append(node)
            if verbose:
                Logger.info(f"[Entry] {node}")

        agCd = sanitize_r2_bugs(r2.cmd("agCd"))
        agRd = sanitize_r2_bugs(r2.cmd("agRd"))

        nx_g = nx.drawing.nx_agraph.from_agraph(pygraphviz.AGraph(agCd))
        nx_g_references = nx.drawing.nx_agraph.from_agraph(pygraphviz.AGraph(agRd))

        # ==============
        # Process agCd - Global callgraph https://r2wiki.readthedocs.io/en/latest/options/a/ag/
        # ==============

        for addr, data in nx_g.nodes.items():
            node = CGNode(data["label"], addr)
            self.add_node(node)
            if verbose:
                Logger.info(f"[Node - agC] {node}")

        for (a, b, w), data in nx_g.edges.items():
            node1 = self.get_node_by_rva(a)
            node2 = self.get_node_by_rva(b)
            node1.add_call_to(node2)
            if verbose:
                Logger.info(f"[Call - agC] {node1} -> {node2}")

        # ==============
        # Process agRd - Global references graph
        # ==============

        def get_rva_of_label(lbl: str):
            """
            Padding to 8 chars is important to match the existing addressing e.g. 0x0043012c
            """
            return r2.cmd(f"s {lbl} ; s:8").strip()

        def add_node_by_label(lbl: str):
            rva = get_rva_of_label(lbl)
            n = CGNode(lbl, rva)
            return self.add_node(n)

        for addr, data in nx_g_references.nodes.items():
            if InstructionParameter.is_function(addr):
                rva = get_rva_of_label(addr)
                if rva not in self.addresses:
                    node = CGNode(data["label"], rva)
                    self.add_node(node)
                    if verbose:
                        Logger.info(f"[Node - agR] {node}")

        for (a, b, w), data in nx_g_references.edges.items():
            node1 = self.get_node_by_label(a)
            node2 = self.get_node_by_label(b)

            if InstructionParameter.is_function(a):
                if node1 is None:
                    node1 = add_node_by_label(a)
            if InstructionParameter.is_function(b):
                if node2 is None:
                    node2 = add_node_by_label(b)
                if node1 is None:
                    Logger.error(f"None: {a} {b}")
                else:
                    if not node1.has_call_to(node2):
                        if verbose:
                            Logger.info(f"[Call - agR] {node1} -> {node2}")
                        node1.add_call_to(node2)

        for label, cg_node in self.nodes.items():
            cg_node: CGNode
            addr = cg_node.rva.value
            if cg_node.type == FunctionType.DLL:
                continue
            try:
                pdfj = json.loads(r2.cmd(f"s {addr} ; pdfj"))
                cg_node.set_instructions_from_function_disassembly(pdfj)
            except json.decoder.JSONDecodeError as e:
                # TODO: conceptual question whether we want to keep these nodes or not.
                #  The fact that pdf fails means r2 could not find commands on that address, that being dynamic one
                #  e.g. call [eax]
                #   - in this case, we could even delete the node
                #   - but still, the call relation should mean a link in the graph
                if verbose:
                    Logger.warning(f"Could not run pdfj on {cg_node}: {e}")
            except Exception as e:
                Logger.error(f"Could not process instructions on {cg_node}: {e} [{self.md5} {self.file_path}]")
                raise e

        r2.quit()
        self.scan_time = time.time() - ts

    def get_edges(self) -> Set[Tuple[CGNode, CGNode]]:
        edges = set()
        for label, node in self.nodes.items():
            for other in node.get_calls():
                edges.add((node, other))
        return edges

    def get_representation(self):
        edge_set = set()
        for (a, b) in self.get_edges():
            edge_set.add((a.label, b.label))
        return edge_set

    def DFS(self, node_sorter: Callable[[CGNode], str] = lambda n: n.label):
        node_list = []
        visited_nodes = {}

        def dfs(node: CGNode):
            if node.label in visited_nodes:
                return

            visited_nodes[node.label] = True
            node_list.append(node)

            for node in sorted(node.get_calls(), key=lambda n: node_sorter(n)):
                dfs(node)

        for node in sorted(self.entrypoints, key=lambda n: node_sorter(n)):
            dfs(node)

        for node in sorted(self.nodes.values(), key=lambda n: node_sorter(n)):
            dfs(node)

        return node_list

    def get_node_calls_from_instructions(self, node: CGNode) -> List[Tuple[str, int]]:
        """
        TODO: this will need an emulator for rcall and ucall
        Correlate the node's calls given the `agCd` and `agRd` with the node's instructions given by `pdfj`
        :return: labels of the nodes called by the instructions of the given node, in the order of its instructions
        """

        if node.type == FunctionType.DLL:
            return []

        node_calls_labels_from_instructions = []

        for k, i in enumerate(node.instructions):
            if not i.refs:
                continue

            for ref in i.refs:
                # 0x0043012c
                rva_hex = f"{ref.addr:#0{10}x}"
                if rva_hex in self.addresses:
                    node_calls_labels_from_instructions.append((self.addresses[rva_hex].label, k))

        return node_calls_labels_from_instructions

    def DFS_instructions(self) -> List[Instruction]:
        """
        Based on :func:`<core.model.call_graph.CallGraph.DFS>`
        The traversal here is done on the instructions level, not the nodes.
        The order of the instructions is preserved according to execution flow.
        :return: List[IInstruction]
        """
        instructions = []
        dfs_nodes = self.DFS()
        visited_nodes = set()
        visiting_nodes = set()

        def build_instruction_traversal(node: CGNode):
            if node.label in visited_nodes or node.label in visiting_nodes:
                return

            visiting_nodes.add(node.label)

            Logger.info(f"Traversing {node.label}")

            last_index = 0
            node_calls = self.get_node_calls_from_instructions(node)

            for callee_label, i in node_calls:
                instructions.extend(node.instructions[last_index: i])
                Logger.info(f"Traversing {node.label} -> adding {node.instructions[last_index: i]}")
                build_instruction_traversal(self.get_node_by_label(callee_label))
                last_index = i + 1

            visited_nodes.add(node.label)

        for n in dfs_nodes:
            build_instruction_traversal(n)

        return instructions

    def get_image(self, verbose=False):
        global_opcodes = b""
        dfs_nodes = self.DFS()
        for node in dfs_nodes:
            i: Instruction
            global_opcodes += b"".join([i.opcode for i in node.instructions])
        if verbose:
            Logger.info(f"Image length: {len(global_opcodes)} for {self.md5}")

        return global_opcodes

    def __eq__(self, other):
        if isinstance(other, CallGraph):
            return (self.md5 == other.md5 and
                    self.file_path == other.file_path and
                    self.scan_time == other.scan_time and
                    self.entrypoints == other.entrypoints and
                    self.nodes == other.nodes and
                    self.addresses == other.addresses)
        return False
