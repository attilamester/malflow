import hashlib
import json
import os.path
import time
from typing import List, Dict, Set, Union, Tuple, Callable, Optional

import networkx as nx
import pygraphviz
import r2pipe
from networkx import MultiDiGraph

from core.model.fingerprint import CGFingerprint
from core.model.function import FunctionType, CGNode
from core.model.instruction import Instruction, InstructionParameter
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

    def get_node_by_label(self, label) -> Optional[CGNode]:
        return self.nodes.get(label, None)

    def get_node_by_rva(self, rva: str) -> Optional[CGNode]:
        return self.addresses.get(rva, None)

    def add_node(self, node: CGNode):
        if node.label in self.nodes:
            if node.rva.value != self.nodes[node.label].rva.value:
                raise Exception(f"Conflict while adding node {node} ; existing {self.nodes[node.label]}")
            else:
                return
        self.nodes[node.label] = node
        self.addresses[node.rva.value] = node

    @staticmethod
    def get_compressed_file_name(md5: str):
        return f"{md5}.cgcompressed.pickle"

    def format_nx(self, fingerprint_matches: Dict[str, Dict[str, Set[Tuple[str, int]]]]) -> nx.DiGraph:
        def add_info_of_fingerprint_matches(_dict, hash):
            _dict["matches"] = []
            _dict["match_info"] = []
            label_matches = {}
            for label, matches in fingerprint_matches[hash].items():
                _dict["matches"].extend([f"{m[0]},{label},tf{m[1]}" for m in matches])
                label_matches[label] = len(matches)

            for label, value in dict(sorted(label_matches.items())).items():
                _dict["match_info"].append(f"{label}:{value}")

        nx_g = nx.DiGraph()
        for str, node in self.nodes.items():
            nx_g.add_node(node.label, instructions=[i.mnemonic for i in node.instructions])

            _fp = CallGraph.get_node_fingerprint(node)
            fp = CGFingerprint(None, None, _fp)
            if fp.hash in fingerprint_matches:
                add_info_of_fingerprint_matches(nx_g.nodes[node.label], fp.hash)

        for str, n1 in self.nodes.items():
            for n2 in n1.calls:
                nx_g.add_edge(n1.label, n2.label)

                _fp = CallGraph.get_edge_fingerprint(n1, n2)
                fp = CGFingerprint(None, None, _fp)
                if fp.hash in fingerprint_matches:
                    add_info_of_fingerprint_matches(nx_g.edges[n1.label, n2.label], fp.hash)

        return nx_g

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

        agCd = r2.cmd("agCd")
        agRd = r2.cmd("agRd")

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
            self.add_node(n)
            return n

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
                if not (node2 in node1.calls):
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
                Logger.error(f"Could not process instructions on {cg_node}: {e}")
                raise e

        self.scan_time = time.time() - ts

    def get_edges(self) -> Set[Tuple[CGNode, CGNode]]:
        edges = set()
        for label, node in self.nodes.items():
            for other in node.calls:
                edges.add((node, other))
        return edges

    def get_representation(self):
        edge_set = set()
        for (a, b) in self.get_edges():
            edge_set.add((a.label, b.label))
        return edge_set

    @staticmethod
    def get_node_fingerprint(n: CGNode) -> str:
        if hasattr(n, "fingerprint"):
            return getattr(n, "fingerprint")

        if n.type == FunctionType.SUBROUTINE:
            if not n.instructions:
                b = "-"
            else:
                b = "_".join([i.mnemonic for i in n.instructions])
                # b = hashlib.md5(b.encode("ascii")).hexdigest()
        else:
            b = n.label
        if not b:
            raise Exception(n)

        n.fingerprint = b
        return b

    @staticmethod
    def get_edge_fingerprint(n1: CGNode, n2: CGNode) -> str:
        buff = ""
        buff += CallGraph.get_node_fingerprint(n1)
        buff += "|"
        buff += CallGraph.get_node_fingerprint(n2)
        return buff

    def get_fingerprints(self) -> Set[CGFingerprint]:
        fingerprints = set()
        fingerprint_hashes: Dict[str, CGFingerprint]
        fingerprint_hashes = {}
        nodes_without_edges = set(self.nodes.values())
        for n1, n2 in self.get_edges():
            buff = CallGraph.get_edge_fingerprint(n1, n2)
            fp = CGFingerprint(self.md5, CGFingerprint.Type.CALL, buff)
            if fp.hash in fingerprint_hashes:
                fingerprint_hashes[fp.hash].tf += 1
            else:
                fingerprint_hashes[fp.hash] = fp
                fingerprints.add(fp)

            if n1 in nodes_without_edges:
                nodes_without_edges.remove(n1)
            if n2 in nodes_without_edges:
                nodes_without_edges.remove(n2)

        for n in nodes_without_edges:
            fp = CGFingerprint(self.md5, CGFingerprint.Type.FUNCTION, CallGraph.get_node_fingerprint(n))
            if fp.hash in fingerprint_hashes:
                fingerprint_hashes[fp.hash].tf += 1
            else:
                fingerprint_hashes[fp.hash] = fp
                fingerprints.add(fp)

        return fingerprints

    def DFS(self, node_sorter: Callable[[CGNode], str] = lambda n: n.label):
        node_list = []
        visited_nodes = {}

        def dfs(node: CGNode):
            if node in visited_nodes:
                return

            visited_nodes[node] = True
            node_list.append(node)

            for node in sorted(node.calls, key=lambda n: node_sorter(n)):
                dfs(node)

        for node in sorted(self.entrypoints, key=lambda n: node_sorter(n)):
            dfs(node)

        for node in sorted(self.nodes.values(), key=lambda n: node_sorter(n)):
            dfs(node)

        return node_list

    def get_signature(self):
        buff = self.DFS(node_sorter=CallGraph.get_node_fingerprint)
        return [CallGraph.get_node_fingerprint(n) for n in buff]

    def get_image(self, verbose=False):
        global_opcodes = b""
        dfs_nodes = self.DFS(node_sorter=CallGraph.get_node_fingerprint)
        for node in dfs_nodes:
            i: Instruction
            global_opcodes += b"".join([i.opcode for i in node.instructions])
        if verbose:
            Logger.info(f"Image length: {len(global_opcodes)}  for {self.md5}")

        return global_opcodes
