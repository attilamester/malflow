import hashlib
import json
import os.path
import time
from typing import Callable, Dict, List, Optional, Tuple, Union

import networkx as nx
import pygraphviz
import r2pipe
from networkx import MultiDiGraph

from malflow.core.model.function import CGNode, FunctionType
from malflow.core.model.instruction import Instruction, InstructionParameter
from malflow.core.model.radare2_definitions import is_symbol_flag
from malflow.core.model.radare2_definitions.sanitizer import sanitize_r2_bugs
from malflow.util.logger import Logger


class CallGraph:
    md5: str
    file_path: str
    entrypoints: List[CGNode]
    nodes: Dict[str, CGNode]
    addresses: Dict[int, CGNode]
    scan_time: Union[float, None]
    imports: List

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

    def get_node_by_rva(self, rva: int) -> Optional[CGNode]:
        return self.addresses.get(rva, None)

    def add_node(self, node: CGNode) -> CGNode:
        if node.label in self.nodes:
            if node.rva.value != self.nodes[node.label].rva.value:
                # This should not happen. Let the app crash & check the logs.
                raise Exception(f"Conflict while adding node {node} ; existing {self.nodes[node.label]} "
                                f"[{self.md5} {self.file_path}]")
            else:
                return node
        if node.rva.addr in self.addresses:
            return self.addresses[node.rva.addr]
        self.nodes[node.label] = node
        self.addresses[node.rva.addr] = node
        return node

    @staticmethod
    def basepath(path):
        dirname = os.path.dirname(path)
        basename = os.path.basename(path)
        return os.path.join(dirname, basename.split(".")[0])

    def open_r2(self) -> r2pipe.open:
        """
        Returns the r2pipe instance for this PE.
        :return: r2pipe.open
        """
        r2 = r2pipe.open(self.file_path, flags=["-2"])  # close stderr
        return r2

    @staticmethod
    def close_r2(r2: r2pipe.open):
        r2.quit()

    def scan(self, verbose=False):
        Logger.info(f"Scanning {self.file_path}")

        ts = time.time()

        r2 = self.open_r2()
        r2.cmd("aaa")

        entrypoint_info = r2.cmd("ies").strip().split("\n")
        entries = -1
        for entry in entrypoint_info:
            if not entry:
                continue

            entry = entry.split()

            try:
                rva = entry[0]
                if not rva.startswith("0x"):
                    raise ValueError("Not a valid RVA")
            except (IndexError, ValueError):
                if verbose:
                    Logger.error(f"Could not parse entrypoint line: {entry} [{self.md5} {self.file_path}]")
                continue
            entries += 1
            label = entry[1]
            if not (label.startswith("entry") or label.startswith("eip") or label.startswith("main")):
                label = f"entry{entries}"

            node = CGNode(label, rva)
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
            node1 = self.get_node_by_rva(int(a, 16))
            node2 = self.get_node_by_rva(int(b, 16))
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
                rva_int = int(rva, 16)
                if rva_int not in self.addresses:
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
                pdfj = json.loads(r2.cmd(f"s {addr} ; pdfj").strip())
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
                if verbose:
                    Logger.error(f"Could not process instructions on {cg_node}: {e} [{self.md5} {self.file_path}]")
                raise e

        r2.quit()
        self.scan_time = time.time() - ts

    def get_edges(self) -> List[Tuple[CGNode, CGNode]]:
        edges = []
        for label, node in self.nodes.items():
            for other in node.get_calls():
                edges.append((node, other))
        return edges

    def get_representation(self):
        edge_set = set()
        for (a, b) in self.get_edges():
            edge_set.add((a.label, b.label))
        return edge_set

    def dfs(self, node_sorter: Callable[[CGNode], str] = None):
        node_list = []
        visited_nodes = {}

        if node_sorter is None:
            node_sorter = dfs_sorter

        def _dfs(node: CGNode):
            if node.label in visited_nodes:
                return

            visited_nodes[node.label] = True
            node_list.append(node)

            for n in sorted(node.get_calls(), key=node_sorter):
                _dfs(n)

        for node in sorted(self.entrypoints, key=lambda n: node_sorter(n)):
            _dfs(node)

        for node in sorted(self.nodes.values(), key=lambda n: node_sorter(n)):
            _dfs(node)

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
                if ref.addr in self.addresses:
                    node_calls_labels_from_instructions.append((self.addresses[ref.addr].label, k))

        return node_calls_labels_from_instructions

    def dfs_instructions(self, max_instructions: int = None, allow_multiple_visits: bool = False,
                         store_call: bool = False, store_cg_node: bool = False) -> List[Union[Instruction, CGNode]]:
        """
        Based on :func:`<core.model.call_graph.CallGraph.DFS>`
        The traversal here is done on the instructions level, not the nodes.
        The order of the instructions is preserved according to execution flow.
        :param max_instructions: Maximum number of instructions to traverse
        :param allow_multiple_visits: Allow multiple visits to the same function node
        :param store_call: Whether to store the actual `call` instruction.
        :param store_cg_node: Whether to store the CGNode when visiting a function node in the instruction list.
        :return: List[Instruction | Tuple[CGNode, int: depth]]
        """
        instructions = []
        dfs_nodes = self.dfs()
        visited_nodes = set()
        node_calls_cache = {}

        def get_node_calls_from_instructions(node):
            if node.label not in node_calls_cache:
                node_calls_cache[node.label] = self.get_node_calls_from_instructions(node)
            return node_calls_cache[node.label]

        def build_instruction_traversal(node: CGNode, depth=0):
            if max_instructions and len(instructions) > max_instructions:
                return

            if node.label in visited_nodes:
                if not allow_multiple_visits:
                    return
                else:
                    if store_call:
                        instructions.extend(node.instructions)
                    else:
                        node_calls = get_node_calls_from_instructions(node)
                        call_indices = {t[1] for t in node_calls}
                        instructions.extend(
                            [instr for i, instr in enumerate(node.instructions) if i not in call_indices])
                    return

            visited_nodes.add(node.label)

            if store_cg_node:
                instructions.append((node, depth))

            last_index = 0
            node_calls = get_node_calls_from_instructions(node)

            if not node_calls:
                instructions.extend(node.instructions)
            else:
                for callee_label, i in node_calls:
                    callee = self.get_node_by_label(callee_label)
                    instructions.extend(node.instructions[last_index: i + store_call])
                    last_index = i + 1
                    build_instruction_traversal(callee, depth + 1)
                instructions.extend(node.instructions[last_index:])

        for n in dfs_nodes:
            if n.label in visited_nodes:
                continue
            if max_instructions and len(instructions) > max_instructions:
                break
            build_instruction_traversal(n, 0)

        return instructions

    def dfs_instructions_summary_txt(self, completeness: int = 0) -> str:
        """
        :param completeness: How much information is included in the summary.
            0 - full DFS list
            TODO: add more levels
        :return:
        """
        buffer = ""
        for instr in self.dfs_instructions(max_instructions=None, allow_multiple_visits=False, store_call=True,
                                           store_cg_node=True):
            if isinstance(instr, Instruction):
                if instr.rva is not None:
                    # malflow versions using r2 5.8.8 do not have instruction address saved in the call graph model
                    buffer += f"{instr.rva.value} {instr.disasm}\n"
                else:
                    buffer += f"NaN {instr.disasm}\n"
            else:
                cg_node, depth = instr
                buffer += f"Dep{depth} {cg_node.rva.value} {cg_node.label}\n"
        return buffer

    def dump_json(self, dir_path: str = None) -> str:
        if dir_path is None:
            dir_path = os.path.dirname(self.file_path)
        file_path = os.path.join(dir_path, f"{self.md5}.cg.json")
        Logger.info(f"Dumping call graph [md5: {self.md5} | {self.file_path}] in [JSON] to {file_path}")
        with open(file_path, "w") as f:
            json.dump({
                "md5": self.md5,
                "file_path": self.file_path,
                "scan_time": self.scan_time,
                "entrypoints": [ep.label for ep in self.entrypoints],
                "nodes": {
                    label: {
                        "rva": node.rva.value,
                        "type": node.type.value,
                        "instructions": [instr.get_fmt() for instr in node.instructions],
                        "calls": [n.label for n in node.get_calls()]
                    }
                    for label, node in self.nodes.items()
                }
            }, f)
        return file_path

    def create_pygraphviz(self, layout="dot") -> pygraphviz.AGraph:
        ag = pygraphviz.AGraph(directed=True)

        def get_label(node: CGNode) -> str:
            imports = "\n".join(
                [callee_label for callee_label, callee in node.calls.items() if callee.type == FunctionType.DLL]
            )
            # instructions = ''.join(
            #     ['  • ' + node.instructions[i].disasm + ('\n' if (i + 1) % 3 == 0 else '   ')
            #      for i in range(min(len(node.instructions), 10))]
            # ) + ('\n  ...' if len(node.instructions) > 10 else '')
            return f"""
{label}\n
--------------------
[ {node.rva.value} ]
--------------------
{imports}
{len(node.instructions)} instructions\n
"""

        def get_tooltip(node: CGNode) -> str:
            if node.instructions:
                return "\n".join([instr.disasm for instr in node.instructions])
            return "- No instructions -"

        for label, node in self.nodes.items():
            ag.add_node(label, shape="box", style="filled",
                        fillcolor="lightblue" if node.type == FunctionType.DLL else "black",
                        fontcolor="white" if node.type != FunctionType.DLL else "black",
                        label=get_label(node),
                        tooltip=get_tooltip(node))

        for (a, b) in self.get_edges():
            ag.add_edge(a.label, b.label)

        ag.layout(prog=layout)

        return ag

    def create_pygraphviz_fdp(self) -> pygraphviz.AGraph:
        """
        Force directed placement
        """
        ag = pygraphviz.AGraph(directed=True)

        for label, node in self.nodes.items():
            ag.add_node(label, shape="point", style="filled",
                        fillcolor="lightblue" if node.type == FunctionType.DLL else "black",
                        fontcolor="white" if node.type != FunctionType.DLL else "black",
                        label="",
                        tooltip=f"{node.label}\n{node.rva.value}\n{len(node.instructions)} instructions")

        for (a, b) in self.get_edges():
            ag.add_edge(a.label, b.label)

        ag.layout(prog="fdp")

        return ag

    def dump_dot(self, dir_path: str = None) -> str:
        if dir_path is None:
            dir_path = os.path.dirname(self.file_path)
        file_path = os.path.join(dir_path, f"{self.md5}.cg.dot")
        Logger.info(f"Dumping call graph [md5: {self.md5} | {self.file_path}] in [DOT] to {file_path}")

        ag = self.create_pygraphviz()
        ag.draw(file_path, format="dot")

    def dump_svg(self, dir_path: str = None) -> str:
        if dir_path is None:
            dir_path = os.path.dirname(self.file_path)

        Logger.info(f"Dumping call graph [md5: {self.md5} | {self.file_path}] in [SVG] to {dir_path}")

        ag = self.create_pygraphviz("dot")
        ag.draw(os.path.join(dir_path, f"{self.md5}.cg.svg"), format="svg")
        ag = self.create_pygraphviz_fdp()
        ag.draw(os.path.join(dir_path, f"{self.md5}.cg.fdp.svg"), format="svg")

    def __eq__(self, other):
        if isinstance(other, CallGraph):
            return (self.md5 == other.md5 and
                    self.file_path == other.file_path and
                    self.scan_time == other.scan_time and
                    self.entrypoints == other.entrypoints and
                    self.nodes == other.nodes and
                    self.addresses == other.addresses)
        return False


def dfs_sorter(node: CGNode):
    if is_symbol_flag(node.label):
        return "_" + node.label
    return node.label
