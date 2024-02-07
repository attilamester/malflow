import os
import unittest
from typing import List, Set, Tuple, Union, Optional

from cases.data.r2_scanner_data import R2_SCANNER_DATA, R2ScannerData
from core.data.malware_bazaar import MalwareBazaar
from core.model import CallGraph, CallGraphCompressed
from core.model.function import CGNode
from util import config


# ================
# Helper functions for call graph test generation
# ================

def generate_test_data(cg: CallGraph, test_sample: R2ScannerData):
    def buff_nodes(cg: CallGraph):
        nodes = get_sample_get_nodes(cg)
        return "{\"" + "\", \"".join(sorted(nodes)) + "\"}"

    def buff_links(cg: CallGraph):
        links = get_sample_get_links(cg)
        return "{" + ", ".join([f"(\"{t[0]}\", \"{t[1]}\")" for t in sorted(links)]) + "}"

    def buff_instruction_parameters(i: List[str]):
        if i:
            return "[\"" + "\", \"".join(i) + "\"]"
        else:
            return "[]"

    def buff_instruction_prefix(p: Optional[str]):
        if p:
            return f", \"{p}\""
        else:
            return ""

    def buff_function_instructions(cg_node: CGNode):
        instructions = get_sample_get_function_instructions(cg_node)
        return "[" + ", ".join(
            [
                f"(\"{i[0]}\", {buff_instruction_parameters(i[1])}"
                f"{buff_instruction_prefix(i[2]) if len(i) == 3 else ''})"
                for i in instructions
            ]) + "]"

    print("\n")
    print(f"<<< Ground truth for md5:{cg.md5} >>>")
    print(f"<<< nodes: \n{buff_nodes(cg)}\n>>>")
    print(f"<<< links: \n{buff_links(cg)}\n>>>")
    print(f"<<< functions: ")
    cg_node: CGNode
    cg_nodes: List[CGNode]
    if test_sample.functions:
        cg_nodes = [cg.get_node_by_label(lbl) for lbl in test_sample.functions.keys()]
    else:
        cg_nodes = cg.DFS()[:5]
    for node in cg_nodes:
        print(f"<<< {node.label}")
        print(f"<<< {buff_function_instructions(node)}")
    return


def get_sample_get_nodes(cg: CallGraph) -> Set[str]:
    return set([node.label for node in cg.nodes.values()])


def get_sample_get_links(cg: CallGraph) -> Set[Tuple[str, str]]:
    return {(caller.label, callee.label) for caller in cg.nodes.values() for callee in caller.get_calls()}


def get_sample_get_function_instructions(cg_node: CGNode) -> List[Tuple[Union[str, List[str]], ...]]:
    return [(i.mnemonic, [p.value for p in i.parameters]) + ((i.prefix.value,) if i.prefix else ()) for i in
            cg_node.instructions]


# ================
# Test functions for call graph
# ================

def test_sample(unittest: unittest.TestCase, test_sample: R2ScannerData, gen_test_data: bool = False):
    sample = MalwareBazaar.get_sample(sha256=test_sample.sha256)
    cg = CallGraph(file_path=sample.filepath, scan=True, verbose=False)

    if gen_test_data:
        """
        Used only when creating new test sample data.
        """
        generate_test_data(cg, test_sample)
        return

    unittest.assertEqual(test_sample.nodes, get_sample_get_nodes(cg))
    unittest.assertEqual(test_sample.links, get_sample_get_links(cg))

    if test_sample.functions:
        for function_name in test_sample.functions.keys():
            cg_node = cg.get_node_by_label(function_name)
            unittest.assertIsNotNone(cg_node)
            unittest.assertEqual(test_sample.functions[function_name],
                                 get_sample_get_function_instructions(cg_node))

    if test_sample.dfs:
        unittest.assertEqual(test_sample.dfs, [node.label for node in cg.DFS()])
    test_callgraph_compression(unittest, cg)


def test_callgraph_compression(unittest: unittest.TestCase, cg: CallGraph):
    dir = "./"
    compressed_file_path = CallGraphCompressed.get_compressed_path(dir, cg.md5)
    cg_compressed = CallGraphCompressed(cg)

    # Check: before dump: file does not exist
    if os.path.isfile(compressed_file_path):
        os.remove(compressed_file_path)
    unittest.assertFalse(os.path.isfile(compressed_file_path))

    # Check: after dump, file exists
    cg_compressed.dump_compressed(dir)
    unittest.assertTrue(os.path.isfile(compressed_file_path))

    # Check: after load
    cg_compressed_from_disk = CallGraphCompressed.load(compressed_file_path)
    unittest.assertEqual(cg_compressed, cg_compressed_from_disk)
    unittest.assertEqual(cg, cg_compressed_from_disk.decompress())

    os.remove(compressed_file_path)


class TestR2Scanner(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        config.load_env()

    def test_md5_bart_35987(self):
        test_sample(self, R2_SCANNER_DATA["35987"])

    def test_md5_chinachopper_43c16(self):
        test_sample(self, R2_SCANNER_DATA["43c16"])

    def test_md5_netwalker_9c7be(self):
        test_sample(self, R2_SCANNER_DATA["9c7be"])

    def test_md5_powerstats_b2457(self):
        test_sample(self, R2_SCANNER_DATA["b2457"])

    def test_md5_powerstats_0aab6(self):
        test_sample(self, R2_SCANNER_DATA["0aab6"])

    def test_md5_powerstats_973bf(self):
        test_sample(self, R2_SCANNER_DATA["973bf"])
