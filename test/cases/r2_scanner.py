import unittest
from typing import List

from cases.r2_scanner_data import R2_SCANNER_DATA, R2ScannerData
from core.data.malware_bazaar import MalwareBazaar
from core.model import CallGraph
from core.model.function import CGNode
from core.model.instruction import Instruction
from util import config


class TestR2Scanner(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        config.load_env()

    def __test_sample(self, test_sample: R2ScannerData, generate_ground_truth: bool = False):
        sample = MalwareBazaar.get_sample(sha256=test_sample.sha256)
        cg = CallGraph(file_path=sample.filepath, scan=True, verbose=False)
        nodes: List[CGNode]
        nodes = list(cg.nodes.values())

        actual_nodes = set([node.label for node in nodes])
        actual_links = {(caller.label, callee.label) for caller in nodes for callee in caller.calls}

        if generate_ground_truth:
            """
            Used only when creating new test sample data.
            """

            buff_nodes = "\", \"".join(sorted(actual_nodes))
            buff_links = ", ".join([f"(\"{t[0]}\", \"{t[1]}\")" for t in sorted(actual_links)])

            def buff_instruction_parameters(i: Instruction):
                if i.parameters:
                    return "\"" + "\", \"".join([p.value for p in i.parameters]) + "\""
                else:
                    return ""

            print("\n")
            print(f"<<< Ground truth for md5:{cg.md5} >>>")
            print(f"<<< nodes: \n{{\"{buff_nodes}\"}}\n>>>")
            print(f"<<< links: \n{{{buff_links}}}\n>>>")
            print(f"<<< functions: ")
            node: CGNode
            for node in cg.DFS()[:5]:
                print(f"<<< {node.label}")
                instructions = ", ".join(
                    [f"(\"{i.mnemonic}\", [{buff_instruction_parameters(i)}])" for i in node.instructions])
                print(f"<<< [{instructions}]")
            return

        self.assertEqual(test_sample.nodes, actual_nodes)
        self.assertEqual(test_sample.links, actual_links)

        if test_sample.functions:
            for function_name, function_instructions in test_sample.functions.items():
                cg_node = cg.get_node_by_label(function_name)
                self.assertIsNotNone(cg_node)
                actual_instructions = [(i.mnemonic, [p.value for p in i.parameters]) for i in cg_node.instructions]
                self.assertEqual(function_instructions, actual_instructions)

    def test_md5_bart_35987(self):
        self.__test_sample(R2_SCANNER_DATA["35987"])

    def test_md5_chinachopper_43c16(self):
        self.__test_sample(R2_SCANNER_DATA["43c16"])

    def test_md5_netwalker_9c7be(self):
        self.__test_sample(R2_SCANNER_DATA["9c7be"])

    def test_md5_powerstats_b2457(self):
        self.__test_sample(R2_SCANNER_DATA["b2457"])

    def test_md5_powerstats_0aab6(self):
        self.__test_sample(R2_SCANNER_DATA["0aab6"])

    def test_md5_powerstats_973bf(self):
        self.__test_sample(R2_SCANNER_DATA["973bf"])
