import unittest
from typing import List

from core.data.malware_bazaar import MalwareBazaar
from core.model import CallGraph
from core.model.function import CGNode
from core.util import config


class TestR2Scanner(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        config.load_env()

    def test_1(self):
        # md5:3598753547549daa3db3ce408a8d634b
        sample = MalwareBazaar.get_sample(sha256="c7cb8e453252a441522c91d7fccc5065387ce2e1e0e78950f54619eda435b11d")
        cg = CallGraph(file_path=sample.filepath, scan=True, verbose=False)
        nodes: List[CGNode]
        nodes = list(cg.nodes.values())

        expected_nodes = {
            "entry0", "main",
            "fcn.004064b0", "fcn.004060f8", "fcn.0040611f", "fcn.0040615c", "fcn.00406560", "fcn.00406660",
            "fcn.004065b0", "fcn.0040618c", "fcn.00406310", "fcn.00406335", "fcn.00406264", "fcn.004063b5",
            "fcn.004062e8", "fcn.004061ba", "fcn.00406388", "fcn.0040629f", "fcn.0040623f", "fcn.00406008",
            "fcn.00402000", "fcn.00406000", "fcn.00406040", "fcn.004066a6", "fcn.00406530",
            "sub.MSVCRT.dll_memset", "sub.KERNEL32.dll_HeapCreate", "sub.MSVCRT.dll_strlen", "sub.MSVCRT.dll_memmove",
            "sym.imp.KERNEL32.dll_HeapAlloc", "sym.imp.KERNEL32.dll_HeapReAlloc", "sym.imp.KERNEL32.dll_HeapFree",
            "sym.imp.KERNEL32.dll_HeapCreate", "sym.imp.KERNEL32.dll_GetProcAddress",
            "sym.imp.KERNEL32.dll_GetModuleHandleA", "sym.imp.MSVCRT.dll_memset", "sym.imp.MSVCRT.dll_strlen",
            "sym.imp.MSVCRT.dll_memmove", "sym.imp.KERNEL32.dll_LoadLibraryA",
            "unk.0x430278"}
        expected_links = {
            ("entry0", "fcn.0040615c"),
            ("entry0", "sub.KERNEL32.dll_HeapCreate"),
            ("entry0", "fcn.004060f8"),
            ("entry0", "sub.MSVCRT.dll_memset"),
            ("entry0", "main"),
            ("entry0", "fcn.0040611f"),
            ("entry0", "fcn.004064b0"),
            ("entry0", "unk.0x430278"),
            ("sub.MSVCRT.dll_memset", "sym.imp.MSVCRT.dll_memset"),
            ("main", "sym.imp.KERNEL32.dll_GetModuleHandleA"),
            ("sub.KERNEL32.dll_HeapCreate", "sym.imp.KERNEL32.dll_HeapCreate"),
            ("fcn.004064b0", "sym.imp.KERNEL32.dll_HeapCreate"),
            ("fcn.004064b0", "sym.imp.KERNEL32.dll_HeapAlloc"),
            ("fcn.004060f8", "fcn.0040629f"),
            ("fcn.004060f8", "fcn.004062e8"),
            ("fcn.0040611f", "sym.imp.KERNEL32.dll_LoadLibraryA"),
            ("fcn.0040611f", "fcn.004061ba"),
            ("fcn.0040615c", "fcn.0040623f"),
            ("fcn.0040615c", "sym.imp.KERNEL32.dll_GetProcAddress"),
            ("fcn.00406560", "sym.imp.KERNEL32.dll_HeapAlloc"),
            ("fcn.00406560", "fcn.00406660"),
            ("fcn.00406560", "sub.MSVCRT.dll_strlen"),
            ("sub.MSVCRT.dll_strlen", "sym.imp.MSVCRT.dll_strlen"),
            ("fcn.004065b0", "sym.imp.KERNEL32.dll_HeapFree"),
            ("fcn.004065b0", "fcn.00406660"),
            ("fcn.004065b0", "sym.imp.KERNEL32.dll_HeapReAlloc"),
            ("fcn.004065b0", "sub.MSVCRT.dll_strlen"),
            ("fcn.004065b0", "sym.imp.KERNEL32.dll_HeapAlloc"),
            ("fcn.0040618c", "fcn.00406335"),
            ("fcn.0040618c", "fcn.00406310"),
            ("fcn.00406264", "sub.MSVCRT.dll_memset"),
            ("fcn.00406264", "fcn.004063b5"),
            ("fcn.004063b5", "sym.imp.KERNEL32.dll_HeapFree"),
            ("fcn.004062e8", "sym.imp.KERNEL32.dll_HeapAlloc"),
            ("fcn.004061ba", "sym.imp.KERNEL32.dll_HeapReAlloc"),
            ("fcn.004061ba", "sym.imp.KERNEL32.dll_HeapAlloc"),
            ("fcn.004061ba", "fcn.00406388"),
            ("fcn.00406388", "sym.imp.KERNEL32.dll_HeapAlloc"),
            ("fcn.0040629f", "sym.imp.KERNEL32.dll_HeapAlloc"),
            ("fcn.00406008", "fcn.004065b0"),
            ("fcn.00402000", "fcn.00406040"),
            ("fcn.00402000", "fcn.00406530"),
            ("fcn.00402000", "fcn.004066a6"),
            ("fcn.00402000", "fcn.00406000"),
            ("fcn.00406000", "fcn.00406560"),
            ("fcn.004066a6", "sub.MSVCRT.dll_memmove"),
            ("fcn.00406530", "sym.imp.KERNEL32.dll_HeapFree"),
            ("sub.MSVCRT.dll_memmove", "sym.imp.MSVCRT.dll_memmove")}
        actual_nodes = set([node.label for node in nodes])
        actual_links = {(caller.label, callee.label) for caller in nodes for callee in caller.calls}
        self.assertEqual(expected_nodes, actual_nodes)
        self.assertEqual(expected_links, actual_links)
