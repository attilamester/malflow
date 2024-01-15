from enum import Enum
from typing import List, Optional, Set, Dict

from core.model.instruction import Instruction, InstructionParameter
from util.logger import Logger


class FunctionType(Enum):
    SUBROUTINE = "subroutine"
    DLL = "dll"
    STATIC_LINKED_LIB = "static_linked_lib"


class RVA:
    value: str  # e.g. "0x<hex-string>"

    def __init__(self, rva: str):
        self.value = rva.lower()
        if not self.value.startswith("0x"):
            raise Exception(f"Invalid RVA {rva}")
        try:
            int(self.value[2:], 16)
        except:
            raise Exception(f"Invalid RVA {rva}")

    def __str__(self):
        return f"RVA('{self.value}')"

    def __repr__(self):
        return str(self)


class CGNode:
    """
    Represents a function
    """
    label: str
    rva: Optional[RVA]
    instructions: List[Instruction]
    calls: Set["CGNode"]
    type: FunctionType

    def __init__(self, label: str, rva: str):
        self.label = label
        self.instructions = []
        if label.startswith("sym."):
            self.type = FunctionType.DLL
        elif InstructionParameter.is_function(label):
            self.type = FunctionType.SUBROUTINE
        else:
            Logger.warning(f"UNKNOWN node type: {label} at {rva}")
            self.type = FunctionType.STATIC_LINKED_LIB

        self.rva = RVA(rva) if rva else None
        self.calls = set()

    def add_call_to(self, node: "CGNode"):
        self.calls.add(node)

    def set_instructions_from_function_disassembly(self, pdfj: Dict):
        """
        :param pdfj: output of `pdfj` command on a function address
        """
        self.instructions = []
        for op in pdfj["ops"]:
            if "disasm" not in op or "bytes" not in op:
                continue
            self.instructions.append(Instruction(op["disasm"], op["bytes"].encode()))

    def __str__(self):
        return f"CGNode({self.label}, {self.rva}, {self.type})"
