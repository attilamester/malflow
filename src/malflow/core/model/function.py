from enum import Enum
from typing import List, Optional, Dict, ValuesView

from malflow.core.model.instruction import Instruction, InstructionParameter
from malflow.core.model.rva import RVA
from malflow.util.logger import Logger


class FunctionType(Enum):
    SUBROUTINE = "subroutine"
    DLL = "dll"
    STATIC_LINKED_LIB = "static_linked_lib"


class CGNode:
    """
    Represents a function
    """
    label: str
    rva: Optional[RVA]
    instructions: List[Instruction]
    calls: Dict[str, "CGNode"]
    type: FunctionType

    def __init__(self, label: str, rva: str):
        self.label = label
        self.instructions = []
        if label.startswith("sym."):
            self.type = FunctionType.DLL
        elif (InstructionParameter.is_function(label) or
              InstructionParameter.is_section(label) or
              InstructionParameter.is_block(label)):
            self.type = FunctionType.SUBROUTINE
        else:
            Logger.warning(f"UNKNOWN node type: {label} at {rva}")
            self.type = FunctionType.SUBROUTINE

        self.rva = RVA(rva) if rva else None
        self.calls = {}

    def get_calls(self) -> ValuesView["CGNode"]:
        return self.calls.values()

    def add_call_to(self, node: "CGNode"):
        if node.label in self.calls:
            if node.rva.addr != self.calls[node.label].rva.addr:
                raise Exception(f"Conflict while adding call to {node} ; existing {self.calls[node.label]}")
        else:
            self.calls[node.label] = node

    def has_call_to(self, node: "CGNode"):
        return node.label in self.calls

    def set_instructions_from_function_disassembly(self, pdfj: Dict):
        """
        :param pdfj: output of `pdfj` command on a function address
        """
        self.instructions = []
        for op in pdfj["ops"]:
            if "disasm" not in op or "bytes" not in op:
                continue
            refs = op.get("refs", []) if ("call" in op["type"] or "jmp" in op["type"]) else []
            try:
                instruction = Instruction(op["disasm"], op["bytes"].encode(), refs, op.get("addr", None))
                self.instructions.append(instruction)
            except Exception as e:
                Logger.error(f"Skipping instruction on {self.rva}. Could not process instruction: {e}")

    def __str__(self):
        return f"CGNode({self.label}, {self.rva}, {self.type})"

    def __eq__(self, other):
        if isinstance(other, CGNode):
            return (self.label == other.label and
                    self.rva == other.rva and
                    self.type == other.type and
                    self.instructions == other.instructions and
                    list(self.calls.keys()) == list(other.calls.keys()))
        return False
