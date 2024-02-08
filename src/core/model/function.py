from enum import Enum
from typing import List, Optional, Dict, ValuesView

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

    def __eq__(self, other):
        if isinstance(other, RVA):
            return self.value == other.value
        return False


class CGNode:
    """
    Represents a function
    """
    label: str
    rva: Optional[RVA]
    instructions: List[Instruction]
    calls: Dict[str, "CGNode"]
    refs: Dict[str, "CGNode"]
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
        self.refs = {}

    def get_calls(self) -> ValuesView["CGNode"]:
        return self.calls.values()

    def add_call_to(self, node: "CGNode"):
        if node.label in self.calls:
            if node.rva.value != self.calls[node.label].rva.value:
                raise Exception(f"Conflict while adding call to {node} ; existing {self.calls[node.label]}")
        else:
            self.calls[node.label] = node

    def add_ref_to(self, node: "CGNode"):
        if node.label in self.refs:
            if node.rva.value != self.refs[node.label].rva.value:
                raise Exception(f"Conflict while adding ref to {node} ; existing {self.calls[node.label]}")
        else:
            self.refs[node.label] = node

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
            self.instructions.append(Instruction(op["disasm"], op["bytes"].encode(), refs))

    # def get_node_calls_from_instructions(self) -> List[Tuple[str, int]]:
    #     """
    #     Correlate the node's calls given the `agCd` and `agRd` with the node's instructions given by `pdfj`
    #     :return: labels of the nodes called by the instructions of the given node, in the order of its instructions
    #     """
    #
    #     if self.type == FunctionType.DLL:
    #         return []
    #
    #     node_calls_labels = [n.label for n in self.get_calls()]
    #     node_calls_labels_from_instructions = []
    #
    #     RAM = {}
    #     STACK = deque()
    #     for k, i in enumerate(self.instructions):
    #         if "call" in i.mnemonic or "jmp" in i.mnemonic:
    #             ind = i.disasm.index(i.mnemonic)
    #             callee = i.disasm[ind + len(i.mnemonic):]
    #             for label in node_calls_labels:
    #                 _label = label.replace("unk.", "")
    #                 if _label in callee or _label in RAM.get(callee, ""):
    #                     node_calls_labels_from_instructions.append((label, k))
    #                     break
    #         else:
    #             if i.mnemonic == "mov":
    #                 parameter_tokens = i.disasm.replace("mov", "").split(", ", maxsplit=1)
    #                 if len(parameter_tokens) != 2:
    #                     continue
    #                 RAM[parameter_tokens[0]] = RAM.get(parameter_tokens[1], parameter_tokens[1])
    #             elif i.mnemonic == "push":
    #                 STACK.append(i.disasm.replace("push", "").strip())
    #             elif i.mnemonic == "pop":
    #                 if STACK:
    #                     RAM[i.disasm.replace("pop", "").strip()] = STACK.pop()
    #
    #     # if set(node_calls_labels_from_instructions) != set(node_calls_labels):
    #     #     raise Exception(
    #     #         f"For {self} could not find some callees: \n"
    #     #         f"{node_calls_labels} != \n"
    #     #         f"{node_calls_labels_from_instructions}")
    #     return node_calls_labels_from_instructions

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
