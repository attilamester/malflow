from enum import Enum
from typing import List, Optional, Set, Dict

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


class Instruction:
    opcode: bytes
    mnemonic: str  # e.g. mov, sub
    registries: List[str] = []
    constants: List[float] = []
    addresses: List[RVA] = []

    def __init__(self, mnemonic: str = None, opcode: bytes = None):
        if mnemonic:
            self.mnemonic = Instruction.standardize_mnemonic(mnemonic)
        if opcode:
            self.opcode = opcode

    def __str__(self):
        return f"<{self.mnemonic}>"

    def __repr__(self):
        return str(self)

    @staticmethod
    def standardize_mnemonic(mnemonic):
        if mnemonic in ["jz", "jnz", "repz", "repnz", "cmovz", "cmovnz", "loopz", "loopnz", "setn", "setnz"]:
            mnemonic = mnemonic[:-1] + "e"

        if mnemonic in ["pushal", "pushaw"]:
            mnemonic = mnemonic[:-1]

        if mnemonic == "retn":
            mnemonic = "ret"
        if mnemonic in ["ea", "odsd"]:
            mnemonic = f"l{mnemonic}"
        return mnemonic


class CGNode:
    """
    Represent a function
    """
    label: str
    rva: Optional[RVA]
    instructions: List[Instruction]
    calls: Set["CGNode"]
    type: FunctionType

    def __init__(self, label: str, rva: str):
        self.label = label
        self.instructions = []

        if label.startswith("fcn.") or label.startswith("entry") or label.startswith("sub.") or label.startswith(
                "unk.") or label.startswith("section") or label == "main":
            self.type = FunctionType.SUBROUTINE
        elif label.startswith("sym."):
            self.type = FunctionType.DLL
        else:
            Logger.warning(f"UNKNOWN node type: {label} at {rva}")
            self.type = FunctionType.STATIC_LINKED_LIB

        self.rva = RVA(rva) if rva else None
        self.calls = set()

    def add_call_to(self, node: "CGNode"):
        self.calls.add(node)

    def add_instructions(self, instruction_string: str):
        """
        :param instruction_string: resulting from radare2 command `agf <address>`
        :return:
        """
        parts = instruction_string.split("      ")
        for i, instr in enumerate(parts):
            if i == 0:  # the first part is just comment
                continue
            instr = instr.strip()
            if not instr or instr[0] == ";":
                continue
            per_l = instr.find("\\l")
            if per_l > 0:
                instr_ = instr[: instr.index("\\l")]
            else:
                instr_ = instr

            tokens = instr_.split(" ", maxsplit=1)
            if not tokens[0]:
                continue

            i = Instruction(tokens[0])

            self.instructions.append(i)

    def set_instructions_from_function_disassembly(self, pdfj: Dict):
        """
        :param pdfj: output of `pdfj` command on a function address
        """
        self.instructions = []
        for op in pdfj["ops"]:
            self.instructions.append(Instruction(mnemonic=op["type"], opcode=op["bytes"].encode()))

    def __str__(self):
        return f"CGNode({self.label}, {self.rva}, {self.type})"
