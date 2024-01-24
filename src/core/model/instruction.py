from enum import Enum
from typing import List, Optional

from core.model.radare2_definitions import get_function_types, is_symbol_flag, get_class_attribute_types, Registers


class Instruction:
    disasm: str  # full instruction string, e.g. `mov eax, 0xc`
    opcode: bytes  # e.g. `0x83c40c`
    mnemonic: str  # e.g. `mov`
    prefix: Optional["InstructionPrefix"]
    parameters: List["InstructionParameter"]
    has_bnd: bool  # typedef struct r_x86nz_opcode_t

    def __init__(self, disasm: str, opcode: bytes):
        self.disasm = disasm
        self.opcode = opcode
        self.prefix = None
        self.parameters = []
        self.has_bnd = False  # MPX - used to check the bounds of memory addresses used by the instruction
        self.process()

    def process(self):
        opcode_tokens = self.disasm.split(" ", maxsplit=1)
        if opcode_tokens[0] == "bnd":
            opcode_tokens = opcode_tokens[1].split(" ", maxsplit=1)
            self.has_bnd = True
        self.mnemonic = Instruction.standardize_mnemonic(opcode_tokens[0])
        if self.mnemonic in InstructionPrefixes:
            self.prefix = InstructionPrefix(self.mnemonic)
            opcode_tokens = opcode_tokens[1].split(" ", maxsplit=1)
            self.mnemonic = Instruction.standardize_mnemonic(opcode_tokens[0])

        parameters = []
        if len(opcode_tokens) == 2:
            parameters = opcode_tokens[1].split(",")
        self.parameters = [InstructionParameter.construct(token) for token in parameters]

    def __str__(self):
        return f"<{self.mnemonic}>"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if isinstance(other, Instruction):
            return (self.disasm == other.disasm and
                    self.opcode == other.opcode and
                    self.has_bnd == other.has_bnd and
                    self.prefix == other.prefix and
                    self.parameters == other.parameters)
        return False

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


class InstructionPrefix(Enum):
    SEGCS = "segcs"
    SEGDS = "segds"
    SEGSS = "segss"
    SEGES = "seges"
    LOCK = "lock"
    REP = "rep"
    REPE = "repe"
    REPZ = "repz"
    REPNE = "repne"
    REPNZ = "repnz"


InstructionPrefixes = {prefix.value for prefix in InstructionPrefix}


class InstructionParameter(Enum):
    CONSTANT = "CONST"
    REGISTER = "REG"
    ADDRESS = "ADDR"  # any address
    FUNCTION = "FUNC"  # function address
    STRING = "STR"  # address of string
    BLOCK = "BLOCK"  # address of block e.g. jump after if

    # TODO: divide types more granularly
    # - register types (by scope & size)
    # - addresses (dword [eax + esi] AND dword [0x430278] -> ADDR_BY_REG - ADDR_BY_CONST)

    @staticmethod
    def construct(token: str) -> "InstructionParameter":
        token = token.lower().strip()

        for register_class in Registers:
            if token in register_class.value:
                return InstructionParameter.REGISTER
        if token.startswith("0x"):
            return InstructionParameter.CONSTANT
        if "[" in token:
            return InstructionParameter.ADDRESS
        if InstructionParameter.is_function(token):
            return InstructionParameter.FUNCTION
        if token.startswith("str"):
            return InstructionParameter.STRING
        if InstructionParameter.is_block(token):
            return InstructionParameter.BLOCK
        if token.startswith("section"):
            return InstructionParameter.ADDRESS
        try:
            int(token)
            return InstructionParameter.CONSTANT
        except:
            pass
        raise Exception(f"Undefined instruction parameter type `{token}`")

    @staticmethod
    def is_block(token: str):
        return token.startswith("case.") or token.startswith("switch.")

    @staticmethod
    def is_function(token: str):
        if is_symbol_flag(token):
            return True
        return any(
            [token.startswith(t) for t in get_function_types()]) or any(
            [token.startswith(t) for t in get_class_attribute_types()])
