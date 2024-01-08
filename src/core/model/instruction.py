from enum import Enum
from typing import List, Optional


class Registers(Enum):
    GENERAL_PURPOSE_64 = {"rax", "rbx", "rcx", "rdx", "rsp", "rbp", "rdi", "rsi"}
    GENERAL_PURPOSE_32 = {"eax", "ebx", "ecx", "edx", "esp", "ebp", "edi", "esi"}
    GENERAL_PURPOSE_16 = {"ax", "bx", "cx", "dx", "sp", "bp", "di", "si"}
    GENERAL_PURPOSE_8 = {"ah", "al", "bh", "bl", "ch", "cl", "dh", "dl", "spl", "bpl", "dil", "sil"}

    GENERAL_PURPOSE_EXT_64 = {"r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"}
    GENERAL_PURPOSE_EXT_32 = {"r8d", "r9d", "r10d", "r11d", "r12d", "r13d", "r14d", "r15d"}
    GENERAL_PURPOSE_EXT_16 = {"r8w", "r9w", "r10w", "r11w", "r12w", "r13w", "r14w", "r15w"}
    GENERAL_PURPOSE_EXT_8 = {"r8b", "r9b", "r10b", "r11b", "r12b", "r13b", "r14b", "r15b"}

    SEGMENT = {"ss", "cs", "ds", "es", "fs", "gs"}
    FLAGS = {"eflags", "rflags"}

    MMX_64 = {"mm0", "mm1", "mm2", "mm3", "mm4", "mm5", "mm6", "mm7"}
    SSE_128 = {"xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"}
    FPU = {"st(0)", "st(1)", "st(2)", "st(3)", "st(4)", "st(5)", "st(6)", "st(7)"}


class Instruction:
    disasm: str  # full instruction string, e.g. `mov eax, 0xc`
    opcode: bytes  # e.g. `0x83c40c`
    mnemonic: str  # e.g. `mov`
    prefix: Optional["InstructionPrefix"]
    parameters: List["InstructionParameter"]

    def __init__(self, disasm: str, opcode: bytes):
        self.disasm = disasm
        self.opcode = opcode
        self.prefix = None
        self.parameters = []
        self.process()

    def process(self):
        opcode_tokens = self.disasm.split(" ", maxsplit=1)
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
        if token.startswith("sub") or token.startswith("fcn") or token.startswith("main") or token.startswith("entry"):
            return InstructionParameter.FUNCTION
        if token.startswith("str"):
            return InstructionParameter.STRING
        if token.startswith("case"):
            return InstructionParameter.BLOCK
        if token.startswith("section"):
            return InstructionParameter.ADDRESS
        try:
            int(token)
            return InstructionParameter.CONSTANT
        except:
            pass
        raise Exception(f"Undefined instruction parameter type `{token}`")
