from enum import Enum
from typing import List, Optional

from core.model.radare2_definitions import get_function_types, is_symbol_flag, get_class_attribute_types


class Registers(Enum):
    """
    X86Mapping.c: static const name_map reg_name_maps[]
    """
    GENERAL_PURPOSE_64 = {"rax", "rbx", "rcx", "rdx", "rsp", "rbp", "rdi", "rsi"}
    GENERAL_PURPOSE_32 = {"eax", "ebx", "ecx", "edx", "esp", "ebp", "edi", "esi"}
    GENERAL_PURPOSE_16 = {"ax", "bx", "cx", "dx", "sp", "bp", "di", "si"}
    GENERAL_PURPOSE_8 = {"ah", "al", "bh", "bl", "ch", "cl", "dh", "dl", "spl", "bpl", "dil", "sil"}

    GENERAL_PURPOSE_EXT_64 = {"r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"}
    GENERAL_PURPOSE_EXT_32 = {"r8d", "r9d", "r10d", "r11d", "r12d", "r13d", "r14d", "r15d"}
    GENERAL_PURPOSE_EXT_16 = {"r8w", "r9w", "r10w", "r11w", "r12w", "r13w", "r14w", "r15w"}
    GENERAL_PURPOSE_EXT_8 = {"r8b", "r9b", "r10b", "r11b", "r12b", "r13b", "r14b", "r15b"}

    IP = {"ip", "eip", "rip"}
    SEGMENT = {"ss", "cs", "ds", "es", "fs", "gs"}
    FLAGS = {"flags", "eflags", "rflags"}

    MMX_64 = {"mm0", "mm1", "mm2", "mm3", "mm4", "mm5", "mm6", "mm7"}
    SSE_128 = {"xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8", "xmm9", "xmm10", "xmm11",
               "xmm12", "xmm13", "xmm14", "xmm15", "xmm16", "xmm17", "xmm18", "xmm19", "xmm20", "xmm21", "xmm22",
               "xmm23", "xmm24", "xmm25", "xmm26", "xmm27", "xmm28", "xmm29", "xmm30", "xmm31"}
    SSE_256 = {"ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11",
               "ymm12", "ymm13", "ymm14", "ymm15", "ymm16", "ymm17", "ymm18", "ymm19", "ymm20", "ymm21", "ymm22",
               "ymm23", "ymm24", "ymm25", "ymm26", "ymm27", "ymm28", "ymm29", "ymm30", "ymm31"}
    SSE_512 = {"zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11",
               "zmm12", "zmm13", "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21", "zmm22",
               "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29", "zmm30", "zmm31"}
    FPU = {"st(0)", "st(1)", "st(2)", "st(3)", "st(4)", "st(5)", "st(6)", "st(7)",
           "fp0", "fp1", "fp2", "fp3", "fp4", "fp5", "fp6", "fp7",
           "fpsw"}

    DEBUG = {"dr0", "dr1", "dr2", "dr3", "dr4", "dr5", "dr6", "dr7", "dr8", "dr9", "dr10", "dr11", "dr12", "dr13",
             "dr14", "dr15"}
    CONTROL = {"cr0", "cr1", "cr2", "cr3", "cr4", "cr5", "cr6", "cr7", "cr8", "cr9", "cr10", "cr11", "cr12", "cr13",
               "cr14", "cr15"}
    PSUDO = {"eiz", "riz"}
    MPX = {"bnd0", "bnd1", "bnd2", "bnd3"}  # memory protection extension
    KERNEL = {"k0", "k1", "k2", "k3", "k4", "k5", "k6", "k7"}  # kernel registers


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
