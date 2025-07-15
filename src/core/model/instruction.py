from collections import deque
from enum import Enum
from typing import List, Optional, Dict, TypedDict

from core.model.radare2_definitions import (get_function_types, is_symbol_flag, get_class_attribute_types,
                                            Registers, Mnemonics)
from util.logger import Logger


class Instruction:
    disasm: str  # full instruction string, e.g. `mov eax, 0xc`
    opcode: bytes  # e.g. `0x83c40c`
    mnemonic: str  # e.g. `mov`
    prefix: Optional["InstructionPrefix"]
    parameters: List["InstructionParameter"]
    has_bnd: bool  # typedef struct r_x86nz_opcode_t
    refs: List["InstructionReference"]

    def __init__(self, disasm: str, opcode: bytes, refs: List[TypedDict("ref", {"addr": int, "type": str})]):
        self.disasm = disasm
        self.opcode = opcode
        self.prefix = None
        self.parameters = []
        self.has_bnd = False  # MPX - used to check the bounds of memory addresses used by the instruction
        self.refs = []
        try:
            self.process()
            self.process_refs(refs)
        except Exception as e:
            Logger.error(f"Could not process instruction `{disasm}`")
            raise e

    def process(self):
        opcode_tokens = self.disasm.split(" ", maxsplit=2)
        if opcode_tokens[0] == "bnd" or (len(opcode_tokens) > 1 and opcode_tokens[1] == "bnd"):
            # bnd prefix may not be the first token
            opcode_tokens = (self.disasm
                             .replace("bnd ", "")
                             .split(" ", maxsplit=1))
            self.has_bnd = True
        else:
            opcode_tokens = self.disasm.split(" ", maxsplit=1)

        if opcode_tokens[0] in InstructionPrefixes:
            self.prefix = InstructionPrefix(opcode_tokens[0])
            opcode_tokens = opcode_tokens[1].split(" ", maxsplit=1)

        self.mnemonic = Instruction.standardize_mnemonic(opcode_tokens[0])
        parameter_tokens = []
        if len(opcode_tokens) == 2:
            # the mnemonic is followed by parameter(s)
            if opcode_tokens[0] in {"callf", "lcall", "jmpf", "ljmp"}:  # far call
                if "," in opcode_tokens[1] or ":" in opcode_tokens[1]:
                    self.parameters = [InstructionParameter.ADDRESS_FAR]
                    return
            elif self.mnemonic in {"call", "jmp"}:
                parameter_tokens = [opcode_tokens[1]]
            else:
                parameter_tokens = InstructionParameter.split_into_parameter_tokens(opcode_tokens[1])
        self.parameters = [InstructionParameter.construct(token) for token in parameter_tokens]

    def process_refs(self, refs: List[Dict]):
        for ref in refs:
            self.refs.append(InstructionReference(ref["addr"], ref["type"]))

    def get_fmt(self) -> str:
        return ("[bnd] " if self.has_bnd else "") + \
            (f"[{self.prefix.value}] " if self.prefix else "") + \
            f"[{self.mnemonic}]" + \
            (" " + ", ".join([p.value for p in self.parameters]) if self.parameters else "")

    @staticmethod
    def parse_fmt(fmt: str) -> "Instruction":
        i = Instruction("nop", b"", [])
        parentheses = fmt.count("[")

        def process_mnemonic_params(token):
            tokens = token.split("]")
            i.mnemonic = tokens[0][1:]
            if len(tokens) == 2:
                i.parameters = [InstructionParameter(p) for p in tokens[1].strip().split(", ") if p]

        def process_prefix_mnemonic_params(token):
            tokens = token.split("] ", maxsplit=1)
            i.prefix = InstructionPrefix(tokens[0][1:])
            process_mnemonic_params(tokens[1])

        if parentheses == 1:
            process_mnemonic_params(fmt)
        elif parentheses == 2:
            if fmt.startswith("[bnd]"):
                fmt = fmt.replace("[bnd] ", "")
                process_mnemonic_params(fmt)
                i.has_bnd = True
            else:
                process_prefix_mnemonic_params(fmt)
        elif parentheses == 3:
            i.has_bnd = True
            fmt = fmt.replace("[bnd] ", "")
            process_prefix_mnemonic_params(fmt)

        return i

    def __str__(self):
        return self.get_fmt()

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if isinstance(other, Instruction):
            return (self.disasm == other.disasm and
                    self.opcode == other.opcode and
                    self.has_bnd == other.has_bnd and
                    self.prefix == other.prefix and
                    self.parameters == other.parameters and
                    self.refs == other.refs)
        return False

    @staticmethod
    def standardize_mnemonic(mnemonic):
        if mnemonic in ["jz", "jnz", "repz", "repnz", "cmovz", "cmovnz", "loopz", "loopnz", "setn", "setnz"]:
            return mnemonic[:-1] + "e"

        if mnemonic == "retn":
            return "ret"
        if mnemonic in ["ea", "odsd"]:
            return f"l{mnemonic}"

        if mnemonic[0] in {"s", "l"} and mnemonic[1:] in Mnemonics._ALL.value:
            return mnemonic[1:]

        if mnemonic not in Mnemonics._ALL.value:
            raise Exception(f"Undefined instruction mnemonic `{mnemonic}`")
        return mnemonic

    def compress(self) -> List:
        return [self.disasm, self.opcode, [[ref.addr, ref.type] for ref in self.refs]]

    @staticmethod
    def decompress(i: List) -> "Instruction":
        """
        :param tokens: return value of `compress`
        :return:
        """
        return Instruction(i[0], i[1], [{"addr": e[0], "type": e[1]} for e in i[2]])


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
    NOTRACK = "notrack"  # allow to jump to any address - only for instructions like call/jmp


InstructionPrefixes = {prefix.value for prefix in InstructionPrefix}


class InstructionParameter(Enum):
    CONSTANT = "CONST"
    REGISTER = "REG"
    ADDRESS = "ADDR"  # any address
    ADDRESS_FAR = "ADDR_FAR"  # any address
    FUNCTION = "FUNC"  # function address
    STRING = "STR"  # address of string
    BLOCK = "BLOCK"  # address of block e.g. jump after if

    # TODO: divide types more granularly
    # - register types (by scope & size)
    # - addresses (dword [eax + esi] AND dword [0x430278] -> ADDR_BY_REG - ADDR_BY_CONST)

    @staticmethod
    def construct(token: str) -> "InstructionParameter":
        token = token.lower().strip()
        token_first_word = token.split(" ")[0]
        for register_class in Registers:
            if token in register_class.value or token_first_word in register_class.value:
                return InstructionParameter.REGISTER
        if token.startswith("0x"):
            return InstructionParameter.CONSTANT
        if token.startswith("str"):
            return InstructionParameter.STRING
        if InstructionParameter.is_function(token):
            return InstructionParameter.FUNCTION
        if InstructionParameter.is_block(token):
            return InstructionParameter.BLOCK
        if InstructionParameter.is_section(token) or "[" in token:
            return InstructionParameter.ADDRESS
        if token.startswith("global_") or token.startswith("resource"):
            return InstructionParameter.ADDRESS
        try:
            int(token)
            return InstructionParameter.CONSTANT
        except:
            pass
        raise Exception(f"Undefined instruction parameter type `{token}`")

    @staticmethod
    def split_into_parameter_tokens(parameters: str) -> List[str]:
        # =================
        # Original implementation: split according to `,`
        # - problem: params. may contain comma
        # - e.g. `std::char_traits<char> >& std::endl<char, std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&)`
        #
        # max_splits = MNEMONIC_INFO.get(self.mnemonic, {}).get("max_operands", 0) - 1
        # parameters = opcode_tokens[1].split(",", maxsplit=max_splits)
        #
        # =================
        OPENINGS = ["(", "[", "{", "<"]
        CLOSINGS = [")", "]", "}", ">"]
        stack = deque()
        tokens = []

        def process_token(token: str):
            if not token:
                return

            for i, char in enumerate(token):
                if char in OPENINGS:
                    stack.append(char)
                elif char in CLOSINGS and stack and stack[-1] == OPENINGS[CLOSINGS.index(char)]:
                    stack.pop()
                elif char == "," and not stack:
                    tokens.append(token[:i])
                    process_token(token[i + 1:].strip())
                    return
            tokens.append(token)

        process_token(parameters)

        return tokens

    @staticmethod
    def is_section(token: str):
        return token.startswith("section.")

    @staticmethod
    def is_block(token: str):
        return token.startswith("case.") or token.startswith("switch.") or token.startswith("segment.")

    @staticmethod
    def is_function(token: str):
        if is_symbol_flag(token):
            return True
        return any(
            [token.startswith(t) for t in get_function_types()]) or any(
            [token.startswith(t) for t in get_class_attribute_types()])


class InstructionReferenceType(Enum):
    """
    R_API const char *r_anal_ref_type_tostring(RAnalRefType type)
    """
    CODE = "CODE"
    CALL = "CALL"
    JUMP = "JUMP"
    DATA = "DATA"
    OTHER = "OTHER"

    @classmethod
    def _missing_(cls, value):
        return cls.OTHER


class InstructionReference:
    addr: int
    type: InstructionReferenceType

    def __init__(self, addr: int, type: str):
        self.addr = addr
        self.type = InstructionReferenceType(type)

    def __eq__(self, other):
        if isinstance(other, InstructionReference):
            return (self.addr == other.addr and self.type == other.type)
        return False
