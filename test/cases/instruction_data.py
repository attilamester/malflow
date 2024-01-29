from dataclasses import dataclass
from typing import Optional, List

from core.model.instruction import InstructionParameter, InstructionPrefix


@dataclass
class InstructionData:
    disasm: str
    mnemonic: str
    prefix: Optional["InstructionPrefix"]
    parameters: List["InstructionParameter"]
    has_bnd: bool


INSTRUCTIONS = [
    InstructionData(**{
        "disasm": "bnd jl 0x414a9e", "mnemonic": "jl", "has_bnd": True, "prefix": None,
        "parameters": [InstructionParameter.CONSTANT]
    }),
    InstructionData(**{
        "disasm": "adc byte [edi - 0x5e], cl", "mnemonic": "adc", "has_bnd": False, "prefix": None,
        "parameters": [InstructionParameter.ADDRESS, InstructionParameter.REGISTER]
    }),
    InstructionData(**{
        "disasm": "call method std::string::operator[](unsigned int) const", "mnemonic": "call", "has_bnd": False,
        "prefix": None, "parameters": [InstructionParameter.FUNCTION]
    }),
    InstructionData(**{
        "disasm": "call method std::basic_ios<char, std::char_traits<char> >::widen(char) const", "mnemonic": "call",
        "has_bnd": False, "prefix": None, "parameters": [InstructionParameter.FUNCTION]
    }),
    InstructionData(**{
        "disasm": "lcall 0, 0x22", "mnemonic": "call", "has_bnd": False, "prefix": None,
        "parameters": [InstructionParameter.ADDRESS_FAR]
    }),
    InstructionData(**{
        "disasm": "ljmp 4:0xc2811a31", "mnemonic": "jmp", "has_bnd": False, "prefix": None,
        "parameters": [InstructionParameter.ADDRESS_FAR]
    }),
    InstructionData(**{
        "disasm": "notrack jmp 0xfb7508c5", "mnemonic": "jmp", "has_bnd": False, "prefix": InstructionPrefix.NOTRACK,
        "parameters": [InstructionParameter.CONSTANT]
    }),
    InstructionData(**{
        "disasm": "bnd notrack jmp 0xfb7508c5", "mnemonic": "jmp", "has_bnd": True, "prefix": InstructionPrefix.NOTRACK,
        "parameters": [InstructionParameter.CONSTANT]
    }),
    InstructionData(**{
        "disasm": "notrack bnd jmp 0xfb7508c5", "mnemonic": "jmp", "has_bnd": True, "prefix": InstructionPrefix.NOTRACK,
        "parameters": [InstructionParameter.CONSTANT]
    })
]
