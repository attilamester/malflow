from dataclasses import dataclass
from typing import Optional, List

from malflow.core.model.instruction import InstructionParameter, InstructionPrefix


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
    }),
    InstructionData(**{
        "disasm": "mov dword [esp], dbg.std::basic_ostream<char, std::char_traits<char> >& std::endl<char, std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&)",
        "mnemonic": "mov", "has_bnd": False, "prefix": None,
        "parameters": [InstructionParameter.ADDRESS, InstructionParameter.FUNCTION]
    }),
    InstructionData(**{
        "disasm": "movsd dword [esp], dbg.std::basic_ostream<char, std::char_traits<char> >& std::endl<char, std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&)",
        "mnemonic": "movsd", "has_bnd": False, "prefix": None,
        "parameters": [InstructionParameter.ADDRESS, InstructionParameter.FUNCTION]
    }),
    InstructionData(**{
        "disasm": "vpunpcklbw zmm2 {k7} {z}, zmm5, zmmword [edx + 0x140]]",
        "mnemonic": "vpunpcklbw", "has_bnd": False, "prefix": None,
        "parameters": [InstructionParameter.REGISTER, InstructionParameter.REGISTER, InstructionParameter.ADDRESS]
    }),
    InstructionData(**{
        "disasm": "pushal",
        "mnemonic": "pushal", "has_bnd": False, "prefix": None,
        "parameters": []
    })
]

INSTRUCTION_PARAMETER_TOKENS = [
    ("param", ["param"]),
    ("param with space", ["param with space"]),
    ("param1, param2", ["param1", "param2"]),
    ("param1, param2 param3", ["param1", "param2 param3"]),
    ("param with space, and one comma", ["param with space", "and one comma"]),
    ("param with special char <, and one comma", ["param with special char <, and one comma"]),
    (
        "dword [esp], dbg.std::basic_ostream<char, std::char_traits<char> >& std::endl<char, std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&)",
        [
            "dword [esp]",
            "dbg.std::basic_ostream<char, std::char_traits<char> >& std::endl<char, std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&)"
        ]
    ),
    (
        "zmm2 {k7} {z}, zmm5, zmmword [edx + 0x140]",
        [
            "zmm2 {k7} {z}",
            "zmm5",
            "zmmword [edx + 0x140]"
        ]
    )
]
