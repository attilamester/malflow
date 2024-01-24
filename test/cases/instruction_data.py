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
        "disasm": "bnd jl 0x414a9e",
        "mnemonic": "jl",
        "has_bnd": True,
        "prefix": None,
        "parameters": [InstructionParameter.CONSTANT]
    }),
    InstructionData(**{
        "disasm": "adc byte [edi - 0x5e], cl",
        "mnemonic": "adc",
        "has_bnd": False,
        "prefix": None,
        "parameters": [InstructionParameter.ADDRESS, InstructionParameter.REGISTER]
    })
]
