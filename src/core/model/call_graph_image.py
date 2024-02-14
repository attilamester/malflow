from typing import List

from core.model import CallGraph
from core.model.instruction import Instruction, InstructionPrefix, InstructionParameter
from core.model.radare2_definitions import Mnemonics


def get_instruction_token_RG(mnemonic: str, prefix: str, bnd: bool) -> str:
    return f"{mnemonic}{f'_{prefix}' if prefix else ''}{'_bnd' if bnd else ''}"


def get_intruction_token_B(parameters: List[InstructionParameter]) -> str:
    return ",".join([p.value for p in parameters])


PREFIX_INDEX = {prefix.value: i + 1 for i, prefix in enumerate(InstructionPrefix)}
PREFIX_INDEX[""] = 0
MNEMONIC_INDEX = {mnemonic: i for i, mnemonic in enumerate(sorted(Mnemonics._ALL.value))}
MNEMONIC_PREFIX_BND_INDEX = {token: i for i, token in enumerate(sorted([
    get_instruction_token_RG(mnemonic, prefix, bnd)
    for mnemonic in MNEMONIC_INDEX.keys()
    for prefix in PREFIX_INDEX.keys()
    for bnd in [False, True]]))}
PARAMETER_INDEX = {parameter.value: i + 1 for i, parameter in enumerate(InstructionParameter)}
PARAMETER_INDEX[""] = 0
PARAMETERIZATION_INDEX = {token: i for i, token in enumerate(sorted(
    [""] +
    [f"{p1}"
     for p1 in PARAMETER_INDEX.keys() if p1 != ""] +
    [f"{p1},{p2}"
     for p1 in PARAMETER_INDEX.keys()
     for p2 in PARAMETER_INDEX.keys()
     if not (p1 == "" or p2 == "")]))}


class InstructionEncoder:

    @staticmethod
    def encode_into_RGB(i: Instruction) -> bytes:
        pass


class CallGraphImage:
    cg: CallGraph

    def __init__(self, cg: CallGraph):
        self.cg = cg

    def get_image(self):
        self.cg.get_image()
        pass

    @staticmethod
    def encode_instruction_rgb(i: Instruction) -> bytes:
        insruction_token = get_instruction_token_RG(i.mnemonic, i.prefix.value if i.prefix else "", i.has_bnd)
        parameter_token = get_intruction_token_B(i.parameters)
        rg_value = MNEMONIC_PREFIX_BND_INDEX[insruction_token]
        b_value = PARAMETERIZATION_INDEX[parameter_token]

        rg = rg_value.to_bytes(2, byteorder="big")
        b = b_value.to_bytes(1, byteorder="big")
        return rg + b[:]
