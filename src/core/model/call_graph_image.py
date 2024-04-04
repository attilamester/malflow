from functools import lru_cache
from typing import List, Tuple

import numpy as np

from core.model import CallGraph
from core.model.instruction import Instruction, InstructionPrefix, InstructionParameter
from core.model.radare2_definitions import Mnemonics


def get_instruction_token_RG(mnemonic: str, prefix: str, bnd: bool) -> str:
    return f"{mnemonic}{f'_{prefix}' if prefix else ''}{'_bnd' if bnd else ''}"


def get_intruction_token_B(parameters: List[InstructionParameter]) -> str:
    return ",".join([p.value for p in parameters[:2]])


def split_instruction_token_RG(token: str) -> Tuple[str, str, bool]:
    tokens = token.split("_")
    mnemonic = tokens[0]
    if len(tokens) == 1:
        return mnemonic, "", False

    prefix = tokens[1]
    if prefix == "bnd":
        return mnemonic, "", True
    if len(tokens) == 1:
        return mnemonic, prefix, False
    return mnemonic, prefix, True


def split_instruction_token_B(token: str) -> List[InstructionParameter]:
    return [InstructionParameter(p) for p in token.split(",") if p]


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

    @staticmethod
    def get_image_from_pixels(img_size, pixels):
        np_pixels = np.array([[int(channel) for channel in pixel] for pixel in pixels], dtype=np.uint8)
        np_pixels.resize((img_size[0] * img_size[1] * 3), refcheck=False)
        np_pixels = np.reshape(np_pixels, (*img_size, 3))

        return np_pixels

    @staticmethod
    def get_image_from_instructions(img_size, instructions: List[Instruction]):
        pixels = [CallGraphImage.encode_instruction_rgb(i) for i in instructions]

        return CallGraphImage.get_image_from_pixels(img_size, pixels)

    def get_image(self, img_size: Tuple[int, int] = (512, 512), **dfs_kwargs) -> Tuple[np.ndarray, int]:
        instructions = self.cg.DFS_instructions(max_instructions=img_size[0] * img_size[1], **dfs_kwargs)
        np_pixels = CallGraphImage.get_image_from_instructions(img_size, instructions)

        return np_pixels, len(instructions)

    @staticmethod
    def encode_instruction_rgb(i: Instruction) -> bytes:
        insruction_token = get_instruction_token_RG(i.mnemonic, i.prefix.value if i.prefix else "", i.has_bnd)
        parameter_token = get_intruction_token_B(i.parameters)
        rg_value = MNEMONIC_PREFIX_BND_INDEX[insruction_token]
        b_value = PARAMETERIZATION_INDEX[parameter_token]

        rg = rg_value.to_bytes(2, byteorder="big")
        b = b_value.to_bytes(1, byteorder="big")
        return rg + b[:]

    @staticmethod
    @lru_cache(maxsize=None)
    def decode_rgb(r: int = None, g: int = None, b: int = None, rgb: bytes = None) -> Instruction:
        if rgb:
            rg_value = int.from_bytes(rgb[:2], byteorder="big")
            b_value = int.from_bytes(rgb[2:], byteorder="big")
        else:
            rg_value = r * 256 + g
            b_value = b
        rg_token = list(MNEMONIC_PREFIX_BND_INDEX.keys())[rg_value]
        b_token = list(PARAMETERIZATION_INDEX.keys())[b_value]
        mnemonic, prefix, bnd = split_instruction_token_RG(rg_token)
        parameters = split_instruction_token_B(b_token)
        i = Instruction("nop", b"0", [])
        i.mnemonic = mnemonic
        i.prefix = InstructionPrefix(prefix) if prefix else None
        i.has_bnd = bnd
        i.parameters = parameters
        return i
