from functools import lru_cache
from typing import List, Tuple, Type

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
    if len(tokens) == 2:
        return mnemonic, prefix, False
    return mnemonic, prefix, True


def split_instruction_token_B(token: str) -> List[InstructionParameter]:
    return [InstructionParameter(p) for p in token.split(",") if p]


PREFIX_INDEX = {prefix.value: i + 1 for i, prefix in enumerate(InstructionPrefix)}
PREFIX_INDEX[""] = 0
MNEMONIC_INDEX = {mnemonic: i for i, mnemonic in enumerate(sorted(Mnemonics._ALL.value))}
MNEMONIC_INDEX_INVERSE = {i: mnemonic for mnemonic, i in MNEMONIC_INDEX.items()}
MNEMONIC_PREFIX_BND_INDEX = {token: i for i, token in enumerate(sorted([
    get_instruction_token_RG(mnemonic, prefix, bnd)
    for mnemonic in MNEMONIC_INDEX.keys()
    for prefix in PREFIX_INDEX.keys()
    for bnd in [False, True]]))}
MNEMONIC_PREFIX_BND_INDEX_INVERSE = {i: token for token, i in MNEMONIC_PREFIX_BND_INDEX.items()}
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
PARAMETERIZATION_INDEX_INVERSE = {i: token for token, i in PARAMETERIZATION_INDEX.items()}


class InstructionEncoder:

    @classmethod
    def encode(cls, i: Instruction) -> bytes:
        raise NotImplementedError()

    @classmethod
    def _decode(cls, r: int, g: int, b: int) -> Instruction:
        raise NotImplementedError()

    @classmethod
    def decode(cls, r: int = None, g: int = None, b: int = None, rgb: bytes = None):
        if rgb:
            r = int.from_bytes(rgb[:1], byteorder="big")
            g = int.from_bytes(rgb[1:2], byteorder="big")
            b = int.from_bytes(rgb[2:], byteorder="big")
        return cls._decode(r, g, b)


class InstructionEncoderComplete(InstructionEncoder):
    @classmethod
    def encode(cls, i: Instruction) -> bytes:
        insruction_token = get_instruction_token_RG(i.mnemonic, i.prefix.value if i.prefix else "", i.has_bnd)
        parameter_token = get_intruction_token_B(i.parameters)
        rg_value = MNEMONIC_PREFIX_BND_INDEX[insruction_token]
        b_value = PARAMETERIZATION_INDEX[parameter_token]

        rg = rg_value.to_bytes(2, byteorder="big")
        b = b_value.to_bytes(1, byteorder="big")
        return rg + b[:]

    @classmethod
    @lru_cache(maxsize=None)
    def _decode(cls, r: int, g: int, b: int):
        rg_value = r * 256 + g
        b_value = b

        rg_token = MNEMONIC_PREFIX_BND_INDEX_INVERSE[rg_value]
        b_token = PARAMETERIZATION_INDEX_INVERSE[b_value]

        mnemonic, prefix, bnd = split_instruction_token_RG(rg_token)
        parameters = split_instruction_token_B(b_token)

        i = Instruction("nop", b"0", [])
        i.mnemonic = mnemonic
        i.prefix = InstructionPrefix(prefix) if prefix else None
        i.has_bnd = bnd
        i.parameters = parameters
        return i


class InstructionEncoderMnemonic(InstructionEncoder):
    @classmethod
    def encode(cls, i: Instruction) -> bytes:
        gb_value = MNEMONIC_INDEX[i.mnemonic]
        return gb_value.to_bytes(3, byteorder="big")

    @classmethod
    @lru_cache(maxsize=None)
    def _decode(cls, r: int, g: int, b: int):
        if r != 0:
            raise ValueError(f"R channel should be zero: {r} {g} {b}")
        else:
            rgb_value = g * 256 + b

        i = Instruction("nop", b"0", [])
        i.mnemonic = MNEMONIC_INDEX_INVERSE[rgb_value]
        return i


class InstructionEncoderMnemonicPrefixBnd(InstructionEncoder):

    @classmethod
    def encode(cls, i: Instruction) -> bytes:
        insruction_token = get_instruction_token_RG(i.mnemonic, i.prefix.value if i.prefix else "", i.has_bnd)
        gb_value = MNEMONIC_PREFIX_BND_INDEX[insruction_token]
        return gb_value.to_bytes(3, byteorder="big")

    @classmethod
    @lru_cache(maxsize=None)
    def _decode(cls, r: int, g: int, b: int):
        if r != 0:
            raise ValueError(f"R channel should be zero: {r} {g} {b}")
        else:
            rgb_value = g * 256 + b

        rg_token = MNEMONIC_PREFIX_BND_INDEX_INVERSE[rgb_value]
        mnemonic, prefix, bnd = split_instruction_token_RG(rg_token)

        i = Instruction("nop", b"0", [])
        i.mnemonic = mnemonic
        i.prefix = InstructionPrefix(prefix) if prefix else None
        i.has_bnd = bnd
        return i


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
    def get_image_from_instructions(img_size, instructions: List[Instruction],
                                    instruction_encoder: Type[InstructionEncoder] = Type[InstructionEncoderComplete]):
        pixels = [instruction_encoder.encode(i) for i in instructions]

        return CallGraphImage.get_image_from_pixels(img_size, pixels)

    def get_image(self, img_size: Tuple[int, int] = (512, 512), **dfs_kwargs) -> Tuple[np.ndarray, int]:
        instructions = self.cg.DFS_instructions(max_instructions=img_size[0] * img_size[1], **dfs_kwargs)
        np_pixels = CallGraphImage.get_image_from_instructions(img_size, instructions)

        return np_pixels, len(instructions)

    @staticmethod
    def encode_instruction_rgb(i: Instruction) -> bytes:
        return InstructionEncoderComplete.encode(i)

    @staticmethod
    @lru_cache(maxsize=None)
    def decode_rgb(r: int = None, g: int = None, b: int = None, rgb: bytes = None) -> Instruction:
        return InstructionEncoderComplete.decode(r, g, b, rgb)
