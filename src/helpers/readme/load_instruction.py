import pickle
from typing import List

from core.model.instruction import Instruction
from util.compression import Compressor


def load_instruction_pickle(pickle_path) -> List[Instruction]:
    with open(pickle_path, "rb") as f:
        content = f.read()
    compressor = Compressor.get_decompressor(content)
    instructions = pickle.loads(
        compressor.decompress(content))  # [['push ebp', b'55', []], ['mov ebp, esp', b'8bec', []]]
    return [Instruction.decompress(i) for i in instructions]


"""
# from core.model.call_graph_image import CallGraphImage
# CallGraphImage.encode_instruction_rgb(instr)
"""
