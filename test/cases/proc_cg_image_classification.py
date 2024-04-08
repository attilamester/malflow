import unittest

from cases.data.instruction_data import INSTRUCTIONS
from core.model.call_graph_image import CallGraphImage
from core.model.instruction import Instruction


class TestProcessorCgImageClassification(unittest.TestCase):

    def test_instruction_to_pixel(self):
        for i_data in INSTRUCTIONS:
            instr = Instruction(i_data.disasm, b"0", [])
            encoded_rgb = CallGraphImage.encode_instruction_rgb(instr)
            decoded_instr = CallGraphImage.decode_rgb(rgb=encoded_rgb)

            instr.parameters = instr.parameters[:2]
            self.assertEqual(str(instr), str(decoded_instr))
