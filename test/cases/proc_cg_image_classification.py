import unittest

from cases.data.instruction_data import INSTRUCTIONS
from core.model.call_graph_image import InstructionEncoderMnemonic, InstructionEncoderComplete, \
    InstructionEncoderMnemonicPrefixBnd
from core.model.instruction import Instruction


class TestProcessorCgImageClassification(unittest.TestCase):

    def test_instruction_to_pixel(self):
        for i_data in INSTRUCTIONS:
            instr = Instruction(i_data.disasm, b"0", [])
            instr.parameters = instr.parameters[:2]

            ie = InstructionEncoderComplete
            encoded_rgb = ie.encode(instr)
            decoded_instr = ie.decode(rgb=encoded_rgb)
            decoded_inst2 = ie.decode(r=encoded_rgb[0], g=encoded_rgb[1], b=encoded_rgb[2])
            self.assertEqual(instr.has_bnd, decoded_instr.has_bnd)
            self.assertEqual(instr.prefix, decoded_instr.prefix)
            self.assertEqual(instr.parameters, decoded_instr.parameters)
            self.assertEqual(instr.mnemonic, decoded_instr.mnemonic)
            self.assertEqual(instr.refs, decoded_instr.refs)
            self.assertEqual(str(instr), str(decoded_inst2))

            ie = InstructionEncoderMnemonic
            encoded_rgb = ie.encode(instr)
            decoded_instr = ie.decode(rgb=encoded_rgb)
            decoded_inst2 = ie.decode(r=encoded_rgb[0], g=encoded_rgb[1], b=encoded_rgb[2])
            self.assertEqual(instr.mnemonic, decoded_instr.mnemonic)
            self.assertEqual(instr.mnemonic, decoded_inst2.mnemonic)

            ie = InstructionEncoderMnemonicPrefixBnd
            encoded_rgb = ie.encode(instr)
            decoded_instr = ie.decode(rgb=encoded_rgb)
            decoded_inst2 = ie.decode(r=encoded_rgb[0], g=encoded_rgb[1], b=encoded_rgb[2])
            self.assertEqual(decoded_instr, decoded_inst2)
            self.assertEqual(instr.mnemonic, decoded_instr.mnemonic)
            self.assertEqual(instr.prefix, decoded_instr.prefix)
            self.assertEqual(instr.has_bnd, decoded_instr.has_bnd)
