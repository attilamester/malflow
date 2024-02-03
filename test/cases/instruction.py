import unittest

from cases.instruction_data import InstructionData, INSTRUCTIONS, INSTRUCTION_PARAMETER_TOKENS
from core.model.instruction import Instruction, InstructionParameter


class TestInstruction(unittest.TestCase):

    def __test_instruction(self, instruction_data: InstructionData):
        i = Instruction(instruction_data.disasm, b"0")

        self.assertEqual(instruction_data.disasm, i.disasm)
        self.assertEqual(instruction_data.mnemonic, i.mnemonic)
        self.assertEqual(b"0", i.opcode)
        self.assertEqual(instruction_data.has_bnd, i.has_bnd)
        self.assertEqual(instruction_data.prefix, i.prefix)
        self.assertEqual(instruction_data.parameters, i.parameters)

    def test_instructions(self):
        for item in INSTRUCTIONS:
            self.__test_instruction(item)

    def test_instruction_parameter_tokens(self):
        for item in INSTRUCTION_PARAMETER_TOKENS:
            self.assertEqual(item[1], InstructionParameter.split_into_parameter_tokens(item[0]))
