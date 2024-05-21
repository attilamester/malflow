import unittest

from cases.data.instruction_data import InstructionData, INSTRUCTIONS, INSTRUCTION_PARAMETER_TOKENS
from core.model.instruction import Instruction, InstructionParameter


class TestInstruction(unittest.TestCase):

    def __test_instruction(self, instruction_data: InstructionData):
        i = Instruction(instruction_data.disasm, b"0", [])

        self.assertEqual(instruction_data.disasm, i.disasm)
        self.assertEqual(instruction_data.mnemonic, i.mnemonic)
        self.assertEqual(b"0", i.opcode)
        self.assertEqual(instruction_data.has_bnd, i.has_bnd)
        self.assertEqual(instruction_data.prefix, i.prefix)
        self.assertEqual(instruction_data.parameters, i.parameters)

    def __test_instruction_fmt(self, instruction_data: InstructionData):
        i = Instruction(instruction_data.disasm, b"0", [])

        fmt = i.get_fmt()
        i_parsed = Instruction.parse_fmt(fmt)
        self.assertEqual(i.mnemonic, i_parsed.mnemonic)
        self.assertEqual(i.prefix, i_parsed.prefix)
        self.assertEqual(i.has_bnd, i_parsed.has_bnd)
        self.assertEqual(i.parameters, i_parsed.parameters)

    def test_instructions(self):
        for item in INSTRUCTIONS:
            self.__test_instruction(item)

    def test_instruction_parameter_tokens(self):
        for input, expected in INSTRUCTION_PARAMETER_TOKENS:
            self.assertEqual(expected, InstructionParameter.split_into_parameter_tokens(input))

    def test_instruction_fmt_parser(self):
        for i_data in INSTRUCTIONS:
            self.__test_instruction_fmt(i_data)
