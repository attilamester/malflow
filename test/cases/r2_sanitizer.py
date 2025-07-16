import unittest

from cases.data.r2_sanitizer_data import R2_ERROR_MESSAGES, DOT_LINES
from malflow.core.model.radare2_definitions.sanitizer import sanitize_r2_dot_line, sanitize_r2_bugs


class TestR2Sanitizer(unittest.TestCase):

    def test_sanitize_r2_bugs(self):
        for input, expected in R2_ERROR_MESSAGES:
            self.assertEqual(expected, sanitize_r2_bugs(input))

    def test_sanitize_r2_dot_line(self):
        for input, expected in DOT_LINES:
            self.assertEqual(expected, sanitize_r2_dot_line(input))
