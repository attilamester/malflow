import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))
sys.path.append(os.path.join(PROJECT_ROOT, "test"))

from cases.r2_scanner import TestR2Scanner
from cases.instruction import TestInstruction

if __name__ == "__main__":
    _test = [
        TestR2Scanner, TestInstruction
    ]
    unittest.main()
