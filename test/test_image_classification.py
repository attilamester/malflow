import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))
sys.path.append(os.path.join(PROJECT_ROOT, "test"))

from cases.proc_cg_image_classification import TestProcessorCgImageClassification

if __name__ == "__main__":
    _test = [
        TestProcessorCgImageClassification
    ]
    unittest.main()
