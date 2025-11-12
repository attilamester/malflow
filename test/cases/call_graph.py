import datetime
import os.path
import traceback
import unittest
from typing import Callable
from unittest.mock import Mock, patch

import r2pipe

from cases.data.call_graph_data import CALL_GRAPH_DATA, CallGraphData
from cases.r2_scanner import test_sample
from malflow.core.model.sample import Sample


class TestCallGraph(unittest.TestCase):

    def tearDown(self):
        try:
            etype, value, tb = self._outcome.errors[0][1]
            trace = "".join(traceback.format_exception(etype=etype, value=value, tb=tb, limit=None))
            date = "{date}\n".format(date=str(datetime.datetime.now()))
            name = "\n" + self._testMethodName + "-\n"
            print(name + date + trace)
        except:
            pass

    @patch("malflow.core.data.malware_bazaar.MalwareBazaar.get_sample",
           return_value=Sample(filepath=os.path.abspath(__file__), check_hashes=False))
    def test_call_graph_scanner(self, mock_get_sample):

        def get_mock_r2_cmd(data: CallGraphData) -> Callable:
            def mock_r2_cmd(cmd: str) -> str:
                if cmd == "aaa":
                    return ""
                if cmd == "agCd":
                    return data.agCd
                if cmd == "agRd":
                    return data.agRd
                if cmd == "ies":
                    return data.ies
                if cmd in data.s_pdfj:
                    return data.s_pdfj[cmd]
                return ""

            return mock_r2_cmd

        def get_mock_r2_open(data: CallGraphData) -> Mock:

            mock_r2 = Mock()
            mock_r2.cmd = Mock(side_effect=get_mock_r2_cmd(data))
            mock_r2.quit = Mock(return_value=None)

            return mock_r2

        for data in CALL_GRAPH_DATA:
            with patch.object(r2pipe, "open", return_value=get_mock_r2_open(data)):
                test_sample(self, data.r2_data)
