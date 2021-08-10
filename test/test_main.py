import unittest
import lanceotron

import lanceotron.find_and_score_peaks
from lanceotron.find_and_score_peaks import find_and_score_peaks

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class TestAll(unittest.TestCase):
    def test_example(self):
        find_and_score_peaks(dotdict({
            "file": "test/chr22.bw",
            "folder": "./",
            "threshold": 4,
            "window": 400,
            "skipheader": False
        }))