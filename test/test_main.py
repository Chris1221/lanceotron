import unittest
from lanceotron import find_and_score_peaks, dotdict

# Simple run through on toy data
class TestAll(unittest.TestCase):
    def test_example(self):
        find_and_score_peaks(dotdict({
            "file": "test/chr22.bw",
            "folder": "./",
            "threshold": 4,
            "window": 400,
            "skipheader": False
        }))