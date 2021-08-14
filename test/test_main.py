import unittest
import lanceotron.utils 
import pkg_resources
import os

import numpy as np

from lanceotron import find_and_score_peaks
from tensorflow import keras

# Simple run through on toy data
class TestAll(unittest.TestCase):
    """Full run through on toy data
    """
    def test_example(self):
        find_and_score_peaks(**{
            "file": "test/chr22.bw",
            "folder": "./",
            "threshold": 4,
            "window": 400,
            "skipheader": False
        })

class TestResources(unittest.TestCase):
    """Ensure that all of the resources are available.
    """
    def test_model(self):
        assert os.path.isfile(pkg_resources.resource_filename("lanceotron.static", "wide_and_deep_fully_trained_v5_03.h5"))
    def test_widescaler(self):
        assert os.path.isfile(pkg_resources.resource_filename('lanceotron.static', 'standard_scaler_wide_v5_03.p'))
    def test_deepscaler(self):
        assert os.path.isfile(pkg_resources.resource_filename('lanceotron.static', 'standard_scaler_deep_v5_03.p'))


class TestModel(unittest.TestCase):
    """Ensure that the model is correctly formed
    """
    def setUp(self):
        self.model = lanceotron.utils.build_model()
    def test_nlayer(self):
        assert len(self.model.layers) == 37, "Did the model change? It has the wrong number of layers."
    def test_lastlayer(self):
        assert isinstance(self.model.layers[-1], keras.layers.Dense), "The last layer is not a dense prediction layer. Did the model load correctly?"
    def test_hasweights(self):
        np.testing.assert_almost_equal(self.model.layers[-1].get_weights()[1][1], -0.015733806, err_msg= "The model does not have the correct weights. Check the weight file.")
