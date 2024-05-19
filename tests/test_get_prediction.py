import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../app")))
from app import get_prediction

import numpy as np

class TestGetPrediction(unittest.TestCase) :

    def setUp(self) :
        class PlaceHolderClassifier :
            def predict_proba(self, row) :
                return np.array([[0.3, 0.7]])
        self.classifier = PlaceHolderClassifier()
        self.row = np.array([[1, 2, 3]])

    def test_not_none(self) :
        prediction = get_prediction(self.row, self.classifier, 0.5)
        self.assertIsNotNone(prediction)

    def test_type(self) :
        prediction = get_prediction(self.row, self.classifier, 0.5)
        self.assertIsInstance(prediction, int)

    def test_value(self):
        prediction = get_prediction(self.row, self.classifier, 0.5)
        self.assertEqual(prediction, 1)
        prediction = get_prediction(self.row, self.classifier, 0.8)
        self.assertEqual(prediction, 0)

if __name__ == "__main__" :
    unittest.main()
