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
        output = {}
        get_prediction(self.row, self.classifier, 0.5, output)
        self.assertIsNotNone(output.get("pred_proba"))
        self.assertIsNotNone(output.get("acceptance"))
        self.assertIsNotNone(output.get("pred_binary"))

    def test_type(self) :
        output = {}
        get_prediction(self.row, self.classifier, 0.5, output)
        self.assertIsInstance(output.get("pred_proba"), float)
        self.assertIsInstance(output.get("acceptance"), (int, float))
        self.assertIsInstance(output.get("pred_binary"), int)

    def test_values_1(self):
        output = {}
        get_prediction(self.row, self.classifier, 0.5, output)
        self.assertEqual(output["pred_proba"], 0.7)
        self.assertEqual(output["acceptance"], 0.5)
        self.assertEqual(output["pred_binary"], 1)

    def test_values_2(self):
        output = {}
        get_prediction(self.row, self.classifier, 0.8, output)
        self.assertEqual(output["pred_proba"], 0.7)
        self.assertEqual(output["acceptance"], 0.8)
        self.assertEqual(output["pred_binary"], 0)

if __name__ == "__main__" :
    unittest.main()
