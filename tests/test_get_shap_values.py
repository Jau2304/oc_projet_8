import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../app")))
from app import get_shap_values

import numpy as np
import pandas as pd

class TestGetShapValues(unittest.TestCase) :

    def setUp(self) :
        class MockExplainer :
            def shap_values(self, row) :
                return np.array([[0.1, 0.2, 0.3, 0.4]])
        self.explainer = MockExplainer()
        self.row = pd.DataFrame({"feature_1": [1], "feature_2": [2], "feature_3": [3], "feature_4": [4]})
        self.max_display = 2

    def test_keys(self) :
        output = {}
        get_shap_values(self.row, self.explainer, self.max_display, output)
        self.assertTrue("top_features" in output)
        self.assertTrue("top_features_values" in output)
        self.assertTrue("top_shap_values" in output)

    def test_keys_not_none(self) :
        output = {}
        get_shap_values(self.row, self.explainer, self.max_display, output)
        self.assertIsNotNone(output.get("top_features"))
        self.assertIsNotNone(output.get("top_features_values"))
        self.assertIsNotNone(output.get("top_shap_values"))

    def test_keys_types(self):
        output = {}
        get_shap_values(self.row, self.explainer, self.max_display, output)
        self.assertIsInstance(output["top_features"], list)
        self.assertIsInstance(output["top_features_values"], list)
        self.assertIsInstance(output["top_shap_values"], list)

    def test_values(self):
        output = {}
        get_shap_values(self.row, self.explainer, self.max_display, output)
        top_features = ["feature_4", "feature_3"]
        top_features_values = [4, 3]
        top_shap_values = [0.4, 0.3]
        self.assertEqual(output["top_features"], top_features)
        self.assertEqual(output["top_features_values"], top_features_values)
        self.assertEqual(output["top_shap_values"], top_shap_values)

if __name__ == "__main__":
    unittest.main()
