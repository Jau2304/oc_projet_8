import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../app")))
from app import get_feature_importance

import numpy as np
import pandas as pd
from unittest.mock import MagicMock

class TestGetFeatureImportance(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            "feature_1": [1, 2, 3],
            "feature_2": [4, 5, 6],
            "feature_3": [7, 8, 9]
        })
        self.mock_classifier = MagicMock()
        self.mock_classifier.feature_importances_ = np.array([0.2, 0.1, 0.4])
    
    def test_not_none(self):
        output = get_feature_importance(self.mock_classifier, self.df)
        self.assertIsNotNone(output)

    def test_dataframe(self):
        output = get_feature_importance(self.mock_classifier, self.df)
        self.assertIsInstance(output, pd.DataFrame)

    def test_dataframe_size(self):
        output = get_feature_importance(self.mock_classifier, self.df)
        self.assertEqual(output.shape[0], self.df.shape[1])
        self.assertEqual(output.shape[1], 2)

    def test_dataframe_columns(self):
        output = get_feature_importance(self.mock_classifier, self.df)
        self.assertIn("feature", output.columns)
        self.assertIn("importance", output.columns)

    def test_importance_is_float(self):
        output = get_feature_importance(self.mock_classifier, self.df)
        self.assertTrue(output["importance"].apply(lambda x : isinstance(x, float)).all())

    def test_importance_values(self):
        output = get_feature_importance(self.mock_classifier, self.df)
        importances = [0.4, 0.2, 0.1]
        self.assertListEqual(output["importance"].tolist(), importances)
        features = ["feature_3", "feature_1", "feature_2"]
        self.assertListEqual(output["feature"].tolist(), features)

if __name__ == "__main__":
    unittest.main()
