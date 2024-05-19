import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../app")))
from app import read_request

import pandas as pd

class TestReadRequest(unittest.TestCase) :

    def setUp(self) :
        self.df = pd.DataFrame({"feature_1" : [1, 2, 3], "feature_2" : [4, 5, 6]})
        self.data = {"selected_index": 1, "shap_max_display" : 10}

    def test_not_none(self) :
        row, max_display = read_request(self.data, self.df)
        self.assertIsNotNone(row)
        self.assertIsNotNone(max_display)

    def test_types(self) :
        row, max_display = read_request(self.data, self.df)
        self.assertIsInstance(row, pd.DataFrame)
        self.assertIsInstance(max_display, int)

    def test_row_shape(self) :
        row, max_display = read_request(self.data, self.df)
        expected_shape = (1, self.df.shape[1])
        self.assertEqual(row.shape, expected_shape)

    def test_row_values(self) :
        row, _ = read_request(self.data, self.df)
        for feature in self.df.columns :
            expected_value = self.df.iloc[self.data["selected_index"]][feature]
            self.assertEqual(row.iloc[0][feature], expected_value)

    def test_max_display(self) :
        row, max_display = read_request(self.data, self.df)
        self.assertEqual(max_display, self.data["shap_max_display"])

if __name__ == "__main__" :
    unittest.main()
