import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../app")))
from app import load_data

import pandas as pd
import shap
from lightgbm import LGBMClassifier
import pickle

class TestLoadData(unittest.TestCase) :

    def setUp(self) :
        self.csv_file_name = "test_data.csv"
        self.classifier_file_name = "test_classifier.pkl"
        self.create_test_data_csv()
        self.create_test_classifier_pkl()

    def tearDown(self) :
        if os.path.exists(self.csv_file_name) :
            os.remove(self.csv_file_name)
        if os.path.exists(self.classifier_file_name) :
            os.remove(self.classifier_file_name)

    def create_test_classifier_pkl(self) :
        X_train = pd.DataFrame({"feature_1": [1, 2, 3], "feature_2": [4, 5, 6]})
        y_train = pd.Series([0, 1, 0])
        classifier = LGBMClassifier(verbose = -1)
        classifier.fit(X_train, y_train)
        with open(self.classifier_file_name, "wb") as file :
            pickle.dump(classifier, file)

    def create_test_data_csv(self) :
        df = pd.DataFrame({"feature_1" : [2, 4, 1], "feature_2" : [6, 3, 5]})
        df.to_csv(self.csv_file_name, index = False)

    def test_not_none(self) :
        df, classifier, explainer = load_data(self.csv_file_name, self.classifier_file_name)
        self.assertIsNotNone(df)
        self.assertIsNotNone(classifier)
        self.assertIsNotNone(explainer)

    def test_types(self) :
        df, classifier, explainer = load_data(self.csv_file_name, self.classifier_file_name)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(isinstance(classifier, LGBMClassifier))
        self.assertTrue(isinstance(explainer, shap.Explainer))

if __name__ == "__main__" :
    unittest.main()