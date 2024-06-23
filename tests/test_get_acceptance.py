import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../app")))

from app import get_acceptance

class TestGetAcceptance(unittest.TestCase) :

    def setUp(self) :
        self.acceptance_file_path = "test_acceptance.txt"
        self.create_test_acceptance_txt()

    def tearDown(self) :
        if os.path.exists(self.acceptance_file_path) :
            os.remove(self.acceptance_file_path)

    def create_test_acceptance_txt(self) :
        with open(self.acceptance_file_path, "w") as file :
            file.write("test_model.pkl : 0.75\n")
            file.write("another_model.pkl : 0.6\n")

    def test_value(self) :
        acceptance = get_acceptance(self.acceptance_file_path, "test_model.pkl")
        self.assertEqual(acceptance, 0.75)

if __name__ == "__main__" :
    unittest.main()

