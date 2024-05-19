import subprocess
import os

test_files = []

for filename in os.listdir(os.path.dirname(__file__)) :
    if filename.startswith("test_") and filename.endswith(".py") :
        test_files.append(filename)

for test_file in test_files :
    print("\nTests de", test_file)
    subprocess.run(["python", test_file])