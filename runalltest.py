#!/bin/bash
import os
import glob

testfiles = glob.glob("./test/*.py")
testfiles.remove("./test/__init__.py")
# this module aims to test all our files as the call python3 -m unittest doesnt seem to work due to import issues (cant seem to find the files)
for file in testfiles:
    filename = file[2:]
    print(filename)
    os.system(f"python3 -m unittest {filename}")
