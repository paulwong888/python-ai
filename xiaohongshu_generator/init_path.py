import os
import sys

def init():
    # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    # print("a()")
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # print(dir_path)
    parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
    # print(parent_dir_path)
    sys.path.insert(0, parent_dir_path)

