import os
import sys

def find_and_add_project_root(marker="main.py"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = current_dir
    while True:
        files = os.listdir(root_dir)
        parent_dir = os.path.dirname(root_dir)
        if marker in files:
            break
        elif root_dir == parent_dir:
            raise FileNotFoundError("Project root marker not found.")
        else:
            root_dir = parent_dir
    if root_dir not in sys.path:
        sys.path.append(root_dir)
    
    return root_dir
