import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add lib to PYTHONPATH
# detect_path = osp.join(this_dir, 'detect', 'lib')
# add_path(detect_path)

# recog_path = osp.join(this_dir, 'recog')
# add_path(recog_path)
detect_path = osp.join(this_dir)

