import os
from pathlib import Path

# config_base_path = "config/cornell_box_dy"

conf_list = [
    './config/cornell_box_dy/1e-2_1e-3.yaml'
]

for c in conf_list:
    os.system(f"python nrc.py -c {c}")