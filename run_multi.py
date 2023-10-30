import os
from pathlib import Path

# config_base_path = "config/cornell_box_dy"

# conf_list = [
#     './config/cornell_box_dy/1e-2_1e-3.yaml'
# ]

conf_list = list(Path('./config').glob('hash_*')) + list(Path('./config').glob('tri_*')) +  list(Path('./config').glob('*_cpp_target*'))
# conf_list = list(Path('./config').glob('*_cpp_target*'))

conf_list = [Path('./config') / 'hash_5e-3.yaml', Path('./config') / 'hash_5e-3_cpp_target.yaml']
conf_list = [Path('./config') / 'hash_1e-2.yaml', Path('./config') / 'hash_1e-2_cpp_target.yaml']
conf_list = [Path('./config') / 'hash_1e-2_ma.yaml']


conf_list = []
conf_list += list(Path('./config/lr_tune').glob('*'))
conf_list += list(Path('./config/tv').glob('*'))
# conf_list = ['./config/tv/1e2.yaml']
# conf_list = ['./config/hash_1e-2_ma.yaml']

conf_list = Path('./config/lipshitz').glob('*')

conf_list = [
    Path('./config') / 'lipshitz'/ '1e-2_lr_5e-3.yaml',
    Path('./config') / 'lr_tune'/ 'hash_1e-2.yaml',
    Path('./config') / 'tv'/ '1e2.yaml',
    ]



print(conf_list)

scenes = [
    # 'cornell_box_dy',
    # 'living_room_dy_light',
    'living_room_dy_tex',
]



for c in conf_list:
    for s in scenes:
        os.system(f"python nrc.py -c {str(c)} -s scene/{s}")