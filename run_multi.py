import os
from pathlib import Path

# config_base_path = "config/cornell_box_dy"

# conf_list = [
#     './config/cornell_box_dy/1e-2_1e-3.yaml'
# ]

# conf_list = list(Path('./config').glob('hash_*')) + list(Path('./config').glob('tri_*')) +  list(Path('./config').glob('*_cpp_target*'))
# # conf_list = list(Path('./config').glob('*_cpp_target*'))

# conf_list = [Path('./config') / 'hash_5e-3.yaml', Path('./config') / 'hash_5e-3_cpp_target.yaml']
# conf_list = [Path('./config') / 'hash_1e-2.yaml', Path('./config') / 'hash_1e-2_cpp_target.yaml']
# conf_list = [Path('./config') / 'hash_1e-2_ma.yaml']


# conf_list = []
# conf_list += list(Path('./config/lr_tune').glob('*'))
# conf_list += list(Path('./config/tv').glob('*'))
# # conf_list = ['./config/tv/1e2.yaml']
# # conf_list = ['./config/hash_1e-2_ma.yaml']

# conf_list = Path('./config/lipshitz').glob('*')

conf_list = [
    Path('./config') / 'lipshitz'/ '1e-2_lr_5e-3.yaml',
    Path('./config') / 'lr_tune'/ 'hash_1e-2.yaml',
    Path('./config') / 'tv'/ '1e2.yaml',
    ]


conf_list = list(Path('./config/lr_tune').glob('tri_1e-3_large_mlp_low_freq_per_spp_ma*'))
conf_list = list(Path('./config/lr_tune').glob('tri_1e-4_large_mlp_low_freq_per_spp.yaml'))
conf_list = list(Path('./config/lr_grid').glob('*'))
conf_list = list(Path('./config/grid_tv').glob('*'))
conf_list = list(Path('./config/grid_lip').glob('*'))

conf_list = [
    # './config/check_target.yaml',

    # './config/lr_tune/tri_2e-4_large_mlp_low_freq_ma_099.yaml',
    # './config/lr_tune/tri_2e-4_large_mlp_low_freq_per_spp.yaml',
    # './config/lr_tune/tri_2e-4_large_mlp_per_spp.yaml',
    './config/lr_tune/tri_1e-2_large_mlp_low_freq_maa',

    # './config/lr_tune/tri_2e-4_large_mlp_low_freq_per_spp_ma_099.yaml',
    # './config/lr_grid/1e-3.yaml',
    # './config/lr_grid/1e-3_ma_099.yaml',
    # './config/grid_tv/1e0.yaml',
    # './config/grid_tv/1e0_ma_099.yaml',
    # './config/grid_tv/1e1_ma_099.yaml',
    # './config/grid_lip/1e-3_ma_099.yaml',
    # './config/grid_lip/1e-3_ma_099.yaml',
    # './config/lip_tri/lr_2e-4_lip_1e-4.yaml',
]


# conf_list = list(Path('./config/').glob('check_target*'))

print(conf_list)

scenes = [
    # 'cornell_box_dy',
    # 'living_room_dy_light',
    # 'living_room_dy_tex',

    'cornell_box_dy2_spp_16',
    'living_room_diffuse',
]



for c in conf_list:
    for s in scenes:
        os.system(f"python nrc.py -c {str(c)} -s scene/{s} -n 300")