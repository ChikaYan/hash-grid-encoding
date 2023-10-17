import encoding
import numpy as np
import torch
import json
from pathlib import Path
from model.model import NRC
from typing import Optional
import imageio
from model.utils import Timer
import pandas as pd
import yaml
import wandb
from tqdm import tqdm

INVALID_INDEX = 0x007FFFFF
IMAGE_SIZE = (512, 512)

DEVICE = 'cuda'

def get_float3(df, name):
    return df[[f'{name}_x', f'{name}_y', f'{name}_z']].to_numpy()

def set_float3(df, idx, name, value):
    df.at[idx, f'{name}_x'] = value[0]
    df.at[idx, f'{name}_y'] = value[1]
    df.at[idx, f'{name}_z'] = value[2]

def propagate_radiance_values(
        df_pre_train_infer:pd.DataFrame, 
        df_train_vertex:pd.DataFrame, 
        inferred_radiance:torch.Tensor, 
        radiance_scale:float=1.
    ):
    N = len(df_pre_train_infer)

    for i in range(N):
        # if 'prev_vertex_data_index' not in df_pre_train_infer[i]:
        if pd.isna(df_pre_train_infer.at[i, 'prev_vertex_data_index']):
            break
        prev_vertex_data_index = df_pre_train_infer.at[i, 'prev_vertex_data_index']

        if prev_vertex_data_index == INVALID_INDEX:
            continue

        contribution = torch.zeros(3)

        if df_pre_train_infer.at[i, 'has_query']:
            offset = IMAGE_SIZE[0] * IMAGE_SIZE[1]
            contribution = torch.clamp_min(inferred_radiance[offset + i], 0)

            if radiance_scale > 0.:
                contribution = contribution / radiance_scale


            inferred_query = df_pre_train_infer.iloc[offset + i]

            contribution = contribution * \
                torch.tensor(get_float3(inferred_query, 'diffuse_reflectance')) + torch.tensor(get_float3(inferred_query, 'specular_reflectance'))
            
        last_train_data_index = int(df_pre_train_infer.at[i, 'prev_vertex_data_index'])
        while (last_train_data_index != INVALID_INDEX):
            vertex_info = df_train_vertex.iloc[last_train_data_index]
            query = df_pre_train_infer.iloc[last_train_data_index]
            target_value = torch.tensor(get_float3(vertex_info, 'target_train_buffer'))
            indirect_cont = torch.tensor(get_float3(vertex_info, 'local_throughput')) * contribution
            contribution = target_value + indirect_cont

            # here we store target value as the actual RGB rendering, instead of the coefficients before factorization
            contribution = contribution / (torch.tensor(get_float3(query, 'diffuse_reflectance') + get_float3(query, 'specular_reflectance')))
            contribution = torch.nan_to_num(contribution, 0., 0., 0.)
            set_float3(df_train_vertex, last_train_data_index, 'target_train_buffer', contribution.numpy())
            # df_train_vertex.at[last_train_data_index, 'target_train_buffer'] = target_value.tolist()


            last_train_data_index = int(vertex_info['prev_vertex_data_index'])

    

    
def prepare_batch(df_query:pd.DataFrame, df_train_vertex:Optional[pd.DataFrame]=None, cpp_target:bool=False):
    N = len(df_query)

    p = torch.tensor(get_float3(df_query, 'position')).contiguous()
    wi = torch.tensor([df_query['vOut_phi'].to_list(), df_query['vOut_theta'].to_list()]).T.contiguous()
    n = torch.tensor([df_query['normal_phi'].to_list(), df_query['normal_theta'].to_list()]).T.contiguous()
    alpha = torch.tensor(get_float3(df_query, 'diffuse_reflectance')).contiguous()
    beta = torch.tensor(get_float3(df_query, 'specular_reflectance')).contiguous()
    r = torch.tensor(df_query['roughness'].to_list()).contiguous()

    if df_train_vertex is not None:
        if cpp_target:
            targets = torch.tensor(get_float3(df_query, 'target')[:N]).contiguous()
        else:
            targets = torch.tensor(get_float3(df_train_vertex, 'target_train_buffer')[:N]).contiguous()

    else:
        targets = None

    inputs = [p, wi, n, alpha, beta, r]

    return inputs, targets







def main():
    config_path = './config/cornell_box_dy/base.yaml'

    with open(config_path, 'rt') as f:
        config = yaml.safe_load(f.read())

    wandb.init(
        project=config['log']['project'],
        name=config['log']['name'],
        config=config['exp'],
    )

    log_dir = Path('exp') / config['log']['name']
    log_dir.mkdir(exist_ok=True, parents=True)
    
    torch.manual_seed(0)


    exp_path = Path(config['exp']['query_path'])

    radiance_scale = config['exp']['radiance_scale']

    mlp = NRC(ma_alpha=config['exp']['mlp_ma_alpha'], radiance_scale=radiance_scale)


    optimizer = torch.optim.Adam(params=mlp.parameters(), lr=1e-3)

    h5_lists = sorted(list(exp_path.glob('*.h5')))
    N_FRAME = len(h5_lists)

    for frame_id in tqdm(range(N_FRAME)):
        store = pd.HDFStore(str(h5_lists[frame_id]))

        df_pre_train_infer = store['df_pre_train_infer']
        df_train_query = store['df_train_query']
        df_train_vertex = store['df_train_vertex']
        df_rendering_infer = store['df_rendering_infer']
        store.close()


        # query NRC to prepare self-training
        # with Timer('prep pre train infer batch'):
        infer_inputs, _ = prepare_batch(df_pre_train_infer)
        infer_outs = mlp.batch_infer(infer_inputs).to('cpu')


        # with Timer('propagate radiance values'):
        propagate_radiance_values(df_pre_train_infer, df_train_vertex, infer_outs)

        # with Timer('prep train batch'):
        inputs, targets = prepare_batch(df_train_query, df_train_vertex, cpp_target=config['exp']['cpp_target'])

        loss = mlp.train(inputs, targets, optimizer, apply_factorization=False)


        wandb.log({'train/loss': loss})

        # visualize predictions
        # with Timer('prep visualize batch'):
        visual_inputs, _ = prepare_batch(df_rendering_infer)

        # with Timer('infer visualize batch'):
        visual_outs = mlp.batch_infer(visual_inputs) / radiance_scale

        image = torch.clamp(visual_outs.reshape([*IMAGE_SIZE, 3]), 0., 1.).cpu().numpy()

        imageio.imwrite(str(log_dir / f"render_{frame_id:05d}.png"), (image * 255).astype(np.uint8))






        

        




if __name__ == '__main__':
    main()