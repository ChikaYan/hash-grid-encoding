import encoding
import numpy as np
import torch
import json
from pathlib import Path
from model.model import NRC
from typing import Optional
import imageio
from model.utils import Timer, simple_tone_map_s, srgb_calc_luminance, srgb_gamma_s
import pandas as pd
import yaml
import wandb
from tqdm import tqdm
import argparse
from torch.profiler import profile, record_function, ProfilerActivity
import cv2


INVALID_INDEX = 0x007FFFFF
IMAGE_SIZE = (512, 512)

DEVICE = 'cuda'



def unify_float3(df:pd.DataFrame):
    # unify _x, _y, _z entries in df into a list entry
    for col in df.columns:
        if not col.endswith('_x'):
            continue

        entry_name = col[:-2]
        if not (f'{entry_name}_y' in df.columns and  f'{entry_name}_z' in df.columns):
            continue

        df[entry_name] = df[[f'{entry_name}_x', f'{entry_name}_y', f'{entry_name}_z']].to_numpy().tolist()

        df.drop([f'{entry_name}_x', f'{entry_name}_y', f'{entry_name}_z'], axis=1, inplace=True)

def propagate_radiance_values(
        df_pre_train_infer:pd.DataFrame, 
        df_train_vertex:pd.DataFrame, 
        df_train_query:pd.DataFrame,
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

        if df_pre_train_infer.at[i, 'has_query'] == 1.:
            offset = IMAGE_SIZE[0] * IMAGE_SIZE[1]
            contribution = torch.clamp_min(inferred_radiance[offset + i], 0)

            if radiance_scale > 0.:
                contribution = contribution / radiance_scale


            inferred_query = df_pre_train_infer.iloc[offset + i]

            contribution = contribution * \
                (torch.tensor(inferred_query['diffuse_reflectance']) + torch.tensor(inferred_query['specular_reflectance']))
            
        if i == 143:
            print()

        last_train_data_index = int(df_pre_train_infer.at[i, 'prev_vertex_data_index'])
        while (last_train_data_index != INVALID_INDEX):
            vertex_info = df_train_vertex.iloc[last_train_data_index]
            query = df_train_query.iloc[last_train_data_index]
            target_value = torch.tensor(vertex_info['target_train_buffer'])
            indirect_cont = torch.tensor(vertex_info['local_throughput']) * contribution
            contribution = target_value + indirect_cont

            new_target = contribution / (torch.tensor(query['diffuse_reflectance']) + torch.tensor(query['specular_reflectance']))
            new_target = torch.nan_to_num(new_target, 0., 0., 0.)


            df_train_vertex.at[last_train_data_index, 'target_train_buffer'] = new_target.numpy()


            last_train_data_index = int(vertex_info['prev_vertex_data_index'])

    

    
def prepare_batch(df_query:pd.DataFrame, radiance_scale:float=1., spp_id:int=None):
    N = len(df_query)

    p = torch.tensor(np.stack(df_query['position'])).contiguous()
    wi = torch.tensor([df_query['vOut_phi'].to_list(), df_query['vOut_theta'].to_list()]).T.contiguous()
    n = torch.tensor([df_query['normal_phi'].to_list(), df_query['normal_theta'].to_list()]).T.contiguous()
    alpha = torch.tensor(np.stack(df_query['diffuse_reflectance'])).contiguous()
    beta = torch.tensor(np.stack(df_query['specular_reflectance'])).contiguous()
    r = torch.tensor(df_query['roughness'].to_list()).contiguous()

    if spp_id is None:
        targets = []

        for k in df_query.keys():
            if k.startswith('target_'):
                target_1_spp = torch.tensor(np.stack(df_query[k])[:N]).contiguous() * radiance_scale
                targets.append(target_1_spp)

        targets = torch.stack(targets).mean(axis=0)
    else:
        k = f'target_{spp_id:03d}'
        targets = torch.tensor(np.stack(df_query[k])[:N]).contiguous() * radiance_scale


    inputs = [p, wi, n, alpha, beta, r]

    return inputs, targets







def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('-c', '--config', type=str, default='./config/debug.yaml')
    parser.add_argument('-c', '--config', type=str, default='./config/check_target.yaml')
    parser.add_argument('-s', '--scene_path', type=str, default='scene/living_room_diffuse')
    parser.add_argument('-n', '--n_frames', type=int, default=None)
    args = parser.parse_args()

    config_path = args.config


    with open(config_path, 'rt') as f:
        config = yaml.safe_load(f.read())

    exp_conf = config['exp']

    if 'name' in config['log']:
        RUN_NAME = config['log']['name']
    else:
        RUN_NAME = config_path.replace('config/', '').replace('.yaml', '')

    SCENE_NAME = Path(args.scene_path).name
    CHECK_TARGET = exp_conf.get('check_target', False)

    wandb.init(
        project=config['log']['project'],
        name=RUN_NAME,
        config=exp_conf,
        group=SCENE_NAME,
    )

    log_dir = Path('exp') / SCENE_NAME / RUN_NAME
    log_dir.mkdir(exist_ok=True, parents=True)
    (log_dir / 'render').mkdir(exist_ok=True, parents=True)
    
    torch.manual_seed(0)


    exp_path = Path(args.scene_path)

    radiance_scale = exp_conf['model_params']['radiance_scale']

    mlp = NRC(**exp_conf['model_params'])

    optimizer = torch.optim.Adam(params=mlp.parameters(), lr=float(exp_conf['lr']))

    h5_lists = sorted(list(exp_path.glob('*.h5')))
    N_FRAME = len(h5_lists)

    if args.n_frames is not None:
        N_FRAME = min(N_FRAME, args.n_frames)

    imgs = []

    for frame_id in tqdm(range(N_FRAME)):
        store = pd.HDFStore(str(h5_lists[frame_id]))

        # df_pre_train_infer = store['df_pre_train_infer']
        df_train_query = store['df_train_query']
        # df_train_vertex = store['df_train_vertex']
        # df_rendering_infer = store['df_rendering_infer']
        store.close()

        for df in [df_train_query]:
            unify_float3(df)

        # with profile(activities=[]) as prof:

        if config['exp'].get('per_spp', False):
            n_spp = len([k for k in df_train_query.keys() if k.startswith('target_')])

            for spp_id in range(n_spp):
                with record_function('prep train batch'):
                    inputs, targets = prepare_batch(df_train_query, radiance_scale=radiance_scale, spp_id=spp_id)

                with record_function('Train'):
                    loss = mlp.train(inputs, targets, optimizer,
                                    **exp_conf['train_params'],
                                    wandb=wandb)
        else:
            with record_function('prep train batch'):
                # inputs, targets = prepare_batch(df_train_query, radiance_scale=radiance_scale)
                inputs, targets = prepare_batch(df_train_query, radiance_scale=radiance_scale, spp_id=0)

            if CHECK_TARGET:
                loss = {}
            else:
                with record_function('Train'):
                    loss = mlp.train(inputs, targets, optimizer,
                                    **exp_conf['train_params'],
                                    wandb=wandb)


        wandb.log({'train/loss': loss})

        if CHECK_TARGET:
            alpha, beta = inputs[3], inputs[4]
            visual_outs = targets / radiance_scale * (alpha + beta)
        else:
            # visualize predictions
            with record_function('prep visualize batch'):
                # visual_inputs, _ = prepare_batch(df_rendering_infer)
                visual_inputs = inputs

            with record_function('infer visualize batch'):
                visual_outs = mlp.batch_infer(visual_inputs) / radiance_scale



        image = torch.clamp(visual_outs, 0).cpu().numpy()

        if Path(args.scene_path).name.startswith('living_room'):
            brightness = 100
        else:
            brightness = 10

        # apply tonemapping
        lum = srgb_calc_luminance(image)
        lum_t = simple_tone_map_s(brightness * lum)
        s = np.where(lum > 0., lum_t / np.clip(lum, 1e-5, None), 0.)
        image *= s[:,None]
        image = srgb_gamma_s(image)

        image = image.reshape([*IMAGE_SIZE, 3])
        image = np.clip((image * 255), 0, 255).astype(np.uint8)




        # image = torch.clamp(visual_outs.reshape([*IMAGE_SIZE, 3]), 0., 1.).cpu().numpy()
        # image = np.clip((image * 10 * 255), 0, 255).astype(np.uint8)

        imageio.imwrite(str(log_dir / 'render'/ f"render_{frame_id:05d}.png"), image)

        imgs.append(image)

        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
        # print()

    writer = imageio.get_writer(str(log_dir / 'render.mp4'), fps=30)
    for im in imgs:
        writer.append_data(im)
    writer.close()



    






        

        




if __name__ == '__main__':
    main()