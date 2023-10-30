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
import argparse
from torch.profiler import profile, record_function, ProfilerActivity


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

    

    
def prepare_batch(df_query:pd.DataFrame, df_train_vertex:Optional[pd.DataFrame]=None, cpp_target:bool=False, radiance_scale:float=1.):
    N = len(df_query)

    p = torch.tensor(np.stack(df_query['position'])).contiguous()
    wi = torch.tensor([df_query['vOut_phi'].to_list(), df_query['vOut_theta'].to_list()]).T.contiguous()
    n = torch.tensor([df_query['normal_phi'].to_list(), df_query['normal_theta'].to_list()]).T.contiguous()
    alpha = torch.tensor(np.stack(df_query['diffuse_reflectance'])).contiguous()
    beta = torch.tensor(np.stack(df_query['specular_reflectance'])).contiguous()
    r = torch.tensor(df_query['roughness'].to_list()).contiguous()

    if df_train_vertex is not None:
        t1 = torch.tensor(np.stack(df_query['target'])[:N]).contiguous() / 10. * radiance_scale
        t2 = torch.tensor(np.stack(df_train_vertex['target_train_buffer'])[:N]).contiguous() * radiance_scale
        if cpp_target:
            targets = torch.tensor(np.stack(df_query['target'])[:N]).contiguous() / 10. * radiance_scale
        else:
            targets = torch.tensor(np.stack(df_train_vertex['target_train_buffer'])[:N]).contiguous() * radiance_scale

    else:
        targets = None

    inputs = [p, wi, n, alpha, beta, r]

    return inputs, targets







def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='./config/lipshitz/1e-2_lr_1e-3.yaml')
    parser.add_argument('-s', '--scene_path', type=str, default='scene/cornell_box_dy')
    args = parser.parse_args()

    config_path = args.config


    with open(config_path, 'rt') as f:
        config = yaml.safe_load(f.read())

    exp_conf = config['exp']

    if 'name' in config['log']:
        run_name = config['log']['name']
    else:
        run_name = config_path.replace('config/', '').replace('.yaml', '')

    scene_name = Path(args.scene_path).name

    wandb.init(
        project=config['log']['project'],
        name=run_name,
        config=exp_conf,
        group=scene_name,
    )

    log_dir = Path('exp') / scene_name / run_name
    log_dir.mkdir(exist_ok=True, parents=True)
    (log_dir / 'render').mkdir(exist_ok=True, parents=True)
    
    torch.manual_seed(0)


    exp_path = Path(args.scene_path)

    radiance_scale = exp_conf['model_params']['radiance_scale']

    mlp = NRC(**exp_conf['model_params'])

    optimizer = torch.optim.Adam(params=mlp.parameters(), lr=float(exp_conf['lr']))

    h5_lists = sorted(list(exp_path.glob('*.h5')))
    N_FRAME = len(h5_lists)

    imgs = []

    for frame_id in tqdm(range(N_FRAME)):
        store = pd.HDFStore(str(h5_lists[frame_id]))

        df_pre_train_infer = store['df_pre_train_infer']
        df_train_query = store['df_train_query']
        df_train_vertex = store['df_train_vertex']
        df_rendering_infer = store['df_rendering_infer']
        store.close()

        for df in [df_pre_train_infer, df_train_query, df_train_vertex, df_rendering_infer]:
            unify_float3(df)

        # with profile(activities=[]) as prof:
            
        # query NRC to prepare self-training
        with record_function('prep pre train infer batch'):
            infer_inputs, _ = prepare_batch(df_pre_train_infer)
            infer_outs = mlp.batch_infer(infer_inputs).to('cpu')


        with record_function('propagate radiance values'):
            propagate_radiance_values(df_pre_train_infer, df_train_vertex, df_train_query, infer_outs, radiance_scale=radiance_scale)

        with record_function('prep train batch'):
            inputs, targets = prepare_batch(df_train_query, df_train_vertex, cpp_target=exp_conf['cpp_target'], radiance_scale=radiance_scale)

        with record_function('Train'):
            loss = mlp.train(inputs, targets, optimizer,
                            **exp_conf['train_params'],
                            wandb=wandb)


        wandb.log({'train/loss': loss})

        # visualize predictions
        with record_function('prep visualize batch'):
            visual_inputs, _ = prepare_batch(df_rendering_infer)

        with record_function('infer visualize batch'):
            visual_outs = mlp.batch_infer(visual_inputs) / radiance_scale



        image = torch.clamp(visual_outs.reshape([*IMAGE_SIZE, 3]), 0., 1.).cpu().numpy()
        image = np.clip((image * 10 * 255), 0, 255).astype(np.uint8)

        imageio.imwrite(str(log_dir / 'render'/ f"render_{frame_id:05d}.png"), image)

        imgs.append(image)

        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
        # print()

    writer = imageio.get_writer(str(log_dir / 'render.mp4'), fps=10)
    for im in imgs:
        writer.append_data(im)
    writer.close()



    






        

        




if __name__ == '__main__':
    main()