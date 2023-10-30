import os
import torch
from torch import nn
from model import encoder
from model.utils import to_sphe_coords
import numpy as np
from copy import deepcopy
import encoding
from model.utils import apply_weight_init_fn, leaky_relu_init



class LipshitzMLP(torch.nn.Module):
    def __init__(self, in_channels, nr_out_channels_per_layer:list, last_layer_linear:bool=True):
        super(LipshitzMLP, self).__init__()

        self.last_layer_linear=last_layer_linear

        self.layers=torch.nn.ModuleList()
        for i in range(len(nr_out_channels_per_layer)):
            cur_out_channels=nr_out_channels_per_layer[i]
            self.layers.append(  torch.nn.Linear(in_channels, cur_out_channels)   )
            in_channels=cur_out_channels
        apply_weight_init_fn(self, leaky_relu_init, negative_slope=0.0)
        if last_layer_linear:
            leaky_relu_init(self.layers[-1], negative_slope=1.0)

        #we make each weight separately because we want to add the normalize to it
        self.weights_per_layer=torch.nn.ParameterList()
        self.biases_per_layer=torch.nn.ParameterList()
        for i in range(len(self.layers)):
            self.weights_per_layer.append( self.layers[i].weight  )
            self.biases_per_layer.append( self.layers[i].bias  )

        self.lipshitz_bound_per_layer=torch.nn.ParameterList()
        for i in range(len(self.layers)):
            max_w= torch.max(torch.sum(torch.abs(self.weights_per_layer[i]), dim=1))
            #we actually make the initial value quite large because we don't want at the beggining to hinder the rgb model in any way. A large c means that the scale will be 1
            c = torch.nn.Parameter(  torch.ones((1))*max_w*2 ) 
            self.lipshitz_bound_per_layer.append(c)


        self.weights_initialized=True #so that apply_weight_init_fn doesnt initialize anything

    def normalization(self, w, softplus_ci):
        absrowsum = torch.sum(torch.abs(w), dim=1)
        # scale = torch.minimum(torch.tensor(1.0), softplus_ci/absrowsum)
        # this is faster than the previous line since we don't constantly recreate a torch.tensor(1.0)
        scale = softplus_ci/absrowsum
        scale = torch.clamp(scale, max=1.0)
        return w * scale[:,None]

    def lipshitz_bound_full(self):
        lipshitz_full=1
        for i in range(len(self.layers)):
            lipshitz_full=lipshitz_full*torch.nn.functional.softplus(self.lipshitz_bound_per_layer[i])

        return lipshitz_full

    def forward(self, x):
        for i in range(len(self.layers)):
            weight=self.weights_per_layer[i]
            bias=self.biases_per_layer[i]

            weight=self.normalization(weight, torch.nn.functional.softplus(self.lipshitz_bound_per_layer[i])  )

            x=torch.nn.functional.linear(x, weight, bias)

            is_last_layer=i==(len(self.layers)-1)

            if is_last_layer and self.last_layer_linear:
                pass
            else:
                x=torch.nn.functional.gelu(x)


        return x


class NRC_MLP(nn.Module):
    def __init__(
            self, 
            mlp_width=64,
            mlp_depth=5,
            hash_enc=None, 
            ref_factor=True, 
            dtype=torch.float32, 
            use_lipshitz:bool=False,
            ):
        '''
            Note that the actual input dim is 62
            We pad two colums of 1 to allow implicit bias

            ira_factor: Reflectance factorization
        '''
        super(NRC_MLP, self).__init__()
        self.ref_factor = ref_factor
        self.hash_enc = hash_enc
        self.use_lipshitz = use_lipshitz

        input_dim = 64
        if hash_enc is not None:
            input_dim = 64 - 36 + hash_enc.output_dim

        if use_lipshitz:
            channels = [mlp_width for _ in range(mlp_depth)] + [3]
            self.mlp = LipshitzMLP(input_dim, channels)
        else:
            mlp_layers = [
                nn.Linear(input_dim, mlp_width, bias=False).to(dtype=dtype),
                nn.ReLU(),
            ]

            for _ in range(mlp_depth):
                mlp_layers.append(nn.Linear(mlp_width, mlp_width, bias=False).to(dtype=dtype))
                mlp_layers.append(nn.ReLU())

            mlp_layers.append(nn.Linear(mlp_width, 3, bias=False).to(dtype=dtype))


            self.mlp = nn.Sequential(*mlp_layers)

        self.freq_embed = encoder.FreqEmbed()
        self.blob_embd = encoder.OneBlobEmbed()


    @property
    def device(self):
        return next(self.mlp[0].parameters()).device

    def forward(self, p, wi, n, alpha, beta, r, apply_factorization=True):
        '''
        p: [B, 3] position
        wi: [B, 3] scattered dir
        n: [B, 3] surface normal
        alpha: [B, 3] diffuse reflectance
        beta: [B, 3] specular reflectance
        r: [B, 1] surface roughness
        '''

        if self.hash_enc is not None:
            p_enc = self.hash_enc(p)
        else:
            p_enc = self.freq_embed.embed(p)

        input = torch.concat(
            [
                p_enc,
                # self.blob_embd.embed(to_sphe_coords(wi)),
                # self.blob_embd.embed(to_sphe_coords(n)),
                self.blob_embd.embed(wi),
                self.blob_embd.embed(n),
                # self.blob_embd.embed(1.-torch.exp(-r)),
                self.blob_embd.embed(r), # roughness is already applied with 1-exp(-r)
                alpha,
                beta,
                torch.ones_like(p[:,:2])
            ],
            axis=-1
        )


        out = self.mlp(input)

        if self.ref_factor and apply_factorization:
            out = out * (alpha + beta)
            
        # out = out / self.radiance_scale

            
        return out
        # return alpha

class NRC:
    def __init__(
            self, 
            mlp_width=64,
            mlp_depth=5,
            use_hash_grid=True,
            ref_factor=True, 
            mlp_ma_alpha=0.99, 
            dtype=torch.float32, 
            device='cuda', 
            use_lipshitz=False,
            **kwargs,
            ):
        '''
            Note that the actual input dim is 62
            We pad two colums of 1 to allow implicit bias

            ref_factor: Reflectance factorization
            ma_alpha: moving average alpha
        '''
        self.hash_enc = None
        if use_hash_grid:
            self.hash_enc = encoding.MultiResHashGrid(3).to(device)
        self.train_mlp = NRC_MLP(mlp_width, mlp_depth, self.hash_enc, ref_factor, dtype, use_lipshitz=use_lipshitz).to(device)
        self.inference_mlp = deepcopy(self.train_mlp)


        # moving average alpha
        self.ma_alpha = mlp_ma_alpha
        if self.ma_alpha == 1:
            # disable moving average
            self.inference_mlp = self.train_mlp

        self.t = 0

        self.dtype = dtype
        self.num_rendered_frame = 0
        self.device = device

    def parameters(self):
        return self.train_mlp.parameters()

    def _to_tensor(self, x):
        if torch.is_tensor(x):
            return x.to(device=self.device, dtype=self.dtype)
        else:
            return torch.tensor(x, device=self.device, dtype=self.dtype)


    def infer(self, p, wi, n, alpha, beta, r):
        '''
        Forward function for np input
        '''

        with torch.no_grad():
            ret = self.inference_mlp(
            self._to_tensor(p),
            self._to_tensor(wi),
            self._to_tensor(n),
            self._to_tensor(alpha),
            self._to_tensor(beta),
            self._to_tensor(r),
            )

        return ret
    
    def batch_infer(self, inputs, batch_size=8196):
        p, wi, n, alpha, beta, r = inputs

        infer_outs = []

        for i in range(0, p.shape[0], batch_size):
            l, u = i, min(i+batch_size, p.shape[0])
            infer_outs.append(self.infer(
                p[l:u], 
                wi[l:u], 
                n[l:u], 
                alpha[l:u], 
                beta[l:u], 
                r[l:u],
                ))
        infer_outs = torch.concat(infer_outs,axis=0)

        return infer_outs
    
    def  tv_reg(self, ratio=0.0001):
        '''
        Apply total variation loss on part of the grids
        '''

        size = self.hash_enc.finest_resolution - 1

        N = int((self.hash_enc.finest_resolution)**3 * ratio)

        coords = torch.randint(0, size, (N, 3)).to(self.device)

        latents = self.hash_enc(coords/size)

        tv_loss = 0


        for ax_i in range(3):
            coords_2 = coords.clone()
            coords_2[:,ax_i] += 1
            coords_2 = torch.clamp_max(coords_2, size)

            latents_2 = self.hash_enc(coords_2/size)

            # tv_loss += ((latents - latents_2)**2).mean()
            tv_loss += (torch.abs(latents - latents_2)).mean()


        return tv_loss



    def train(
            self, 
            inputs, 
            targets, 
            optimizer, 
            batch_num=4, 
            relative_loss=True, 
            apply_factorization=True, 
            lambda_tv=0.,
            lambda_lipshitz=0.,
            wandb=None
        ):
        '''
        batch_num: slipt data into several batches and perform multiple training steps
        relative_loss: needed when expected outputs are noisy. Paper Sec 5
        '''
        loss_log = {
            'render_loss' : 0.,
            'tv_loss' : 0.,
            'lipshitz_loss' : 0.,
        }

        targets = self._to_tensor(targets)
        p, wi, n, alpha, beta, r = inputs

        # assert outputs.shape[0] % batch_num == 0, 'Batch slipt uneven!'

        # ids = np.array_split(np.random.permutation(targets.shape[0]), batch_num)
        ids = torch.split(torch.randperm(targets.shape[0]), targets.shape[0] // batch_num)
        
        # multiple steps per frame
        for i in range(batch_num):
            loss = 0.
            optimizer.zero_grad()
            pred = self.train_mlp(
                self._to_tensor(p[ids[i]]),
                self._to_tensor(wi[ids[i]]),
                self._to_tensor(n[ids[i]]),
                self._to_tensor(alpha[ids[i]]),
                self._to_tensor(beta[ids[i]]),
                self._to_tensor(r[ids[i]]),
                apply_factorization=apply_factorization,
            )
            render_loss = (targets[ids[i]] - pred)**2

            if relative_loss:
                luminance = 0.299 * pred[:,0] + 0.587 * pred[:,1] + 0.114 * pred[:,2]
                render_loss = render_loss / (luminance[:, None].detach().clone()**2 + 1e-5)
            
            render_loss = torch.mean(render_loss)
            loss += render_loss

            if lambda_tv > 0.:
                # apply TV
                tv_loss = self.tv_reg()
                loss += tv_loss * lambda_tv
            else:
                tv_loss = torch.tensor(0.)

            if lambda_lipshitz > 0.:
                # apply TV
                lipshitz_loss = self.train_mlp.mlp.lipshitz_bound_full()[0]
                loss += lipshitz_loss * lambda_lipshitz
            else:
                lipshitz_loss = torch.tensor(0.)


            loss.backward()
            optimizer.step()

            # update inference mlp
            self.t += 1
            if self.ma_alpha < 1:
                state_wt = self.train_mlp.state_dict()
                state_inf = self.inference_mlp.state_dict()
                for key in state_inf:
                    # nau = lambda t: 1 - self.ma_alpha**t
                    # s = lambda t: ((1 - self.ma_alpha) / nau(t)) + (self.ma_alpha * nau(t-1))

                    state_inf[key] = (1 - self.ma_alpha) / (1-(self.ma_alpha**self.t)) * state_wt[key] \
                                    + self.ma_alpha * (1-(self.ma_alpha**(self.t-1))) * state_inf[key]
                    
                self.inference_mlp.load_state_dict(state_inf)


            loss_log['render_loss'] += render_loss.item()
            loss_log['tv_loss'] += tv_loss.item()
            loss_log['lipshitz_loss'] += lipshitz_loss.item()

        
        for k in loss_log.keys():
            loss_log[k] /= batch_num

        return loss_log