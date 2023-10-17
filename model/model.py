import os
import torch
from torch import nn
from model import encoder
from model.utils import to_sphe_coords
import numpy as np
from copy import deepcopy
import encoding

class NRC_MLP(nn.Module):
    def __init__(self, input_dim=64, hash_enc=None, ref_factor=True, dtype=torch.float32, radiance_scale=1.):
        '''
            Note that the actual input dim is 62
            We pad two colums of 1 to allow implicit bias

            ira_factor: Reflectance factorization
        '''
        super(NRC_MLP, self).__init__()
        self.ref_factor = ref_factor
        self.hash_enc = hash_enc
        self.radiance_scale = radiance_scale

        if hash_enc is not None:
            input_dim = 64 - 36 + hash_enc.output_dim


        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64, bias=False).to(dtype=dtype),
            nn.ReLU(),
            nn.Linear(64, 64, bias=False).to(dtype=dtype),
            nn.ReLU(),
            nn.Linear(64, 64, bias=False).to(dtype=dtype),
            nn.ReLU(),
            nn.Linear(64, 64, bias=False).to(dtype=dtype),
            nn.ReLU(),
            nn.Linear(64, 64, bias=False).to(dtype=dtype),
            nn.ReLU(),
            nn.Linear(64, 64, bias=False).to(dtype=dtype),
            nn.ReLU(),
            nn.Linear(64, 3, bias=False).to(dtype=dtype),
        )

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
                self.blob_embd.embed(1.-torch.exp(-r)),
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
    def __init__(self, input_dim=64, ref_factor=True, ma_alpha=0.99, dtype=torch.float32, device='cuda', radiance_scale=1.):
        '''
            Note that the actual input dim is 62
            We pad two colums of 1 to allow implicit bias

            ref_factor: Reflectance factorization
            ma_alpha: moving average alpha
        '''

        hash_enc = encoding.MultiResHashGrid(3).to(device)
        # hash_enc = None
        self.train_mlp = NRC_MLP(input_dim, hash_enc, ref_factor, dtype, radiance_scale=radiance_scale).to(device)
        self.inference_mlp = deepcopy(self.train_mlp)
        self.radiance_scale = radiance_scale


        # moving average alpha
        self.ma_alpha = ma_alpha
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


    def train(self, inputs, targets, optimizer, batch_num=4, relative_loss=True, apply_factorization=True):
        '''
        batch_num: slipt data into several batches and perform multiple training steps
        relative_loss: needed when expected outputs are noisy. Paper Sec 5
        '''
        mean_loss = 0.

        targets = self._to_tensor(targets)
        p, wi, n, alpha, beta, r = inputs

        # assert outputs.shape[0] % batch_num == 0, 'Batch slipt uneven!'

        # ids = np.array_split(np.random.permutation(targets.shape[0]), batch_num)
        ids = torch.split(torch.randperm(targets.shape[0]), targets.shape[0] // batch_num)
        
        # multiple steps per frame
        for i in range(batch_num):
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
            loss = (targets[ids[i]] - pred)**2
            luminance = 0.299 * pred[:,0] + 0.587 * pred[:,1] + 0.114 * pred[:,2]

            if relative_loss:
                loss = loss / (luminance[:, None].detach().clone()**2 + 1e-5)
            loss = torch.mean(loss)
            loss.backward()
            optimizer.step()

            # update inference mlp
            self.t += 1
            if self.ma_alpha < 1:
                state_wt = self.train_mlp.state_dict()
                state_inf = self.inference_mlp.state_dict()
                for key in state_inf:
                    state_inf[key] = (1 - self.ma_alpha) / (1-(self.ma_alpha**self.t)) * state_wt[key] \
                                    + self.ma_alpha * (1-(self.ma_alpha**(self.t-1))) * state_inf[key]


            mean_loss += loss

        return mean_loss / batch_num