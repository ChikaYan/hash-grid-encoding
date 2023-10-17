import torch
from torch import nn
import numpy as np

class FreqEmbed:
    N_freqs: int = 12
    periodic_fns: list = [torch.sin]

    def __init__(self, input_dims=3):
        self.input_dims = input_dims
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.input_dims
        out_dim = 0
            
        N_freqs = self.N_freqs
        max_freq = N_freqs - 1
        
        freq_bands = torch.arange(0., N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * 2**freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

class OneBlobEmbed:

    # range of value
    s_range = [-np.pi, np.pi] 
    # number of bins
    k_bin: int = 4
    

    def __init__(self):
        self.create_kernel()

    def create_kernel(self):
        bin_bound = torch.linspace(self.s_range[0], self.s_range[1], self.k_bin+1)
        self.bin_low = bin_bound[:-1]
        self.bin_up = bin_bound[1:]
        self.bin_up[-1] += 1e-6

        sigma = 1./self.k_bin

        self.kernel = torch.zeros([self.k_bin, self.k_bin])

        for i in range(self.k_bin):
            for j in range(self.k_bin):
                self.kernel[i,j] = i-j
        self.kernel = torch.exp(-(self.kernel)**2 / (2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
    
    def embed(self, inputs):
        assert ((self.s_range[0] <= inputs) & (inputs <= self.s_range[1])).all(), 'Input values are out of range!'
        B = inputs.shape[0]
        self.bin_low = self.bin_low.to(inputs.device)
        self.bin_up = self.bin_up.to(inputs.device)
        self.kernel = self.kernel.to(inputs.device)

        input_bin = inputs.clone().view(-1)
        for i in range(self.k_bin):
            mask = (self.bin_low[i] <= input_bin) & (input_bin < self.bin_up[i])
            input_bin[mask] = i

        one_hot = torch.nn.functional.one_hot(input_bin.long(), num_classes=self.k_bin)

        # apply gaussian kernel
        one_blob = torch.zeros_like(one_hot, dtype=inputs.dtype)
        for i in range(self.k_bin):
            one_blob[:, i] = torch.sum(self.kernel[i, :] * one_hot, axis=-1)

        one_blob = one_blob.reshape([B, -1])

        return one_blob