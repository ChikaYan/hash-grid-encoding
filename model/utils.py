import numpy as np
import torch
import dataclasses
from copy import deepcopy, copy

import time

def _nonzero(x, eta=1e-10):
    # x = x.clone()
    # x[x==0.] = eta

    return torch.where(x==0., eta, x)

def to_sphe_coords(dirs):
    '''
    Convert [n, 3] dirs into [n, 2] spherical coordinates
    '''

    theta = torch.arctan(dirs[:,1] / _nonzero(dirs[:, 0]))
    phi = torch.arctan(torch.sqrt(dirs[:, 0]**2 + dirs[:, 1]**2) / _nonzero(dirs[:, 2]))

    return torch.stack([theta, phi], axis=-1)



class Timer:    
    def __init__(self, prompt='') -> None:
        self.prompt = prompt

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start
        if self.prompt is not None:
            print(f"{self.prompt}: {self.interval}")