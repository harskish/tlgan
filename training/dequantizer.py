# Copyright (c) 2022 Erik Härkönen, Aalto University
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

import numpy as np
import torch
from torch_utils import misc
from torch_utils import persistence

# Input c: training frame cond, one of real_cs
# Output c: distribution of cs that cover whole real axis
# Mode: type of filter for 'splatting'
class Dequantizer(torch.nn.Module):
    def __init__(self, cond_args, dataset) -> None:
        super().__init__()
        self.mode = cond_args.dequant
        self.compute_deltas(dataset)

    def compute_deltas(self, dataset):
        assert dataset.get_label(0).size == 1, 'Vector-valued c not tested'
        labels = np.sort([dataset.get_label(i) for i in range(len(dataset))]).squeeze()
        deltas = np.hstack(([0], labels[1:] - labels[:-1], [0]))
        self.cs = torch.tensor(labels, device='cpu', dtype=torch.float32, requires_grad=False)
        self.deltas = torch.tensor(deltas, device='cpu', dtype=torch.float32, requires_grad=False)

    # Finds closest match using binary search
    def find_closest_c(self, c):
        assert c.ndim == 2 and c.shape[-1] == 1, 'Only scalar-valued c tested'
        return torch.searchsorted(self.cs, c).clip(0, len(self.cs) - 1).view(-1)

    def forward(self, c):        
        assert c.device.type == 'cpu'
    
        ind = self.find_closest_c(c)
        noise = 0
        if self.mode == 'gauss':
            delta = torch.maximum(self.deltas[ind], self.deltas[ind + 1]) # larger distance to neighbor
            noise = 0.5 * delta.view(c.shape) * torch.randn_like(c) # two deltas on avg
        elif self.mode == 'tent':
            u = -1 + 2 * torch.rand(c.shape)
            delta = torch.where(u.view(-1) < 0, self.deltas[ind], self.deltas[ind + 1])
            noise = delta.view(c.shape) * u.sign() * (1 - u.abs().sqrt())
        elif self.mode == 'none':
            pass
        else:
            raise RuntimeError('Unknown dequantization filter')

        #import matplotlib.pyplot as plt
        #plt.hist(noise.squeeze().detach().numpy(), bins=30)
        #plt.show()

        misc.assert_shape(noise, c.shape)
        return c + noise

#----------------------------------------------------------------------------