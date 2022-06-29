# Copyright (c) 2022 Erik Härkönen, Aalto University
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

from pathlib import Path
import torch
import numpy as np
import pickle
from viewer.utils import stack_to_global_t
from preproc.tools import VideoFrames
from training.networks_stylegan2 import ConditioningTransform
from training.dataset import ImageFolderDataset

# Class that emulates StyleGAN interface, but returns dataset frames
class DatasetGAN(torch.nn.Module):
    def __init__(self, pkl_path, dset_root) -> None:
        super().__init__()

        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            dataset_args = data['training_set_kwargs']
            G = data['G_ema']

        assert 'path' in dataset_args, 'Dataset name not known'
        dset_path = Path(dset_root) / Path(dataset_args['path']).name
        
        # 1. Convert zs to seeds
        self.mapping = MappingNetwork(G.z_dim, 0, 0, 1)
        
        # 2. Store ts passed in, return sinusoids (not used)
        self.cond_xform = CondXform(G.cond_xform)
        
        # 3. Takes stored ts, add noise using pseudo-seeds, sample dataset
        self.synthesis = SynthesisNetwork(w_dim=G.w_dim, img_resolution=G.img_resolution, cond_args=G.cond_args,
            img_channels=G.img_channels, cond_xform=self.cond_xform, mapping=self.mapping, dataset_path=dset_path)

        self.img_resolution = G.img_resolution
        self.num_ws = G.num_ws
        self.z_dim = G.z_dim
        self.cond_args = G.cond_args

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        cs = self.cond_xform(c, broadcast=True)
        ws = self.mapping(z, cs[:, 0, :], truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        img = self.synthesis(ws, cs, update_emas=update_emas, **synthesis_kwargs)
        return img

class CondXform(ConditioningTransform):
    def __init__(self, parent_xform):
        super().__init__(parent_xform.cond_args, parent_xform.num_ws, False)
        self.train()
        self.ts_last = None

    def forward(self, cs):
        self.ts_last = cs.detach().clone()
        return super().forward(cs)

class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output, None = do not broadcast.
        num_layers      = 8,        # Number of mapping layers.
        embed_features  = None,     # Label embedding dimensionality, None = same as w_dim.
        layer_features  = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.998,    # Decay for tracking the moving average of W during training, None = do not track.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        self.seeds = None

    # Maps z to deterministic 'seed'
    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        vmin, vmax = (-1e5, 1e5)
        buckets = torch.linspace(vmin, vmax, 100_000_000, device=z.device)
        self.seeds = torch.searchsorted(buckets, z[:, 0].contiguous())
        return self.seeds.unsqueeze(-1).repeat((1, 512))

class SynthesisNetwork(torch.nn.Module):
    def __init__(self,
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output image resolution.
        img_channels,               # Number of color channels.
        channel_base    = 32768,    # Overall multiplier for the number of channels.
        channel_max     = 512,      # Maximum number of channels in any layer.
        num_fp16_res    = 4,        # Use FP16 for the N highest resolutions.
        cond_args       = {},       # Conditioning parameters.
        cond_xform      = None,     # used for dataset sampling
        mapping         = None,     # used for dataset sampling
        dataset_path    = None,
        **ignore,
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.c_dim = cond_args.dims
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.num_fp16_res = num_fp16_res
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        self.cond_xform = cond_xform
        self.mapping = mapping

        self.alpha_img = torch.ones((1, self.img_resolution, self.img_resolution), device='cuda')
        
        self.tol_hours = np.float('inf')
        self.add_noise = True
        self.vid = None

        if not (dataset_path.is_dir() or dataset_path.is_file()):
            print(f'ERROR: Dataset {dataset_path.name} not found under {dataset_path.parent}')
        else:
            self.vid = VideoFrames(dataset_path, torch=True, to_byte=False)
            self.parse_conds(dataset_path)

    # Recover ts from [cos, sin, cos, sin, ...]
    def sinusoids_to_ts(self, cs):
        fs = self.cond_xform.get_frequencies()
        two_pi_f_t = torch.atan2(cs[:, 0, 1::2], cs[:, 0, 0::2]).cpu().numpy() # [-pi, pi]
        f_t = two_pi_f_t / (2*np.pi) # t times f = phase in [-0.5, 0.5]
        ts = f_t / fs # ts that generates said phases
        
        # print(
        #     f'Angles: [{two_pi_f_t[0, 0]:.5f}, {two_pi_f_t[0, 1]:.5f}, {two_pi_f_t[0, 2]:.5f}],',
        #     f'Phases: [{f_t[0, 0]:.5f}, {f_t[0, 1]:.5f}, {f_t[0, 2]:.5f}],',
        #     f'ts: [{ts[0, 0]:.5f}, {ts[0, 1]:.5f}, {ts[0, 2]:.5f}]'
        # )
        
        return stack_to_global_t(ts, fs)

    def parse_conds(self, dset_path):
        training_set = ImageFolderDataset(str(dset_path), use_labels=True)
        self.labels = np.zeros(len(training_set))
        for idx in range(len(training_set)):
            label = training_set.get_details(idx).raw_label.flat[::-1]
            self.labels[idx] = label
        self.num_days = training_set._get_meta().get('num_days', 1)
        
        pH = training_set._get_meta().get('pad_horizontal', 0)
        pV = training_set._get_meta().get('pad_vertical', 0)
        res = training_set.resolution
        self.out_rect = (pH, pV, res - pH, res - pV)

    def forward(self, ws, cs, **block_kwargs):
        assert self.mapping.seeds.shape[0] == self.cond_xform.ts_last.shape[0], \
            "Seeds and ts don't match"

        # Dataset frames not available
        # => return zeros (mapped to gray color)
        if not self.vid:
            return torch.zeros((self.mapping.seeds.shape[0], 3, self.img_resolution, self.img_resolution), device=ws.device)
        
        imgs = []
        for i, s in enumerate(self.mapping.seeds):
            # Add noises if active
            self.cond_xform.add_noise = self.add_noise
            torch.manual_seed(s.item())
            np.random.seed(s.item())
            t_clean = self.cond_xform.ts_last[i:i+1]
            self.cond_xform.check_shapes(t_clean)
            ts = self.cond_xform.add_noises(t_clean).cpu().numpy()
            fs = self.cond_xform.get_frequencies()
            t = stack_to_global_t(ts, fs).reshape(-1)
            
            # Find best match in training data
            dists = np.abs(t - self.labels)
            idx = np.argmin(dists)
            if dists[idx]*self.num_days*24 < self.tol_hours:
                img = -1 + 2 * self.vid[idx].permute(2, 0, 1) / 255.0
                img = torch.cat((img, self.alpha_img), dim=0) # alpha one
            else:
                # Has zero alpha to indicate missing data
                img = -1 * torch.ones((4, *self.vid.out_res), dtype=torch.float32, device='cuda')
            imgs.append(img.unsqueeze(0))

        # Restore state
        self.mapping.seeds = None
        self.cond_xform.ts_last = None
        self.cond_xform.add_noise = False

        return torch.concat(imgs, dim=0)