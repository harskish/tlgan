# Copyright (c) 2022 Erik Härkönen, Aalto University
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

"""Computes relative importance of input parameters
   z, c, and noise, by using ratios of variances, averaged
   over the output image tensor."""

import numpy as np
import torch
import copy

#----------------------------------------------------------------------------

def plot(t, title, outname, vmax=None, colorbar=True):
    import matplotlib
    matplotlib.use('Agg') # for writing to file
    import matplotlib.pyplot as plt
    from matplotlib.cm import gnuplot2 as colormap

    Z = np.mean(t, axis=0) # mean across RGB
    H, W = Z.shape
    aspect = W / H

    scale = 12
    plt.rcParams.update({'font.size': int(2*scale)})
    fig = plt.figure(figsize=(aspect*scale, scale))
    plt.title(title)
    plt.tight_layout()
    
    X, Y = np.meshgrid(
        np.linspace(-1, 1, W + 1), np.linspace(1, -1, H + 1))
    
    plt.axis('off')
    mesh = plt.pcolormesh(X, Y, Z, vmin=0, vmax=(vmax or t.max()), cmap=colormap, shading='flat')
    if colorbar:
        fig.colorbar(mesh)
        plt.savefig(outname)
    else:
        plt.savefig(outname, bbox_inches='tight', pad_inches=0)
    plt.close('all')

def broadcast_mean(t, num_gpus):
    if num_gpus > 1:
        ys = []
        for src in range(num_gpus):
            # sends y or receives into y, depending on rank
            y = t.clone()
            torch.distributed.broadcast(y, src=src)
            ys.append(y)
        t = torch.stack(ys, dim=0).mean(dim=0)
    return t

def run_G(G, *args, **kwargs):
    img = G(*args, **kwargs)
    x0, y0, x1, y1 = getattr(G.synthesis, 'out_rect', (0, 0, G.img_resolution, G.img_resolution))
    return img[:, :, y0:y1, x0:x1]

# Computes E[var(G)] using a single-sample Monte Carlo estimator of the variance.
# target_var: variable w.r.t which the variance is computed.
def var_mc_estimator(G, opts, n_samples, batch_gen, target_var, rel_range=(0,1)):
    assert target_var in ['z', 'c', 'noise'] or target_var.startswith('c')

    is_uncond = G.cond_args.type == 'none'
    is_eq = 'signal_func' in G.init_kwargs
    if (target_var.startswith('c') and is_uncond) or (target_var == 'noise' and is_eq):
        return [*np.zeros((2, 3, G.img_resolution, G.img_resolution), dtype=np.float32)]

    rand_n = lambda : 'random'
    rand_z = lambda : torch.randn((batch_gen, G.z_dim), device=opts.device) # Gaussian
    rand_c = lambda : torch.rand((batch_gen, G.c_dim), device=opts.device) # uniform
    def rand_c_per_f(c_start, idx):
        assert G.c_dim == 1, 'Not tested for vector-valued conditioning'
        cs = c_start.repeat_interleave(G.cond_xform.num_f, dim=1) # start from given global c
        cs[:, idx] = torch.rand(batch_gen, device=opts.device) # randomize input to single frequency
        return cs

    # Make divisible
    n_samples = ((n_samples - 1) // (batch_gen * opts.num_gpus) + 1) * (batch_gen * opts.num_gpus)
    samples_per_gpu = n_samples // opts.num_gpus
    progress = opts.progress.sub(tag=f'variance ratio {target_var}',
        num_items=samples_per_gpu, rel_lo=rel_range[0], rel_hi=rel_range[1])

    e_var_g = 0 # accumulated variance expectation
    e_std_g = 0 # accumulated stddev expectation
    with torch.no_grad():
        for i in range(0, samples_per_gpu, batch_gen):
            # Batch 1, no striding
            z = rand_z()
            c = rand_c()
            n = 'const' # not unique per batch index...
            i1 = run_G(G, z, c, noise_mode=n, **opts.G_kwargs)

            # Batch 2: vary only variance target variable
            if target_var == 'z':
                z = rand_z()
            elif target_var == 'noise':
                n = rand_n()
            elif target_var == 'c':
                c = rand_c()
            elif target_var.startswith('c'):
                idx = int(target_var[1:]) # index of frequency to wiggle
                c = rand_c_per_f(c, idx)
            else:
                raise RuntimeError(f'Invalid target variable "{target_var}"')

            # Accumulate minibatch of single-sample variance estimates
            i2 = run_G(G, z, c, noise_mode=n, **opts.G_kwargs)
            diff = i1 - i2
            e_var_g = e_var_g + torch.square(diff).sum(dim=0) / samples_per_gpu
            progress.update(i+batch_gen)

    # Average over GPUs
    e_var_g = broadcast_mean(e_var_g, opts.num_gpus)

    # Constant term (appendix eq. 5)
    e_var_g *= 0.5

    e_std_g = e_var_g.sqrt()
    return (e_var_g.cpu().numpy(), e_std_g.cpu().numpy())

# Computes E[var(G)] for each input variable z,c,noise.
# Returns ratios of expected variances, which can be
# used to rank the relative importance of the inputs.
def compute_var_ratio_mc(opts, n_samples, batch_gen=None):
    if batch_gen is None:
        batch_gen = 16 if opts.G.img_resolution <= 256 else 8

    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    n_tasks = 3 + (G.cond_args.dims // 2)

    # Collection 1: (z, c, n), i.e. single conditioning variable
    var_z, std_z = var_mc_estimator(G, opts, n_samples, batch_gen, 'z',     rel_range=(0/n_tasks, 1/n_tasks))
    var_c, std_c = var_mc_estimator(G, opts, n_samples, batch_gen, 'c',     rel_range=(1/n_tasks, 2/n_tasks))
    var_n, std_n = var_mc_estimator(G, opts, n_samples, batch_gen, 'noise', rel_range=(2/n_tasks, 3/n_tasks))
    var_total = np.clip(var_c + var_z + var_n, 1e-3, None)
    std_total = np.clip(std_c + std_z + std_n, 1e-3, None)

    res = {
        'var_z': float(np.mean(var_z / var_total)),
        'var_c': float(np.mean(var_c / var_total)),
        'var_noise': float(np.mean(var_n / var_total)),
        'std_z': float(np.mean(std_z / std_total)),
        'std_c': float(np.mean(std_c / std_total)),
        'std_noise': float(np.mean(std_n / std_total)),
    }
    
    # Collection 2: (z, <c_0 ... c_n>, n), i.e. separate cond. vars per frequency
    if G.cond_args.type in ['fourier', 'f_concat']:
        c_vars = []
        c_stds = []
        for i in range(G.cond_xform.num_f):
            var, std = var_mc_estimator(G, opts, n_samples, batch_gen, f'c{i}', rel_range=((3 + i)/n_tasks, (4 + i)/n_tasks))
            c_vars.append(var)
            c_stds.append(std)

        var_total = np.clip(var_z + var_n + sum(c_vars), 1e-3, None)
        std_total = np.clip(std_z + std_n + sum(c_stds), 1e-3, None)
        
        names = {
            1: ['lin'],
            2: ['lin', 'day'],
            3: ['lin', 'year', 'day']
        }
        
        labs = names[len(c_vars)]
        for i in range(len(c_vars)):
            res[f'var_{labs[i]}'] = float(np.mean(c_vars[i] / var_total))
            res[f'std_{labs[i]}'] = float(np.mean(c_stds[i] / std_total))

    if opts.rank != 0:
        return { k: float('nan') for k in res }

    return res

    
