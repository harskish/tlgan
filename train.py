# Copyright (c) 2022 Erik Härkönen, Aalto University
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
# This is a derivative work of StyleGAN3 by NVIDIA Corporation

"""Train a GAN using the techniques described in the paper
"Disentangling Random and Cyclic Effects from Time-Lapse Sequences"."""

import os
import click
import re
import json
import tempfile
import torch
import numpy as np
from typing import Iterable
from pathlib import Path

import dnnlib
from training import training_loop, cond_specs
from training.dataset import ImageFolderDataset
from torch_utils import training_stats
from torch_utils import custom_ops

#----------------------------------------------------------------------------

def subprocess_fn(rank, c, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if c.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=c.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=c.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    training_loop.training_loop(rank=rank, **c)

#----------------------------------------------------------------------------

def launch_training(c, desc, outdir, dry_run):
    dnnlib.util.Logger(should_flush=True)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    c.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
    assert not os.path.exists(c.run_dir)

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(c, indent=2, sort_keys=True))
    print()
    print(f'Output directory:    {c.run_dir}')
    print(f'Number of GPUs:      {c.num_gpus}')
    print(f'Batch size:          {c.batch_size} images')
    print(f'Training duration:   {c.total_kimg} kimg')
    print(f'Conditioning:        {c.cond_args.type}')
    print(f'Dataset path:        {c.training_set_kwargs.path}')
    print(f'Dataset size:        {c.training_set_kwargs.max_size} images')
    print(f'Dataset resolution:  {c.training_set_kwargs.resolution}')
    print(f'Dataset labels:      {c.training_set_kwargs.use_labels}')
    print(f'Dataset x-flips:     {c.training_set_kwargs.xflip}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(c.run_dir)
    with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(c, f, indent=2)

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if c.num_gpus == 1:
            subprocess_fn(rank=0, c=c, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(c, temp_dir), nprocs=c.num_gpus)

#----------------------------------------------------------------------------

def arch_sg2(c, opts, n_gpus):
    c.G_kwargs = dnnlib.EasyDict(class_name=None, z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict())
    c.D_kwargs = dnnlib.EasyDict(class_name='training.networks_stylegan2.Discriminator', block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    c.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8, lr=opts.glr)
    c.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8, lr=opts.dlr)
    c.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.StyleGAN2Loss')
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2, num_workers=opts.workers)

    # Hyperparameters & settings.
    res = c.training_set_kwargs.resolution
    c.batch_size = opts.batch # 32
    c.batch_gpu =  min(4096 // c.training_set_kwargs.resolution, c.batch_size // n_gpus) if opts.batch_gpu is None else opts.batch_gpu
    c.D_kwargs.epilogue_kwargs.mbstd_group_size = min(c.batch_gpu, 4)

    c.num_gpus = n_gpus
    c.G_kwargs.channel_base = c.D_kwargs.channel_base = opts.cbase # 16384 also OK-ish for 256x256
    c.G_kwargs.channel_max = c.D_kwargs.channel_max = opts.cmax # 512
    c.G_kwargs.mapping_kwargs.num_layers = 2 if opts.map_depth is None else opts.map_depth
    c.D_kwargs.block_kwargs.freeze_layers = opts.freezed # 0
    c.G_kwargs.out_rect = (0, 0, res, res)
    c.metrics = opts.metrics
    c.ema_kimg = c.batch_size * 10 / 32
    
    if opts.aug != 'noaug':
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)
        if opts.aug == 'ada':
            c.ada_target = opts.target
        if opts.aug == 'fixed':
            c.augment_p = opts.p

    c.G_kwargs.class_name = 'training.networks_stylegan2.Generator'
    c.loss_kwargs.style_mixing_prob = 0.9 # Enable style mixing regularization.
    c.loss_kwargs.pl_weight = 2 # Enable path length regularization.
    c.G_reg_interval = 4 # Enable lazy regularization for G.
    c.G_kwargs.fused_modconv_default = 'inference_only' # Speed up training by using regular convolutions instead of grouped convolutions.
    c.loss_kwargs.pl_no_weight_grad = True # Speed up path length regularization by skipping gradient computation wrt. conv2d weights.
    c.loss_kwargs.r1_gamma = (c.training_set_kwargs.resolution / 256) ** 2 if opts.gamma is None else opts.gamma # {128: 0.25, 256: 1, 512: 4, 1024: 16}
    
    return f'sg2-a{n_gpus}'

def _set_noise_global(c, noise, n_frames, n_days):
    scale = parse_noises([noise], c, n_frames, n_days)[0]
    c.cond_args.noise = scale # global noise, measured in frame deltas
    c.cond_args.noise_f = [] # disable per-f noise
    return f'-n{str(noise).replace(" ", "")}'

def _set_noise_per_f(c, noises, n_frames, n_days):
    scales = parse_noises(noises, c, n_frames, n_days)
    assert np.all(np.diff(scales) <= 0), 'Noises not descending, are you sure?'
    c.cond_args.noise = 0        # disable global noise
    c.cond_args.noise_f = scales # in frame deltas, for [1e-3f] + [manual freqs]
    return '-n_' + '_'.join([str(f).replace(' ', '') for f in noises])

def _set_freqs(c, base_freqs, explicit_lin):
    assert c.cond_args.type in ['fourier', 'f_concat']
    
    desc = ''
    if explicit_lin:
        c.cond_args.f_manual = [*base_freqs]
        c.cond_args.include_lin = True
        c.cond_args.dims = 2*(len(c.cond_args.f_manual) + 1)
        desc += '-fman_lin'
    else:
        c.cond_args.f_manual = [1e-3, *base_freqs]
        c.cond_args.include_lin = False
        c.cond_args.dims = 2*len(c.cond_args.f_manual)
        desc += '-fman_impl'

    if base_freqs:
        desc += "_" + "_".join([str(int(round(f))) for f in base_freqs])

    return desc

# Map noise magnitude to frame deltas
def days(fr_tot, d_tot):
    return fr_tot / d_tot # one sigma in both directions
def hours(fr_tot, d_tot):
    return days(fr_tot, d_tot) / 24
def weeks(fr_tot, d_tot):
    return days(fr_tot, d_tot) * 7
def months(fr_tot, d_tot):
    return days(fr_tot, d_tot) * (365.25/12) # avg days in month
def years(fr_tot, d_tot):
    return days(fr_tot, d_tot) * 365.25

# Convert strings like '2.5years' to sigmas
def parse_noises(noises, c=None, n_frames=None, n_days=None):
    ret = []
    for n in noises:
        if isinstance(n, (float, int)):
            ret.append(n)
        elif 'hour' in n:
            ret.append(hours(n_frames, n_days)*float(n.split('hour')[0]))
        elif 'day' in n:
            ret.append(days(n_frames, n_days)*float(n.split('day')[0]))
        elif 'week' in n:
            ret.append(weeks(n_frames, n_days)*float(n.split('week')[0]))
        elif 'month' in n:
            ret.append(months(n_frames, n_days)*float(n.split('month')[0]))
        elif 'year' in n:
            ret.append(years(n_frames, n_days)*float(n.split('year')[0]))
        else:
            raise RuntimeError(f'Unkown noise scale: {n}')
    
    assert len(ret) == len(noises)
    return ret

def dataset(c, opts, path, cond_type, f=[], noise=[], mask=None, explicit_lin=True):
    try:
        res = list(r for r in [128, 256, 512, 1024] if str(r) in path)
        assert len(res) == 1, 'Path does not indicate resolution'
        res = res[0]
        
        # Training set
        c.training_set_kwargs = dnnlib.EasyDict(
            class_name='training.dataset.ImageFolderDataset',
            path=path, resolution=res, max_size=None, xflip=opts.mirror, use_labels=cond_type != 'none')
        dset = ImageFolderDataset(c.training_set_kwargs.path)
        frames = len(dset)
        meta = dset._get_meta()

        # Must know number of days in sequence
        days = meta.get('num_days', None) if opts.days is None else opts.days
        assert days is not None, 'Number of days not in dataset metadata, must specify manually with --days'

        # Hyperparameters & settings.
        c.total_kimg = opts.kimg
        c.kimg_per_tick = opts.tick
        c.image_snapshot_ticks = c.network_snapshot_ticks = opts.snap
        c.random_seed = c.training_set_kwargs.random_seed = opts.seed

        # Init network arch based on dataset resolution
        desc = arch_sg2(c, opts, n_gpus=opts.gpus)

        # Non-square datasets: define content rect, for masking
        # This improves training dynamics (crips borders from the very beginning)
        if mask is None:
            mask = (meta.get('pad_horizontal', 0), meta.get('pad_vertical', 0))

        rect = [mask[0], mask[1], res - mask[0], res - mask[1]]
        assert 0 <= rect[0] < rect[2] <= res, 'Invalid out rect'
        assert 0 <= rect[1] < rect[3] <= res, 'Invalid out rect'
        c.G_kwargs.out_rect = tuple(rect) # (x1, y1, x2, y2)
        
        name = Path(path).stem.split('_')[0]
        desc += f'-{name}_{res}'

        c.cond_args = dnnlib.EasyDict(cond_specs.specs[cond_type])
        desc += {'concat': '-concat', 'none': '-uncond', 'f_concat': '-fcat', 'fourier': ''}[cond_type]

        if cond_type in ['fourier', 'f_concat']:
            freqs = f or list(filter(lambda f: f > 1, [days/365.25, days])) # only cylces of over 1Hz
            desc += _set_freqs(c, freqs, explicit_lin)
        
        if 'auto' in noise:
            noise_lin = [] if days < 365.25 else [f'{0.2 * days / 365.25:.2f} years'] # fifth of whole sequence length
            desc += _set_noise_per_f(c, [*noise_lin, '4 days', 0], frames, days) # lin, years, days
        elif len(noise) == 1:
            desc += _set_noise_global(c, noise, frames, days)
        elif isinstance(noise, Iterable) and len(noise) > 0:
            desc += _set_noise_per_f(c, noise, frames, days) # lin, years, days

        return desc
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

#----------------------------------------------------------------------------

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

#----------------------------------------------------------------------------

@click.command()

# Required.
@click.option('--outdir',       help='Where to save the results', metavar='DIR',                          required=True)
@click.option('--data',         help='Training data', metavar='[ZIP|DIR]',                                type=str, required=True)
@click.option('--gpus',         help='Number of GPUs to use', metavar='INT',                              type=click.IntRange(min=1), required=True)
@click.option('--batch',        help='Total batch size', metavar='INT',                                   type=click.IntRange(min=1), required=True)

# Optional features.
@click.option('--mirror',       help='Enable dataset x-flips', metavar='BOOL',                            type=bool, default=False, show_default=True)
@click.option('--aug',          help='Augmentation mode',                                                 type=click.Choice(['noaug', 'ada', 'fixed']), default='ada', show_default=True)
@click.option('--resume',       help='Resume from given network pickle', metavar='[PATH|URL]',            type=str)
@click.option('--freezed',      help='Freeze first layers of D', metavar='INT',                           type=click.IntRange(min=0), default=0, show_default=True)

# Time-lapse specific hyperparameters.
@click.option('--cond_lr',      help='Learning rate for all conditioning signals', metavar='FLOAT',       type=click.FloatRange(min=0), default=1.0, show_default=True)
@click.option('--trend_lr',     help='Additional LR multiplier for trend signal', metavar='FLOAT',        type=click.FloatRange(min=0), default=1.0, show_default=True)
@click.option('--cond',         help='Conditioning type',                                                 type=click.Choice(['fourier', 'f_concat', 'concat', 'none']), default='fourier', show_default=True)
@click.option('--noise',        help='Timestamp augmentation noise', metavar='[NAME|A,B,C|auto]',         type=parse_comma_separated_list, default='auto', show_default=True)
@click.option('--days',         help='Number of days in sequence [default: use metadata]', metavar='INT', type=int, default=None)

# Misc hyperparameters.
@click.option('--gamma',        help='R1 regularization weight [default: varies]', metavar='FLOAT',       type=click.FloatRange(min=0))
@click.option('--p',            help='Probability for --aug=fixed', metavar='FLOAT',                      type=click.FloatRange(min=0, max=1), default=0.2, show_default=True)
@click.option('--target',       help='Target value for --aug=ada', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.6, show_default=True)
@click.option('--batch-gpu',    help='Limit batch size per GPU', metavar='INT',                           type=click.IntRange(min=1))
@click.option('--cbase',        help='Capacity multiplier', metavar='INT',                                type=click.IntRange(min=1), default=32<<10, show_default=True)
@click.option('--cmax',         help='Max. feature maps', metavar='INT',                                  type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--glr',          help='G learning rate', metavar='FLOAT',                                  type=click.FloatRange(min=0), default=0.002, show_default=True)
@click.option('--dlr',          help='D learning rate', metavar='FLOAT',                                  type=click.FloatRange(min=0), default=0.002, show_default=True)
@click.option('--map-depth',    help='Mapping network depth  [default: varies]', metavar='INT',           type=click.IntRange(min=1))

# Misc settings.
@click.option('--desc',         help='String to include in result dir name', metavar='STR',               type=str)
@click.option('--metrics',      help='Quality metrics', metavar='[NAME|A,B,C|none]',                      type=parse_comma_separated_list, default='fid50k_full,var_ratio_mc_5k', show_default=True)
@click.option('--kimg',         help='Total training duration', metavar='KIMG',                           type=click.IntRange(min=1), default=8_000, show_default=True)
@click.option('--tick',         help='How often to print progress', metavar='KIMG',                       type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--snap',         help='How often to save snapshots', metavar='TICKS',                      type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--seed',         help='Random seed', metavar='INT',                                        type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--fp32',         help='Disable mixed-precision', metavar='BOOL',                           type=bool, default=False, show_default=True)
@click.option('--nobench',      help='Disable cuDNN benchmarking', metavar='BOOL',                        type=bool, default=False, show_default=True)
@click.option('--workers',      help='DataLoader worker processes', metavar='INT',                        type=click.IntRange(min=1), default=3, show_default=True)
@click.option('-n','--dry-run', help='Print training options and exit',                                   is_flag=True)

def main(**kwargs):
    """Train a GAN using the techniques described in the paper
    "Disentangling Random and Cyclic Effects from Time-Lapse Sequences".

    Examples:

    \b
    # Train TLGAN on Valley using 4 GPUs.
    python train.py --outdir=~/training-runs --data=~/datasets/valley_1024x1024_2225hz.zip --gpus=4 --batch=32

    \b
    # Train an unconditional StyleGAN2 on Teton using 2 GPUs.
    python train.py --outdir=~/training-runs --data=~/datasets/teton_512x512_2225hz.zip --gpus=2 --batch=32 --cond=none --metrics=fid50k_full
    """

    # Initialize config and dataset.
    opts = dnnlib.EasyDict(kwargs) # Command line arguments.
    c = dnnlib.EasyDict() # Main config dict.
    desc = dataset(c, opts, path=opts.data, cond_type=opts.cond, noise=opts.noise)

    # Sanity checks.
    if c.cond_args.type in ['fourier', 'f_concat']:
        assert len(c.cond_args.noise_f) in [0, len(c.cond_args.f_manual) + int(c.cond_args.include_lin)], 'Wrong number of per-f noises'
        assert all(np.sort(c.cond_args.f_manual) == c.cond_args.f_manual), 'f_manual not sorted'
        assert not any('var_ratio' in n for n in c.metrics) or len(c.cond_args.f_manual) in [0, 1, 2, 3], 'Var ratio metric will fail!'
    if c.cond_args.type == 'none':
        assert c.training_set_kwargs.use_labels == False, 'Labels not disabled for uncond model'
    assert c.batch_size % c.num_gpus == 0, '--batch must be a multiple of --gpus'
    assert c.batch_size % (c.num_gpus * c.batch_gpu) == 0, '--batch must be a multiple of --gpus times --batch-gpu'
    assert c.batch_gpu >= c.D_kwargs.epilogue_kwargs.mbstd_group_size, '--batch-gpu cannot be smaller than --mbstd'

    # Resume.
    if opts.resume is not None:
        c.resume_pkl = opts.resume
        c.ada_kimg = 100 # Make ADA react faster at the beginning.
        c.ema_rampup = None # Disable EMA rampup.
        c.loss_kwargs.blur_init_sigma = 0 # Disable blur rampup.

    # Performance-related toggles.
    if opts.fp32:
        c.G_kwargs.num_fp16_res = c.D_kwargs.num_fp16_res = 0
        c.G_kwargs.conv_clamp = c.D_kwargs.conv_clamp = None
    if opts.nobench:
        c.cudnn_benchmark = False

    # Description string.
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Launch.
    launch_training(c=c, desc=desc, outdir=opts.outdir, dry_run=opts.dry_run)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------