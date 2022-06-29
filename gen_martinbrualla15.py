# Copyright (c) 2022 Jaakko Lehtinen, Aalto University.
# Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

"""Run the smoothing of Martin-Brualla SIGGRAPH2015 on the input raw timelapse."""

from asyncio import streams
from pathlib import Path
#from xmlrpc.client import Boolean
import zipfile
import json
import os
import tqdm

import sys
sys.path.append('ext/stylegan3') # or sg2
from training.dataset import ImageFolderDataset

import click
import numpy as np
import imageio
import torch
import torch.fft
import torch.optim
import PIL.Image

from matplotlib import pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.__version__)

#----------------------------------------------------------------------------

def totorch(x, dtype=torch.float32, device=device):
  return torch.tensor(x, device=device, dtype=dtype)

def tonumpy(x):
  if x.shape[0] == 3:
    return x.detach().cpu().numpy().transpose(1, 2, 0)
  else:
    return x.detach().cpu().numpy()[0, :, :]

def myimshow(x):
  plt.imshow(x)
  plt.gca().get_xaxis().set_visible(False) 
  plt.gca().get_yaxis().set_visible(False)
  plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

pad     = lambda x, PAD: torch.nn.functional.pad(x, pad=(PAD, PAD))
unpad   = lambda x, PAD: x[:, :, :, PAD:-PAD]

def init_weights(shape, falloff):
    freqt   = np.fft.fftfreq(shape[3], d=1.0)
    weights = falloff(freqt)
    # weights = weights / np.max(np.abs(weights))
    weights = weights / np.sqrt(np.sum(np.power(weights, 2.0)))
    weights = totorch(weights)[np.newaxis, np.newaxis, np.newaxis]

    return weights


def blur_1perw(img, weights, PAD):
    Fimg    = torch.fft.fftn(pad(img, PAD), dim=3)
    IFimg   = torch.fft.ifftn(Fimg * weights, dim=3)
    return unpad(torch.real(IFimg), PAD)

#----------------------------------------------------------------------------


@click.command()
@click.option('--dataset', help='Dataset name', type=str, default='c:/data/timelapse/mppv3_1024x1024_143hz.zip')
@click.option('--outdir', help='Where to save the output images', type=str, default="out", metavar='DIR')
@click.option('--crop', help='Crop rectangle (x1, y1, x2, y2)', type=click.Tuple([int, int, int, int]), default=(-1, 177, -1, 847), show_default=True)
@click.option('--startframe', help='Index of first frame in dataset', type=int, default=0)
@click.option('--numframes', help='How many frames to include', type=int, default=1200)
@click.option('--frameskip', help='Skip every N frames', type=int, default=5)
@click.option('--slice_size', help='How many columns to process at once', type=int, default=48)
@click.option('--inner_iterations', help='How many frames to run optimization on each x slice', type=int, default=400)
@click.option('--smooth_alpha', help='Weight for smoothness term', type=float, default=100.0)
@click.option('--visualize', help='Debug visualization', is_flag=True, default=True)
@click.option('--data_loss', help='Data fidelity loss', default='Huber', type=click.Choice(['L2', 'L1', 'Huber']))
@click.option('--smoothness_loss', help='Smoothness penalty', default='Huber', type=click.Choice(['L2', 'L1', 'Huber']))
def generate_martinbrualla15(
    dataset: str,
    outdir: str,
    startframe: int,
    numframes: int,
    frameskip: int,
    slice_size: int,
    inner_iterations: int,
    smooth_alpha: float,
    visualize: bool,
    data_loss: str,
    smoothness_loss: str,
    crop: tuple
):
    os.makedirs(outdir, exist_ok=True)
    _, dataset_zipfile = os.path.split(dataset)

    # Get metadata first
    meta = {}
    labs = []
   
    with zipfile.ZipFile(dataset, mode='r') as z:
        with z.open('dataset.json', 'r') as file:
            obj = json.load(file)
            meta = obj['meta']
            labs = obj['labels']
    
    t_start = meta['date_start']
    n_days = meta['num_days']
    
    dataset_obj = ImageFolderDataset(path=dataset, use_labels=True)
    print(f'{len(dataset_obj)} frames in dataset {dataset_zipfile}')

    weights         = None
    input_seq_np    = None  # CHWT
    input_seq_times = []
    PAD             = 64

    slice_w         = slice(None if crop[0] == -1 else crop[0], None if crop[2] == -1 else crop[2])
    slice_h         = slice(None if crop[1] == -1 else crop[1], None if crop[3] == -1 else crop[3])
    
    # read input sequence
    # from datetime import datetime
    for idx, i in tqdm.tqdm(enumerate(range(startframe, startframe+frameskip*numframes, frameskip))):
        # read input image
        (img,t) = dataset_obj[i]

        img = img[:, slice_h, slice_w]

        # on first time, allocate data tensor (when we know the image size)
        if input_seq_np is None:
            input_seq_np = np.zeros(img.shape + (numframes,), dtype=np.uint8)

        input_seq_np[:, :, :, idx] = img

        # save timestamps
        input_seq_times.append(t.item())

        # ts = t_start + t.item() * n_days * 24 * 60 * 60
        # print(datetime.utcfromtimestamp(int(ts)).strftime(r'%d.%m.%Y %H:%M:%S'))    

    # falloff = lambda f: 1.0
    falloff = lambda f: 1.0 / (1.0 + 1000*np.abs(f))
    # falloff = lambda f: 1.0 / (1.0 + 10*np.abs(f))
    weights = init_weights(list(input_seq_np.shape[:3]) + [input_seq_np.shape[3] + 2*PAD], falloff=falloff)

    result_np = np.zeros_like(input_seq_np).astype(np.float32)

    # loop over spans of columns in input tensor
    for xstart in tqdm.tqdm(range(0, input_seq_np.shape[2], slice_size)):
        xend = min(xstart + slice_size, input_seq_np.shape[2])
        input_seq = totorch(input_seq_np[:, :, xstart:xend, :].astype(np.float32)/255.0)

        # allocate space for optimization variables
        optimization_variables = torch.clone(20*input_seq.detach()).requires_grad_(True)
        optim = torch.optim.Adam([optimization_variables], lr=1e-0, betas=(0.5, 0.5))

        if visualize:
            plt.ion()

        # optimize
        for i in tqdm.tqdm(range(inner_iterations)):
            x = blur_1perw(optimization_variables, weights, PAD)

            if data_loss == 'Huber':                
                loss = torch.nn.functional.smooth_l1_loss(x, input_seq, reduction='mean', beta=0.5)
            elif data_loss == 'L1':
                loss = (x - input_seq).abs().mean()
            elif data_loss == 'L2':
                loss = torch.nn.functional.mse_loss(x, input_seq) # L2

            if smoothness_loss == 'Huber':
                loss += smooth_alpha * torch.nn.functional.smooth_l1_loss(x[:, :, :, 1:], x[:, :, :, :-1], reduction='mean', beta=0.05)
            elif smoothness_loss == 'L1':
                loss += smooth_alpha * (x[:, :, :, 1:]-x[:, :, :, :-1]).abs().mean()
            elif smoothness_loss == 'L2':
                loss += smooth_alpha * torch.nn.functional.mse_loss(x[:, :, :, 1:], x[:, :, :, :-1])
          

            optim.zero_grad()
            loss.backward()
            optim.step()

            if visualize and (i+1) % 100 == 0 and i > 0:
                plt.clf()
                clip = lambda x: np.clip(x, 0.0, 1.0)
                # print(f'iter {i:05d} loss = {loss.item()}')
                plt.subplot(1, 2, 1)
                myimshow(clip(tonumpy(input_seq[:, :, 0])))
                plt.subplot(1, 2, 2)
                myimshow(clip(tonumpy(x[:, :, 0])))
                plt.pause(0.1)

        res = blur_1perw(optimization_variables, weights, PAD)
        result_np[:, :, xstart:xend, :] = res.detach().cpu().numpy()

        del loss
        del optim
        del optimization_variables

    # re-expose to maximum brightness 1
    scale = min(1.0, result_np.max())
    result_np = np.clip((result_np/scale)*255.0, 0.0, 255.0).astype(np.uint8)

    descr = f'{dataset_zipfile}-s{startframe}-n{numframes}-skip{frameskip}-ii{inner_iterations}-data{data_loss}-smooth{smoothness_loss}'
    descr += f'-crop{crop[0]}-{crop[1]}-{crop[2]}-{crop[3]}'

    np.save(os.path.join(outdir, descr+'.npy'), result_np)

    np.save(os.path.join(outdir, descr+'-timestamps.npy'), np.stack(input_seq_times))

    vfn = descr + '.mp4'
    vfn = os.path.join(outdir, vfn)
    print(f'Outputting video to {vfn}...')
    imageio_mp4_options = {'mode': 'I', 'codec': 'libx264', 'fps': 60, 'bitrate': '6M'}
    video = imageio.get_writer(vfn, **imageio_mp4_options)

    input_w     = input_seq_np.shape[2]
    input_h     = input_seq_np.shape[1]
    inset_w     = input_w//4
    inset_h     = input_h//4
    result_w    = input_w + ((16 - input_w & 15) & 15)
    result_h    = input_h + ((16 - input_h & 15) & 15)
    frame       = np.zeros((result_h, result_w, 3), dtype=np.uint8)

    for i in tqdm.tqdm(range(result_np.shape[3])):
        inset = PIL.Image.fromarray(input_seq_np[:, :, :, i].transpose(1, 2, 0)).\
                    resize((inset_w, inset_h), resample=PIL.Image.LANCZOS)
        frame[:input_h, :input_w, :] = result_np[:, :, :, i].transpose(1, 2, 0)
        frame[:inset_h, :inset_w, :] = inset

        video.append_data(frame)
    video.close()


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_martinbrualla15() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
