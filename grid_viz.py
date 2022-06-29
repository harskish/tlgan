# Copyright (c) 2022 Erik Härkönen, Aalto University
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

import os
import sys
import imgui
import torch
import glfw
import json
import time
import argparse
from tqdm import tqdm
from functools import lru_cache, partial
from multiprocessing import Lock
from dataclasses import dataclass
from copy import deepcopy
from enum import Enum
import numpy as np
from collections import defaultdict
from mutagen.mp4 import MP4
from PIL import Image
from PIL.PngImagePlugin import PngInfo, PngImageFile
from pathlib import Path
from importlib import import_module

from preproc.tools import StrictDataclass
piexif = import_module('piexif') # normal import breaks pylance

import pickle
from viewer.toolbar_viewer import ToolbarViewer
from viewer.utils import seeds_to_latents, combo_box_vals, stack_to_global_t, open_prog
from torch_utils import persistence
from ext import resize_right
from dataset_gan import DatasetGAN

args = None

# Visualize disentanglement properties along different 1d or 2d slices of the input space
# Interactive viewer for designing and exporting into paper

def file_drop_callback(window, paths, viewer):
    for p in paths:
        suff = Path(p).suffix.lower()
        if suff == '.pkl':
            viewer.state.pkl = p
        if suff in ['.jpg', '.png']:
            viewer.load_state_from_img(p)
        if suff == '.mp4':
            viewer.load_state_from_vid(p)

def get_meta_from_img(path: str):
    state_dump = r'{}'
    
    if path.endswith('.png'):
        test = PngImageFile(path)
        state_dump = test.text['description']
    elif path.endswith('.jpg'):
        exif_dict = piexif.load(Image.open(path).info["exif"])
        state_dump = exif_dict.get('0th', {}).get(piexif.ImageIFD.ImageDescription, r'{}')
    else:
        print(f'Unknown extension {Path(path).suffix}')

    return json.loads(state_dump)

@lru_cache
def get_strip_progression(N, factor=3, batch_mode=False):
    # Init: all ids map to width-1 strips
    ranges = {i : (i, i+1) for i in range(N)}
    if batch_mode:
        return (list(ranges.keys()), list(ranges.values()))

    # Compute strip sizes in progression
    # Largest strip size: divides image in multiple of F
    strip_size = int(np.ceil(N / factor))
    sz = []
    while strip_size > 1:
        sz.append(strip_size)
        strip_size = int(strip_size / factor)

    # Larger override smaller
    for L in sz[::-1]:
        starts = list(range(N))[::L]
        ends = starts[1:] + [N]
        ranges.update({s+(e-s)//2 : (s, e) for s,e in zip(starts, ends)})
    
    # Sort by strip size first, start idx second
    # Filter out strips outside of valid range
    # By construction: every idx in [0, N-1] should appear exactly once
    sort_ranges = lambda pairs : sorted(pairs, key=lambda tup: (-(tup[1][1] - tup[1][0]), tup[0]))
    pairs = sort_ranges([(i,(s,e)) for i,(s,e) in ranges.items() if i < N])
    
    # Make sure end result is same as if rendering with size 1 strips
    outputs = -1 * np.ones(N, dtype=np.int32)
    for i, (s, e) in pairs:
        outputs[s:e] = i
    diff = [i != v for i,v in enumerate(outputs)]
    if any(diff):
        for i in np.argwhere(diff).reshape(-1):
            pairs.append((i, (i, i+1)))
        pairs = sort_ranges(pairs)

    inds, ranges = zip(*pairs)
    return (inds, ranges)

# Up to 10 networks kept in RAM
@lru_cache(maxsize=10)
def get_G_CPU(path):
    with open_prog(path, 'rb') as f:
        data = pickle.load(f)
        return data['G_ema']

# Up to 3 networks kept in VRAM
@lru_cache(maxsize=3)
def get_G_real(path):
    return get_G_CPU(path).cuda()

@lru_cache(maxsize=3)
def get_DGAN(path):
    return DatasetGAN(path, args.dataset_root)

# Slices into 4D input space
# These can be used as axes for grid / video / stack
slices = {
    'z':        slice(0, 1),
    'w_lerp':   slice(0, 1),
    'c':        slice(1, 4),
    'lin+year': slice(1, 3),
    'year+day': slice(2, 4),
    'lin':      slice(1, 2),
    'year':     slice(2, 3),
    'day':      slice(3, 4),
}

rects = {
    'P70_sg2_valley1k_6M.pkl': [0, 128, 1024, 1024-128],
    'P50-sg2-mppv2-512-8M.pkl': [0, 66, 512, 512-66],
}

class GridViz(ToolbarViewer):    
    def __init__(self, name, batch_mode=False):
        self.batch_mode = batch_mode
        self.G_lock = Lock()
        super().__init__(name, batch_mode=batch_mode)
    
    def setup_callbacks(self, window):
        glfw.set_drop_callback(window,
            partial(file_drop_callback, viewer=self))

    def setup_state(self):
        self.state = UIState()
        self.state_soft = UIStateSoft()
        
        if Path(args.input).is_file():
            file_drop_callback(None, [args.input], self)
        elif args.input.startswith('{') and args.input.endswith('}'):
            # Parse as UI state json dump
            self.load_state_from_dict(json.loads(args.input))
        else:
            print(f'Unknown input arg: {args.input}')
        
        self.rend = RendererState()
        self.json_editor_open = False # for exporting/importing UI state as json
        self.json_editor_text = ''
        self.json_editor_multiline = True
        self.video_playing = False

    @property
    def G(self):
        return self.get_G(self.state.pkl)

    def get_G(self, pkl):
        try:
            self.G_lock.acquire()
            if self.state.show_dataset:
                return get_DGAN(pkl)
            else:
                return get_G_real(pkl)
        finally:
            self.G_lock.release()

    def get_out_rect(self, G):
        res = G.img_resolution
        name = os.path.basename(self.state.pkl)
        backup = rects.get(name, [0, 0, res, res])
        return getattr(G.synthesis, 'out_rect', backup)

    def t_to_date(self, t):
        days = self.G.cond_args.get('num_days', 1)
        ts0 = self.G.cond_args.get('start_ts', 946684800) # 1.1.2000
        end = ts0 + t * days * 24 * 60 * 60
        
        from datetime import datetime
        return datetime.utcfromtimestamp(int(end)).strftime(r'%d.%m.%Y %H:%M:%S')

    def export_video(self, state, len_s=10, bitrate='25M'):
        import imageio

        gname = Path(state.pkl).with_suffix('').name
        tag = 'dset' if state.show_dataset else 'gan'
        outdir = Path('.')

        # Don't overwrite existing
        seed_str = "_".join(state.seeds.split(","))
        pattern = f'{state.viz_mode.name.lower()}_{gname}_{state.axis_x}_{state.axis_y}_{seed_str}_{tag}'
        idx = len(list(outdir.glob(f'{pattern}_*.jpg')))
        fname = f'{pattern}_{idx}.mp4'

        indices = list(range(self.state_soft.video_i, self.state_soft.video_i + self.rend.num_frames))
        indices = [i % self.rend.num_frames for i in indices]

        video_kwargs = dict(bitrate=bitrate)
        video_out = imageio.get_writer(fname, mode='I', fps=len(indices)/len_s, codec='libx264', **video_kwargs)

        # Ensure divisiblility by macro block
        blk = 16
        h, w = self.rend.img_shape
        pad_h, clip_w = (blk - (h % blk), w % blk)
        pad_h_half, clip_w_half = (pad_h // 2, clip_w // 2)

        for i in tqdm(range(len(indices))):
            frame = self.get_video_frame(indices[i], self.state_soft.cycles, 'cpu')
            if clip_w > 0:
                frame = frame[:, clip_w_half:clip_w-clip_w_half, :]
            if 0 < pad_h < blk:
                frame = torch.nn.functional.pad(frame, (0, 0, 0, 0, pad_h_half, pad_h-pad_h_half))
            video_out.append_data(frame.numpy()) # hwc of uint8
        video_out.close()
        
        # Add metadata
        file = MP4(fname)
        file['desc'] = json.dumps(self.ui_state_dict, sort_keys=True)
        file.pprint()
        file.save()

        print('Saved as', fname)
    
    # In: hwc
    def export_img(self, grid, state, path=None, ext='png'): # jpg, png
        im = Image.fromarray(np.uint8(grid.clip(0,255).cpu().numpy()))
        metadata = json.dumps(self.ui_state_dict, sort_keys=True)
        
        gname = Path(state.pkl).with_suffix('').name
        tag = 'dset' if state.show_dataset else 'gan'
        outdir = Path('.')

        if path is not None:
            fname = Path(path).name
            outdir = Path(path).parent
            os.makedirs(outdir, exist_ok=True)
        else:
            # Don't overwrite existing
            seed_str = "_".join(state.seeds.split(","))
            pattern = f'{state.viz_mode.name.lower()}_{gname}_{state.axis_x}_{state.axis_y}_{seed_str}_{tag}'
            idx = len(list(outdir.glob(f'{pattern}_*.{ext}')))
            fname = f'{pattern}_{idx}.{ext}'

        if ext.lower() == 'jpg':
            exif_dict = defaultdict(dict)
            exif_dict['0th'][piexif.ImageIFD.ImageDescription] = metadata
            exif_bytes = piexif.dump(exif_dict)
            im.save(outdir / fname, format='jpeg', quality=98, exif=exif_bytes) # max reasonable quality
        elif ext.lower() == 'png':
            chunk = PngInfo()
            chunk.add_text('description', metadata)
            opt = np.prod(grid.shape[:2]) < 2_000_000
            im.save(outdir / fname, format='png', optimize=opt, compress_level=9, pnginfo=chunk) # max compression
        else:
            raise RuntimeError(f'Unknown image extension {ext}')
        
        print('Saved as', fname)

    def draw_date_selector(self):
        s = self.state
        ch0, t0 = imgui.slider_float('Lin',  s.sliders[0],  0, 1) # absolute
        ch1, t1 = imgui.slider_float('Year', s.sliders[1], -1, 1) # relative wrt. t
        ch2, t2 = imgui.slider_float('Day',  s.sliders[2], -1, 1) # relative wrt. t
        s.sliders = (t0, t1, t2)

        if not s.pkl:
            return

        t1 = np.clip(t1, -0.9999, 0.9999) # strange jumps at exactly +-1 whole cycle
        t2 = np.clip(t2, -0.9999, 0.9999)

        fs = self.G.cond_xform.get_frequencies()
        fs = fs if len(fs) else [-1] # cond_xform not used => assume single global cond
        ts = [t0, t1/fs[1], t2/fs[2]] if len(fs) == 3 else [t0, t2/fs[1]] if len(fs) == 2 else [t0]
        s.t = stack_to_global_t(np.array(ts), np.array(fs)).item()

        # Update str in ui
        s.date_str = self.t_to_date(s.t)

    # Draws video frame selector below output image
    def draw_output_extra(self):
        self.state_soft.video_i = imgui.slider_int('', self.state_soft.video_i, 0, self.rend.num_frames - 1)[1]

        imgui.same_line()
        if self.video_playing and imgui.button('||'):
            self.video_playing = False
        if not self.video_playing and imgui.button('|>'):
            self.video_playing = True
        
        if self.video_playing:
            t_norm = (int(time.time() * 1_000) % 45_000) / 45_000 # in [0, 1], repeats every 45s
            self.state_soft.video_i = int(t_norm * (self.rend.num_frames - 1))

    @property
    def seeds(self):
        state = self.rend.last_ui_state
        return [int(seed) for seed in state.seeds.split(',') if seed != ''] # hand-picked

    def init_G_inputs(self, s, W, H):
        input = torch.zeros((H, W, 4), dtype=torch.float32) # z,c0,c1,c2
        input[:, :, slices['c']] = s.t  # used as-is unless one of axes
        input[:, :, slices['z']] = float(self.seeds[0])
        fs = self.get_G(s.pkl).cond_xform.get_frequencies()
        
        # 2D subspace replaced by linspace
        # If single row/col: show midpoint (s.t)
        for axis, shape in zip([s.axis_x, s.axis_y], [(1, W, 1), (H, 1, 1)]):
            N = np.prod(shape)
            if N <= 1:
                continue
            
            lsp_norm = torch.linspace(s.t_range[0], s.t_range[1], N)

            lsp = None
            if axis == 'day':
                # cyclic ranges: equally spaced endpoints (mod f)
                lsp = s.t + (1/fs[-1]) * (lsp_norm - 0.5) * ((N-1) / N)
            elif axis in ['year', 'year+day']:
                # cyclic ranges: equally spaced endpoints (mod f)
                lsp = s.t + (1/fs[1]) * (lsp_norm - 0.5) * ((N-1) / N)
            elif axis == 'z':
                seeds = self.seeds
                pad = [max(seeds) + s + 1 for s in range(N)] # pad end with consecutive
                lsp = torch.tensor((seeds + pad)[:N], dtype=torch.float32)
            else:
                # whole c or linear part or slerp weights
                lsp = lsp_norm
            
            input[:, :, slices[axis]] = lsp.view(*shape)

        # Force separate seed for every input
        if s.rand_seed == 'all':
            input[:, :, slices['z']] = float(self.seeds[0]) + torch.linspace(0, H*W - 1, H*W).view(H, W, -1)
        elif s.rand_seed == 'per row':
            input[:, :, slices['z']] = float(self.seeds[0]) + torch.linspace(0, H - 1, H).view(H, 1, -1)
        elif s.rand_seed == 'per col':
            input[:, :, slices['z']] = float(self.seeds[0]) + torch.linspace(0, W - 1, W).view(1, W, -1)

        return input

    # Run G on normalized (float32 in [0,1]) inputs,
    # which are mapped back to relevant ranges and dtypes
    def run_g_normalized(self, G, inputs, s):
        assert inputs.ndim == 2 and inputs.shape[1] == 4, \
            'Expected inputs of shape [B, 4]'
        
        # Only two frequencies => assume year is missing
        if G.cond_xform.num_f == 2:
            r = slices['year']
            inputs = torch.cat([inputs[:, 0:r.start], inputs[:, r.stop:]], dim=-1)

        # Only global trend
        if G.cond_xform.num_f < 2:
            inputs = torch.cat([inputs[:, 0:slices['year'].start], inputs[:, slices['day'].stop:]], dim=-1)

        ts = inputs[:, slices['c']].cuda()
        cs = G.cond_xform(ts)
        
        ws = None
        if 'w_lerp' in [s.axis_x, s.axis_y]:
            seeds = self.seeds + [1_000]
            zs = torch.from_numpy(seeds_to_latents(seeds[0:2], n_dims=G.z_dim)).cuda() # two endpoints
            ws = G.mapping(zs, cs[:2, 0, :], truncation_psi=s.trunc, truncation_cutoff=G.num_ws-s.trunc_cutoff)
            w1, w2 = ws.unsqueeze(1).unbind(0) # (1, 16, 512)
            weights = inputs[:, slices['z']].view(-1, 1, 1).cuda() # in [0, 1]
            ws = (1 - weights)*w1 + weights*w2 # lerp
        else:
            seeds = inputs[:, slices['z']].int().view(-1)
            zs = torch.from_numpy(seeds_to_latents(seeds, n_dims=G.z_dim)).cuda()
            ws = G.mapping(zs, cs[:, 0, :], truncation_psi=s.trunc, truncation_cutoff=G.num_ws-s.trunc_cutoff)
        
        c_dim = getattr(G.synthesis, 'c_dim', G.cond_args.dims)
        out_tensor = G.synthesis(ws, cs[:, :, 0:c_dim], noise_mode=s.spatial_noise)

        return out_tensor

    # Large grids updated iteratively in spiral order
    def get_spiral(self, W, H):
        cx, cy = (int(W // 2), int(H // 2))
        coords = [(cx, cy)]
        
        # 1. Assume square-shaped image of odd size
        for r in range(max(cx, cy)):
            # Move to top-left corner
            cx -= 1; cy -= 1
            
            # Draw 4 lines
            for (dx, dy) in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                for _ in range(2*(r+1)):
                    cx += dx; cy += dy
                    coords.append((cx, cy))

        # 2. Filter out invalid coords
        return [(x,y) for x,y in coords if (0 <= x < W) and (0 <= y < H)]

    # Compute scaled output size based on window size
    def scaled_output_size(self, s):
        x1, y1, x2, y2 = self.get_out_rect(self.G)

        # AA params
        oW, oH = self.content_size
        iW, iH = (s.W*(x2-x1), s.H*(y2-y1))
        scale = min(oW/iW, oH/iH)*s.res_scale
        out_shape = (int(y2-y1), int(x2-x1))
        if s.use_AA and (0 < scale < 1) and not self.batch_mode:
            out_shape = (int(scale*(y2-y1)), int(scale*(x2-x1)))

        return out_shape

    def get_video_frame(self, i, cycles=1, device='cuda'):
        if self.rend.video_frames is None:
            return None
        
        # Normal video frame
        if self.state_soft.video_mode == VideoMode.FULL:
            idx = (i * cycles) % self.rend.num_frames
            return self.rend.video_frames[idx].permute(1, 2, 0).to(device=device)
        
        # Single row or column
        if self.state_soft.video_mode == VideoMode.SINGLE:
            return self._get_frame_stripes(i, cycles, self.state_soft.num_strips, device)

        # Strided strips ("time-lapse image")
        if self.state_soft.num_strips > 0:
            return self._get_frame_strips(i, cycles, self.state_soft.num_strips, device)
        else:
            return self._get_frame_strided(i, cycles, device)

    def _get_frame_stripes(self, i, cycles, strips, device):
        idx = i % self.rend.num_frames
        _, _, H, W = self.rend.video_frames.shape
        horiz = (self.rend.last_ui_state.viz_mode == VizMode.STACK_X)

        full_img = None
        if horiz:
            full_img = self.rend.video_frames[:W, :, :, idx].permute(2, 0, 1) # wch to hwc
        else:
            full_img = self.rend.video_frames[:H, :, idx, :].permute(0, 2, 1) # hcw to hwc

        if strips <= 1:
            return full_img.to(device=device)

        # Coordinates along time axis (frame dimension) for each strip
        n = W if horiz else H
        strip_sz = n // strips
        left_edges = cycles * n * np.linspace(0, strips - 1, strips) / strips
        centers = [int(i + d + 0.5*strip_sz) % n for d in left_edges]

        # X/Y coords of strips
        strip_starts = np.linspace(0, n, strips + 1, dtype=np.int32)
        
        parts = []
        for (idx, s, e) in zip(centers, strip_starts[:-1], strip_starts[1:]):
            single = full_img[:, idx:idx+1, :] if horiz else full_img[idx:idx+1, :, :] # hwc in
            parts.append(torch.repeat_interleave(single, e-s, dim=1 if horiz else 0))

        frame = torch.cat(parts, dim=1 if horiz else 0)
        
        return frame.to(device=device)
        
        

    # Larger strips
    def _get_frame_strips(self, i, cycles, strips, device):
        horiz = (self.rend.last_ui_state.viz_mode == VizMode.STACK_X)
        h, w = self.rend.img_shape
        n = w if horiz else h
        strip_sz = n // strips

        parts = []
        left_edges = cycles * n * np.linspace(0, strips - 1, strips) / strips
        centers = [int(i + d + 0.5*strip_sz) % n for d in left_edges]

        strip_starts = np.linspace(0, n, strips + 1, dtype=np.int32)

        for (idx, s, e) in zip(centers, strip_starts[:-1], strip_starts[1:]):
            if horiz:
                parts.append(self.rend.video_frames[idx, :, :, s:e])
            else:
                parts.append(self.rend.video_frames[idx, :, s:e, :])

        frame = torch.cat(parts, dim=-1 if horiz else -2)
        
        return frame.permute(1, 2, 0).to(device=device) # to hwc

    # One strip per row/column of image
    def _get_frame_strided(self, i, cycles, device):
        horiz = (self.rend.last_ui_state.viz_mode == VizMode.STACK_X)
        _, c, h, w = self.rend.video_frames.shape
        n = w if horiz else h
        frame_stride = c*h*w*cycles # n cycles: 'move n times faster in time'
        
        stride = None
        if horiz:
            stride = (w*h, w, 1+frame_stride) # color_ch: jump by pixel count, row: jump by w, col: read from next frame
        else:
            stride = (w*h, w+frame_stride, 1) # color_ch: jump by pixel count, row: read from next frame, col: contiguous

        #  Frames wrap around => read twice
        #
        #     spatial col
        #     ----------
        #  f  | | | |4| |  
        #  r  | | | | |5|  1-3: read from view1 
        #  a  |1| | | | |  4-5: read from view2
        #  m  | |2| | | |
        #  e  | | |3| | |
        #     ----------

        parts = []

        if cycles > 1:
            i = 0
        
        for ccl in range(cycles):
            spatial_offset = ccl * (n // cycles) * (1 if horiz else w)
            
            H = n - i if not horiz else h
            W = n - i if horiz else w
            outshape = (c, H, W//cycles) if horiz else (c, H//cycles, W)
            out_view1 = self.rend.video_frames.as_strided(outshape, stride, storage_offset=i*frame_stride+spatial_offset) # i frames forward, 0 rows/cols forward
            parts.append(out_view1)
            
            if i > 0:
                H = i if not horiz else h
                W = i if horiz else w
                outshape = (c, H, W//cycles) if horiz else (c, H//cycles, W)
                out_view2 = self.rend.video_frames.as_strided(outshape, stride, storage_offset=n-i if horiz else (n-i)*w)   # 0 frames forward, i rows/cols forward
                parts.append(out_view2)

        frame = torch.cat(parts, dim=-1 if horiz else -2)
        
        return frame.permute(1, 2, 0).to(device=device) # to hwc

    def compute_stack(self, s, horiz=True):
        # Detect changes
        if self.rend.last_ui_state != s:
            self.rend.last_ui_state = s
            self.rend.out_rect = self.get_out_rect(self.G)
            x1, y1, x2, y2 = self.rend.out_rect
            W, H = (x2-x1, y2-y1)
            self.rend.img_shape = (H, W)
            self.rend.input = self.init_G_inputs(s, W if horiz else 1, 1 if horiz else H) # [H,W,4]
            self.rend.factor = max(2, s.B)
            order, self.rend.sizes = get_strip_progression(W if horiz else H, self.rend.factor, self.batch_mode)
            self.rend.order = [(i, 0) for i in order] if horiz else [(0, i) for i in order] # (x,y)
            self.rend.output = torch.zeros((3, *self.rend.img_shape[0:2]), dtype=torch.uint8).cuda()
            self.rend.i = 0
            # 1024px: 1B*3*1024^3 = 3*2^30B = 3.0 GiB
            self.rend.num_frames = W if horiz else H
            self.state_soft.video_i = min(self.rend.num_frames - 1, self.state_soft.video_i)
            alloc = (max(W, H), 4, H, W)
            if self.rend.video_frames is None or self.rend.video_frames.shape != alloc:
                self.rend.video_frames = torch.empty(alloc, dtype=torch.uint8, device='cuda')

        # Check if work is done
        i = self.rend.i
        if i >= len(self.rend.order):
            return self.get_video_frame(self.state_soft.video_i, self.state_soft.cycles)

        # First iteration: round down to multiple of factor
        B = max(1, (s.B // self.rend.factor) * self.rend.factor) if i == 0 else s.B

        # Run single step forward
        coords = self.rend.order[i:i+B]
        sizes = self.rend.sizes[i:i+B]
        invars = torch.cat([self.rend.input[y, x].view(1, -1) for x,y in coords])
        img_batch = 0.5*(1 + self.run_g_normalized(self.get_G(s.pkl), invars, s))
        img_bytes = (255 * img_batch.clip(0, 1)).byte()
        
        # Crop borders
        x1, y1, x2, y2 = self.rend.out_rect
        img_bytes = img_bytes[:, :, y1:y2, x1:x2].unsqueeze(1)

        # Save to stack
        for img, (x, y), (s, e) in zip(img_bytes, coords, sizes):
            alpha = 255*torch.ones((1, 4-img.shape[1], *img.shape[2:]), dtype=torch.uint8, device=img.device) # 1CHW
            self.rend.video_frames[s:e, :, :, :] = torch.cat((img, alpha), dim=1) # sequential writes

        # Move on to next batch
        self.rend.i += B

        # Draw line to track progress
        frame = self.get_video_frame(self.state_soft.video_i, self.state_soft.cycles) # HWC
        with_stripe = frame.clone()
        if self.rend.i < len(self.rend.order):
            color = torch.tensor([61, 133, 224, 255], dtype=torch.uint8).view(1, 1, -1)
            if horiz:
                c = (x - self.state_soft.video_i) % (x2-x1)
                with_stripe[:, c-1:c+1, :] = color
            else:
                c = (y - self.state_soft.video_i) % (y2-y1)
                with_stripe[c-1:c+1, :, :] = color

        return with_stripe

    def compute_grid(self, s):
        # Detect changes
        # Only works for fields annotated with type (e.g. sliders: list)
        if self.rend.last_ui_state != s:
            self.rend.last_ui_state = s
            self.rend.input = self.init_G_inputs(s, s.W, s.H)
            self.rend.order = self.get_spiral(s.W, s.H)
            self.rend.img_shape = self.scaled_output_size(s)
            self.rend.output = torch.zeros((4, self.rend.img_shape[0]*s.H, self.rend.img_shape[1]*s.W), dtype=torch.uint8).cuda()
            self.rend.i = 0

        # Check if work is done
        if self.rend.i >= len(self.rend.order):
            return None

        # Run single step forward
        coords = self.rend.order[self.rend.i:self.rend.i+s.B]
        invars = torch.cat([self.rend.input[y, x].view(1, -1) for x,y in coords])
        img_batch = 0.5*(1 + self.run_g_normalized(self.G, invars, s))
        
        # Crop borders
        x1, y1, x2, y2 = self.get_out_rect(self.G)
        img_batch = img_batch[:, :, y1:y2, x1:x2]

        # Downscale (no-op if shapes match)
        img_batch = resize_right.resize(img_batch, out_shape=self.rend.img_shape,
            interp_method=resize_right.interp_methods.lanczos3, antialiasing=True)
        
        # Save to grid
        img_bytes = (255 * img_batch.clip(0, 1)).byte()
        for img, (x, y) in zip(img_bytes, coords):
            H, W = self.rend.img_shape
            alpha = 255*torch.ones((4-img.shape[0], H, W), dtype=torch.uint8, device=img.device)
            self.rend.output[:, y*H:(y+1)*H, x*W:(x+1)*W] = torch.cat((img, alpha), dim=0)

        # Move on to next batch
        self.rend.i += s.B
        
        # Output updated grid
        return self.apply_crop(self.rend.output).permute(1, 2, 0)

    def compute(self):
        if self.state.pkl is None:
            return None

        # Export
        if self.state.export_vid:
            self.export_video(self.rend.last_ui_state)
            self.state.export_vid = False
        
        if self.state.export_img:
            if self.state.viz_mode == VizMode.GRID:
                self.export_img(self.apply_crop(self.rend.output).permute(1, 2, 0), self.rend.last_ui_state)
            else:
                self.export_img(self.get_video_frame(self.state_soft.video_i, self.state_soft.cycles), self.rend.last_ui_state)
            self.state.export_img = False

        # Set tolerance for dataset frame matching
        if self.state.show_dataset:
            self.G.synthesis.tol_hours = self.state.gt_tol_hours
            self.G.synthesis.add_noise = self.state.dataset_noise

        # Copy for this frame
        s = deepcopy(self.state)

        if s.viz_mode == VizMode.GRID:
            return self.compute_grid(s)
        elif s.viz_mode == VizMode.STACK_X:
            return self.compute_stack(s, horiz=True)
        elif s.viz_mode == VizMode.STACK_Y:
            return self.compute_stack(s, horiz=False)
        else:
            return None

    # In/out: CHW
    def apply_crop(self, grid):
        H, W = grid.shape[1:]
        iH, iW = self.rend.img_shape
        L, R = [c*iW for c in self.state.crop_hor] # measured in #images
        T, B = [c*iH for c in self.state.crop_ver]
        grid = grid[:, T:H-B, :]
        grid = grid[:, :, L:W-R]
        return grid

    def print_inputs(self):
        input = self.rend.input
        fs = self.G.cond_xform.get_frequencies()
        fs = fs if len(fs) else [-1] # cond_xform not used => assume single global cond
        
        rows, cols, _ = input.shape
        strs = []
        for r in range(rows):
            for c in range(cols):
                ts = [input[r, c, slices['lin']]]
                if len(fs) > 2:
                    ts.append(input[r, c, slices['year']])
                if len(fs) > 1:
                    ts.append(input[r, c, slices['day']])

                ts = torch.stack(ts).view(-1).cpu().numpy()
                t_glob = stack_to_global_t(ts, np.array(fs)).item()
                date = self.t_to_date(t_glob)
                seed = input[r, c, slices['z']].int().cpu().numpy().item()
                strs.append(f'{c}, {r}: seed={seed:05d}, t={t_glob:.10f}, date={date}')
        
        print('\n'.join(strs) + '\n')
    
    def draw_toolbar(self):
        s = self.state
        imgui.text(f'PKL: {os.path.basename(s.pkl or "")}')
        
        # Seed selector
        s.seeds = imgui.input_text('Seeds', s.seeds, 512)[1]
        s.seeds = ''.join(c for c in s.seeds if c in '0123456789,') or '0' # enforce valid non-empty
        seeds = [s for s in s.seeds.split(',') if s != '']
        last = int(seeds[-1])
        imgui.same_line()
        if imgui.button('-##seed_prev'):
            s.seeds = ','.join(seeds[:-1] + [str(max(0, last - 1))])
        imgui.same_line()
        if imgui.button('+##seed_next'):
            s.seeds = ','.join(seeds[:-1] + [str(last + 1)])

        # Viz mode selection
        s.viz_mode = combo_box_vals('Mode', [m for m in VizMode], s.viz_mode, to_str=lambda v: v.name)[1]

        # Grid dims and axes
        s.W = imgui.slider_int('W', s.W, 1, 20)[1]
        s.H = imgui.slider_int('H', s.H, 1, 20)[1]
        s.B = imgui.slider_int('B', s.B, 1, 8)[1]
        s.crop_hor = tuple(imgui.slider_int2('Crop hor.', *s.crop_hor, 0, int(s.W//2))[1])
        s.crop_ver = tuple(imgui.slider_int2('Crop ver.', *s.crop_ver, 0, int(s.H//2))[1])
        self.draw_date_selector()
        s.t_range = tuple(imgui.slider_float2('T range', *s.t_range, 0, 1)[1])
        imgui.text(f'Date: {s.date_str}')
        s.trunc = imgui.slider_float('trunc', s.trunc, 0, 1)[1]
        s.axis_x = combo_box_vals('X-axis', list(slices.keys()), s.axis_x)[1]
        s.axis_y = combo_box_vals('Y-axis', list(slices.keys()), s.axis_y)[1]
        s.rand_seed = combo_box_vals('Rand seed', ['fixed', 'per row', 'per col', 'all'], s.rand_seed)[1]
        s.spatial_noise = combo_box_vals('Spatial noise', ['random', 'const'], s.spatial_noise)[1]
        self.state_soft.cycles = imgui.slider_int('Cycles', self.state_soft.cycles, 1, 5)[1]
        self.state_soft.num_strips = imgui.slider_int('Strips', self.state_soft.num_strips, 0, 50)[1]
        s.res_scale = imgui.slider_float('Scale', s.res_scale, 0.1, 10.0)[1]
        s.use_AA = imgui.checkbox('Anti-alias', s.use_AA)[1]
        s.show_dataset = imgui.checkbox('Visualize dataset', s.show_dataset)[1]
        s.dataset_noise = imgui.checkbox('Dataset noise', s.dataset_noise)[1]
        s.gt_tol_hours = imgui.slider_float('GT tol. (h)', s.gt_tol_hours, 0.1, 24.0)[1]

        self.state_soft.video_mode = combo_box_vals('Video type',
            [m for m in VideoMode], self.state_soft.video_mode, to_str=lambda v: v.name)[1]
        
        # Show video export button if applicable
        if s.viz_mode != VizMode.GRID:
            if s.export_vid:
                imgui.text('Exporting...')
            elif imgui.button('Export video'):
                s.export_vid = True
            imgui.same_line(); imgui.text('{}x{}x{}'.format(self.rend.num_frames, *self.img_shape[::-1][:2]))
        
        # Print exact inputs for images in grid
        if imgui.button('Print inputs'):
            self.print_inputs()

        # Always show image export button
        if s.export_img:
            imgui.text('Exporting...')
        elif imgui.button('Export image'):
            s.export_img = True
        imgui.same_line(); imgui.text('{}x{}'.format(*self.img_shape[::-1][:2]))

        if imgui.button('JSON'):
            self.json_editor_text = json.dumps(self.ui_state_dict, sort_keys=True, indent=4)
            self.json_editor_open = True

        # Export / import state as json
        if self.json_editor_open:
            self.draw_json_editor()

    @property
    def ui_state_dict(self):
        from dataclasses import asdict
        return {
            'state': asdict(self.state),
            'state_soft': asdict(self.state_soft)
        }

    def load_state_from_dict(self, state_dict):
        state_dict_soft = state_dict['state_soft']
        state_dict = state_dict['state']
        
        # Ignore certain values
        ignores = ['export', 'export_vid', 'export_img']
        state_dict = { k: v for k,v in state_dict.items() if k not in ignores }
        state_dict_soft = { k: v for k,v in state_dict_soft.items() if k not in ignores }

        # Convert from int to enum
        if 'viz_mode' in state_dict:
            state_dict['viz_mode'] = VizMode(state_dict['viz_mode'])
        if 'video_mode' in state_dict_soft:
            state_dict_soft['video_mode'] = VideoMode(state_dict_soft['video_mode'])

        # Single seed (old) to seed list
        if 'seed1' in state_dict:
            state_dict['seeds'] = str(state_dict['seed1'])
            del state_dict['seed1']
            del state_dict['seed2']
        
        # Check that pickle exists
        pkl = state_dict.get('pkl')
        if pkl and not Path(pkl).is_file():
            print(f'PKL not found: {pkl}')
            del state_dict['pkl']

        # Volatile state
        for k, v in state_dict.items():
            setattr(self.state, k, v)

        # Non-volatile state
        for k, v in state_dict_soft.items():
            setattr(self.state_soft, k, v)

    # Read UI state from exif data
    def load_state_from_img(self, path: str):
        self.load_state_from_dict(get_meta_from_img(path))

    # Read UI state from mp4 metadata
    def load_state_from_vid(self, path):
        file = MP4(path)
        if 'desc' in file:
            state_dump = file['desc'][0]
            self.load_state_from_dict(json.loads(state_dump))
        else:
            print(f'No metadata in file {path}')

    def draw_json_editor(self):
        imgui.begin('JSON editor')

        imgui.set_window_focus()

        # Check validity
        valid = True
        msg = 'Save'
        color = (255, 255, 255)
        json_obj = None
        try:
            json_obj = json.loads(self.json_editor_text)
        except json.JSONDecodeError:
            valid = False
            msg = 'Error'
            color = (255, 0, 0)

        # Switch to single-line output for copying to clipboard etc.
        if imgui.radio_button('Multiline', self.json_editor_multiline):
            if valid:
                self.json_editor_multiline = not self.json_editor_multiline
                self.json_editor_text = json.dumps(json_obj,
                    indent=(4 if self.json_editor_multiline else None))
        
        W, H = imgui.get_window_size()
        if self.json_editor_multiline:
            _, self.json_editor_text = imgui.input_text_multiline('', self.json_editor_text, 4096, width=W-30, height=H-100*self.ui_scale)
        else:
            _, self.json_editor_text = imgui.input_text('', self.json_editor_text, 4096)

        imgui.push_style_color(imgui.COLOR_TEXT, *color)
        if imgui.button(f'{msg}##save_json_button') and valid:
            self.load_state_from_dict(json_obj)
            self.json_editor_open = False

        imgui.pop_style_color()

        if imgui.button('Close'):
            self.json_editor_open = False
        imgui.end()

class VizMode(int, Enum):
    GRID = 0    # grid of images
    STACK_X = 1 # x-axis stacked horizontally
    STACK_Y = 2 # y-axis stacked vertically

class VideoMode(int, Enum):
    FULL = 0   # full frames => normal video
    STACK = 1  # strided frames => stack
    SINGLE = 2 # single rows/cols => stripes

# Volatile state: requires recomputation of results
@dataclass
class UIState(StrictDataclass):
    pkl: str = None               # Path to pickle (can be hot-swapped)
    seeds: str = '0'              # Seeds to use, comma-separated list
    trunc: float = 1.0            # Z truncation and layer cutoff
    trunc_cutoff: int = 0
    axis_x: str = 'day'           # Input variables (axes) to vary
    axis_y: str = 'year'
    W: int = 1                    # Grid dimensions
    H: int = 1
    B: int = 5                    # Batch size
    t: float = 0.5                # Parsed from sliders
    t_range: tuple = (0.0, 1.0)   # Range of ts for inputs
    date_str: str = ''
    sliders: tuple = (0.5, 0, 0)  # Overrides for (lin, year, day)
    rand_seed: str = 'Fixed'      # Randomize seed within grid / stack?
    use_AA: bool = True           # Anti-alias images?
    res_scale: float = 1.0        # Scale output resolution
    export_vid: bool = False      # Signal video export
    export_img: bool = False      # Signal image export
    show_dataset: bool = False    # Show dataset frames instead of GAN output
    dataset_noise: bool = False   # Add noise to dataset frames
    spatial_noise: str = 'const'  # Spatial noise maps
    gt_tol_hours: float = 1.5     # When to consider dataset frame missing
    crop_hor: tuple = (0, 0)      # Crop resulting grid
    crop_ver: tuple = (0, 0)
    viz_mode: VizMode = VizMode.GRID

# "Soft" state: does not require recomputation
@dataclass
class UIStateSoft(StrictDataclass):
    video_i: int = 0
    cycles: int = 1               # How many cycles to show in stack
    num_strips: int = 0           # Stack mode: how many strips to show (zero: w/h)
    video_mode: VideoMode = VideoMode.STACK

@dataclass
class RendererState(StrictDataclass):
    last_ui_state: UIState = None # Detect changes in UI, restart rendering
    img_shape: tuple = (0, 0)
    out_rect: list = None
    input: torch.Tensor = None    # 4D input vector per image in grid
    output: torch.Tensor = None   # Frames completed so far
    output_sanity : np.ndarray = None
    order: list = None            # Order in which to render grid images (e.g. spiral)
    sizes: list = None            # Stack viz: width/height of slices (progressive refinenemt)
    factor: int = 2               # Stack viz: branching factor
    i: int = 0                    # Current index into order array
    video_frames: torch.Tensor = None
    num_frames: int = 0

def init_torch():
    # Go fast
    torch.autograd.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True
    
    # Stay safe
    os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

# Run in batch mode: generate images from state dicts
def run_batch(state_dicts, out_paths, progress_bar=False):
    init_torch()
    viewer = GridViz('grid_viz_viewer', batch_mode=True)
    
    for d, p in zip(state_dicts, out_paths):
        viewer.load_state_from_dict(d)
        if 'pkl' in d:
            assert viewer.state.pkl == d['pkl'], f'Could not load pickle {d["pkl"]}'

        # Init state
        viewer.compute()
        prog = None if not progress_bar else tqdm(total=len(viewer.rend.order))
        
        while viewer.rend.i < len(viewer.rend.order):
            new = viewer.compute()
            if progress_bar:
                prog.update(viewer.state.B)
            if new is None:
                break
        
        img = None
        if viewer.state.viz_mode == VizMode.GRID:
            img = viewer.apply_crop(viewer.rend.output).permute(1, 2, 0)
        else:
            img = viewer.get_video_frame(viewer.state_soft.video_i, viewer.state_soft.cycles)
        
        viewer.export_img(img, viewer.rend.last_ui_state, path=p, ext='png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TLGAN grid visualizer')
    parser.add_argument('input', type=str, help='Model pickle, image, video, or state json')
    parser.add_argument('--dataset_root', type=str, default=os.environ.get('TLGAN_DATASET_ROOT', str(Path(__file__).parent)), help='Path to datasets (for showing GT frames)')
    args = parser.parse_args()

    init_torch()
    viewer = GridViz('grid_viz_viewer')
    print('Done')
