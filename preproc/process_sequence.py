# Copyright (c) 2022 Erik Härkönen, Aalto University
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

"""Preprocess time-lapse sequence into dataset for training"""

from asyncore import write
from copy import deepcopy
from dataclasses import dataclass
import zipimport
import numpy as np
import argparse
from PIL import Image
from os import makedirs
from tqdm import trange
from pathlib import Path
import json
import imgui
import glfw
import functools
import torch
import kornia
from tools import VideoFrames, StrictDataclass
from datetime import datetime, timezone
from affine import PfromWarpMatrix, WarpMatrix, AffineDecomp, DecompToAffine
import cv2 # pip install opencv_python==4.4.0.46
import os

import sys
sys.path += [str(Path(__file__).parents[1])]
from ext import resize_right
from viewer.utils import combo_box_vals, with_state
from dnnlib import EasyDict
from viewer.toolbar_viewer import ToolbarViewer

@functools.lru_cache(maxsize=5)
def get_frame_tensor_cached(i):
    img = vid[i]
    if not torch.is_tensor(img):
        img = torch.tensor(img, device='cuda')
    return img.permute((2, 0, 1)).unsqueeze(0) # => NCHW

# AMOS-format assumed
def try_parse_filenames(v, idx_s, idx_e):
    formats = [
        r'%Y%m%d_%H%M%S',     # 20100217_030236.jpg
        r'%Y-%m-%d-%H-%M-%S', # 2019-05-21-15-03-42.jpg
    ]

    for format in formats:
        try:
            filenames = [Path(n).stem for n in vid.names[idx_s:idx_e+1]]
            dates = [datetime.strptime(n, format).replace(tzinfo=timezone.utc) for n in filenames] # use UTC (assume no DST offsets in input timestamps)
            ts = np.array([date.timestamp() for date in dates], dtype=np.float64) # unix times
            assert all(ts[:-1] < ts[1:]), 'Raw float64 timestamps not sorted'
            ts = ts - ts.min()
            n_cycles = int(np.ceil(ts.max() / (24*60*60))) # integer for simplicity
            ts /= (n_cycles * 24*60*60) # normalized to [0, b] where b <= 1.0
            assert all(ts[:-1] < ts[1:]), 'Normalized float64 timestamps not sorted' # float32 sometimes breaks strict inequality
            v.show_date = True
            v.date_format = format
            date_start = dates[0].timestamp() # date of first frame
            return (ts, n_cycles, date_start)
        except ValueError:
            pass
    
    return (np.linspace(0, 1, idx_e - idx_s), 1, 0.0)

# Map conditioning to closest index
def c_to_idx(v, t):
    return np.argmin(np.abs(v.cond - t))

def get_outdir(path):
    indir = Path(path)
    if indir.is_file():
        return indir.parent / f'out_{indir.stem}'
    else:
        return indir / 'out'

def add_warps(p):
    global warps
    M = WarpMatrix(np.load(p))
    warps[p.stem] = {
        'raw': M,
        'decomp': AffineDecomp(M)
    }

def get_manual_xform(v, idx, strict=False):
    idx_ref = np.searchsorted(v.manal.range_starts, v.manal.target_idx, side='right') - 1
    idx_cur = np.searchsorted(v.manal.range_starts, idx, side='right') - 1
    p_ref = np.array(v.manal.range_points_norm[idx_ref], dtype=np.float32)
    p_cur = np.array(v.manal.range_points_norm[idx_cur], dtype=np.float32)

    n_pts = min(p_ref.shape[0], p_cur.shape[0])
    align_xform = np.eye(3, dtype=np.float32).reshape(1, 3, 3)

    if v.warp_mode == 'Translate (2-dof)' and n_pts >= 1:
        align_xform[0, 0:2, 2] = np.mean(p_ref[:n_pts] - p_cur[:n_pts], axis=0)
    elif v.warp_mode == 'Affine (4-dof)' and n_pts >= 2:
        # 4-dof: translation + rotation + uniform scale
        align_xform[0, 0:2, 0:3] = cv2.estimateAffinePartial2D(p_cur[:n_pts], p_ref[:n_pts])[0]
    elif v.warp_mode == 'Affine (6-dof)' and n_pts == 3:
        # Full 6-dof affine (incl. shear)
        align_xform[0, 0:2, 0:3] = cv2.getAffineTransform(p_cur[:n_pts] , p_ref[:n_pts])
    elif v.warp_mode == 'Affine (6-dof)' and n_pts > 3:
        # Full 6-dof affine (incl. shear)
        align_xform[0, 0:2, 0:3] = cv2.estimateAffine2D(p_cur[:n_pts], p_ref[:n_pts])[0]
    elif v.warp_mode == 'Homography (8-dof)' and n_pts >= 4:
        # 8-dof homography
        #align_xform[0, 0:3, 0:3] = cv2.findHomography(p_cur[:n_pts], p_ref[:n_pts])[0]
        raise RuntimeError('Not supported, need to use kornia.geometry.transform.homography_warp and change storage to 9d')
    elif strict:
        return None

    return align_xform

def get_current_transform(v, W, H, top_left=(0, 0), idx=None):
    align_xform = np.eye(3, dtype=np.float32).reshape(1, 3, 3)
    
    # Manual align mode
    if v.active_warp == 'Manual':
        align_xform = get_manual_xform(v, v.img_idx_shifted)
    
    elif v.active_warp in warps and idx is not None:
        data = warps[v.active_warp]['decomp']
        sl = slice(max(0, idx-v.warp_smoothing//2), idx+v.warp_smoothing//2+1)        
        weights = np.ones((data.shape[0], 1))[sl]
        mean = np.sum(data[sl] * weights.reshape(-1, 1), axis=0) / np.sum(weights)
        w = DecompToAffine(*mean).reshape(-1)
        align_xform = np.array([
            w[0], w[1], w[2],
            w[3], w[4], w[5],
               0,    0,    1,
        ]).astype(np.float32).reshape(1, 3, 3)

    # Convert normalized (applied on vectors in [0,1]^2)
    # to unnormalized (inputs in [O,W]x[0,H]) affine
    S = np.diag([W, H, 1.0]).astype(np.float32).reshape(1, 3, 3)
    S_inv = np.linalg.inv(S)
    align_xform = S @ align_xform @ S_inv

    # Build affine xfrom
    tr = torch.tensor([
        (v.transl[0]+v.pan_delta[0])*W,
        (v.transl[1]+v.pan_delta[1])*H
    ], dtype=torch.float32).reshape(1, 2)
    center = torch.tensor([top_left[0] - tr[0, 0] + W/2, top_left[1] - tr[0, 1] + H/2], dtype=torch.float32).reshape(1, 2) # for rotation and zoom
    angle = torch.tensor([v.rot_deg], dtype=torch.float32)
    sc = torch.tensor([v.zoom, v.zoom], dtype=torch.float32).reshape(1, 2)
    M = kornia.get_affine_matrix2d(tr, center, sc, angle)
    total = M @ torch.from_numpy(align_xform)

    return total

# Get frame for preview or export
def get_transformed_frame(v, i):
    t = get_frame_tensor_cached(i).float() / 255 # makes crop / rotation edits faster since decoding is skipped
    _, _, H, W = t.shape

    # Affine grid, in full resolution
    total = get_current_transform(v, W, H, idx=i)

    # Apply transformation to input
    t = kornia.warp_affine(t, total[:, :2, :3].cuda(), dsize=(H, W), mode=v.affine_interp, align_corners=True)

    # Scales, converts to float32
    scale_factor = max(4/min(W, H), v.scale) # result at least 4x4

    # Optional extra AA strength
    # Results in compound function `interp_scaled(x) = s2 * s1 * interp(s2 *s1 * x)`
    interp = getattr(resize_right.interp_methods, v.resize_interp)
    interp, support = resize_right.apply_antialiasing_if_needed(interp, interp.support_sz, 1.0/v.resize_extra_AA, v.resize_AA)
    scaled = resize_right.resize(t, scale_factors=scale_factor, antialiasing=v.resize_AA, interp_method=interp, support_sz=support)

    # Crop
    if any(v.crop) or v.crop_even:
        _, _, H, W = scaled.shape
        cL, cT = [min(dim//2-2, int(c*dim)) for c,dim in zip(v.crop, [W, H])]
        cR, cB = [c + (dim%2) for c,dim in zip([cL, cT], [W, H])] if v.crop_even else [cL, cT]
        scaled = scaled[:, :, :, cL:-cR] if cR > 0 else scaled
        scaled = scaled[:, :, cT:-cB, :] if cB > 0 else scaled

    # Pad
    if any(v.pad):
        _, _, H, W = scaled.shape        
        pW, pH = [min(dim//2-2, int(c*dim)) for c,dim in zip(v.pad, [W, H])]
        scaled = torch.nn.functional.pad(scaled, [pW, pW, 0, 0]) if pW > 0 else scaled
        scaled = torch.nn.functional.pad(scaled, [0, 0, pH, pH]) if pH > 0 else scaled
        v.pad_horiz_px = pW
        v.pad_vert_px = pH
    else:
        v.pad_horiz_px = v.pad_vert_px = 0

    return scaled

def get_date(idx, v):
    try:
        fname = Path(vid.names[idx]).stem
        return datetime.strptime(fname, v.date_format).replace(tzinfo=timezone.utc) # parse
    except:
        return None

# Take normalized inputs in [0,1]
def transform_markers(pos, xform, inverse=False):
    affine = cv2.invertAffineTransform(xform[0, :2, :3]) if inverse else xform[0, :2, :3]
    M = np.eye(3, dtype=np.float32)
    M[0:2, 0:3] = affine
    pos = np.array(pos).reshape(-1, 2)
    pos = np.concatenate([pos, np.ones((pos.shape[0], 1))], -1)
    pos_xform = M.reshape(1, 3, 3) @ pos.reshape(-1, 3, 1)
    return [tuple(p[:2, 0]) for p in pos_xform]

def get_manual_xforms(v):
    ps = np.zeros((vid.n_frames, 6), dtype=np.float64)

    starts = v.manal.range_starts
    ends = starts[1:] + [vid.n_frames]
    for idx_cur, (s, e) in enumerate(zip(starts, ends)):
        M = get_manual_xform(v, s, strict=True)
        if M is None:
            print(f'Range {idx_cur} ({s} - {e}): not enougn points')
            return None

        ps[s:e, :] = PfromWarpMatrix(M[0, 0:2, 0:3]).reshape(-1)

    return ps

def mouse_wheel_callback(window, x, y, self):
    if self.mouse_over_content():
        self.state.zoom = max(1e-2, (0.85**np.sign(-y)) * self.state.zoom)
    else:
        self.imgui_scroll_cbk(window, x, y)

from typing import List, Tuple
from dataclasses import field as field_func
field = lambda in_val : field_func(default_factory=lambda : deepcopy(in_val))

# Manual alignment mode
@dataclass
class ManualAlignState(StrictDataclass):
    target_idx: int = 0                                # Range that others are aligned to
    target_points: List = field([])                    # N order-dependent points on target img
    range_points_norm: List[List[float]] = field([[]]) # One list of points per range, normalized to [0, 1]
    range_starts: List[int] = field([0])               # Start idx of ranges with same alignment
    range_keyframes: List[int] = field([0])            # Index of high-visibility image within range
    n_jumps: int = 0                                            # For adding N equally spaced ranges

@dataclass
class State(StrictDataclass):
    args: EasyDict = None # parsed command line args
    #M: int = 0 # TODO: UNUSED?
    rot_deg: int = 0
    transl: List[int] = field([0, 0])
    crop: List[int] = field([0, 0])
    pad: List[int] = field([0, 0])
    pad_horiz_px: int = 0 # for adding __exactly__ right amount to GAN output layer
    pad_vert_px: int = 0
    crop_even: bool = True
    ignore_manual: str = '' # manually ignore ranges of frames
    zoom: float = 1.0
    scale: float = 1.0
    pan_start: Tuple[int] = (0, 0)
    pan_delta: Tuple[int] = (0, 0)
    idx_s: int = 0
    idx_e: int = 0
    img_idx: int = 0
    img_idx_shifted: int = 0 # after phase shift
    affine_interp: str = 'bilinear'
    resize_interp: str = 'lanczos3'
    resize_extra_AA: int = 1.0
    gl_linear: bool = False
    resize_AA: bool = True
    show_grad_img: bool = False
    show_date: bool = False
    warp_mode: str = 'Affine (4-dof)'
    active_warp: str = 'Manual'
    warp_smoothing: int = 1
    export: bool = False
    window_limit_lbrt: List[int] = field([0, 0, 0, 0])
    cond: np.ndarray = None # not jsonified
    cond_cycles: int = 0
    cycle_phase: int = 0    # phase within cycle
    lock_phase: bool = False # show only images with given phase
    debug_points: List[float] = field([])
    manal: ManualAlignState = None

class VideoPreproc(ToolbarViewer):
    def setup_callbacks(self, window):
        from functools import partial
        self.imgui_scroll_cbk = glfw.set_scroll_callback(window, partial(mouse_wheel_callback, self=self))
    
    def setup_state(self):
        self.state = State()
        self.state.manal = ManualAlignState()
        self.state.args = args
        self.init()

    @property
    def ui_state_dict(self):
        from dataclasses import asdict
        
        state = asdict(self.state)
        del state['cond'] # re-parsed in init()
        del state['export']
        
        return state

    def load_state_from_dict(self, in_dict):
        global vid
        
        # Ignore certain values
        ignores = []
        state_dict = { k: v for k,v in in_dict.items() if k not in ignores }
        
        # Check that input sequence exists
        pth = state_dict['args'].get('path')
        if pth and Path(pth).exists():
            vid = VideoFrames(pth)
        else:
            print(f'Input sequence not found: {pth}')
            del state_dict['vid_path']

        # Manal state
        for k, v in state_dict['manal'].items():
            setattr(self.state.manal, k, v)

        # Load state
        del state_dict['manal']
        for k, v in state_dict.items():
            setattr(self.state, k, v)

        # Other init
        self.init()

    def init(self):
        global warps
        warps = {}

        self.state.args = EasyDict(self.state.args)
        self.state.vid = vid
        self.state.vid.nvjpeg = True
        self.state.idx_e = vid.n_frames - 1
        self.state.img_idx = self.state.idx_s
        self.state.img_idx_shifted = self.state.idx_s

        if self.state.args.labels:
            self.state.cond = np.load(self.state.args.labels)
            self.state.cond_cycles = self.state.args.cycles
        else:
            self.state.cond, self.state.cond_cycles, _ = try_parse_filenames(self.state, 0, vid.n_frames - 1)

        # Find affine alignment matrices, if available
        warpfiles = [p for p in (get_outdir(self.state.args.path)).glob('warps_*.npy')]        
        for p in warpfiles:
            add_warps(p)

        # Export window
        self.export_resume = False
        self.export_restart = False

    def export_frames(self):
        v = self.state

        H, W = self.img_shape[-2:]
        target_q = 95 if max(H, W) >= 512 else 98
        times, n_days, date0 = try_parse_filenames(v, v.idx_s, v.idx_e)
        indir = Path(v.args.path)
        outfile = get_outdir(indir) / f'{indir.name}_{W}x{H}_{n_days}hz.zip'
        print(f'Exporting to {outfile}')

        labels = []
        skipped = 0
        
        if self.export_restart:
            outfile.unlink(missing_ok=True)
        elif outfile.is_file() and not self.export_resume:
            print(f'{outfile} already exists, specify behavior in UI')
            v.export = False
            return

        # Export: skip broken instead of outputting black
        vid.exceptions = True
        
        from io import BytesIO
        import mutable_zipfile as zipfile

        os.makedirs(outfile.parent, exist_ok=True)
        zf = zipfile.ZipFile(file=outfile, mode='a', compression=zipfile.ZIP_STORED)

        # Resume previously aborted export
        idx_start = 0
        if self.export_resume and outfile.is_file():
            data = json.loads(zf.read('dataset.json'))
            self.load_state_from_dict(data['meta'])
            times, n_days, date0 = try_parse_filenames(v, v.idx_s, v.idx_e)
            skipped = data['skipped']
            labels = data['labels']
            
            # Remove images that don't have labels, plus one extra for possible corruption
            N = len(labels) - 1
            imgs = sorted(n for n in zf.namelist() if n.endswith('.jpg'))
            labels = labels[:N]
            for img in imgs[N:]:
                zf.remove(img)
            idx_start = int(imgs[N][-12:-4]) + skipped
        
        # Includes manual ignores and global start-end range
        ids = self.get_active_indices()

        # Export metadata first (for resuming later)
        def write_meta():
            d = self.ui_state_dict
            d['date_start'] = date0
            d['num_days'] = n_days
            d['img_shape'] = self.img_shape
            if 'dataset.json' in zf.NameToInfo:
                zf.remove('dataset.json')
            zf.writestr('dataset.json', json.dumps({'meta': d, 'labels': labels, 'skipped': skipped}))
        
        write_meta()

        for j in trange(idx_start, len(ids), desc='Exporting frames'):
            i = ids[j]
            if not v.export:
                vid.exceptions = False
                self.export_resume = self.export_restart = False
                write_meta()
                zf.close()
                return # aborted from UI

            v.img_idx = i # update UI
            idx_str = f'{j-skipped:08d}'
            archive_fname = f'{idx_str[:5]}/img{idx_str}.jpg'

            try:
                img = get_transformed_frame(v, i) * 255
            except RuntimeError:
                print(f'Skipping frame {i}')
                skipped += 1
                continue
            
            img = img.clip(0, 255).byte().squeeze().permute((1, 2, 0))
            
            if j % 10 == 0:
                self.v.upload_image(self.output_key, img)

            image_bits = BytesIO()
            jpg = Image.fromarray(img.cpu().numpy())
            jpg.save(image_bits, format='jpeg', quality=target_q)
            zf.writestr(archive_fname, image_bits.getbuffer())

            # Timestamp (indexed w.r.t idx_s)
            labels.append([archive_fname, [times[i - v.idx_s].item()]]) # length one array

        # Be extra sure
        ts = np.array([t[0] for _,t in labels])
        assert all(ts[1:] > ts[:-1]), 'Labels not sorted'

        # Save labels and metadata
        write_meta()
        zf.close()

        v.export = False
        vid.exceptions = False
        self.export_resume = self.export_restart = False
        print('Export done')
    
    # Get frame from dataset, transform based on currently active xform
    def compute(self):
        v = self.state

        if v.export:
            self.export_frames()
        
        # Locked phase: get closest match
        if v.lock_phase:
            offset = v.cycle_phase - np.fmod(v.cond[v.img_idx], 1/v.cond_cycles)
            v.img_idx_shifted = c_to_idx(v, v.cond[v.img_idx] + offset)
        else:
            v.img_idx_shifted = v.img_idx

        scaled = get_transformed_frame(v, v.img_idx_shifted)

        # Back to HWC
        img_valid = torch.clip(scaled, 0, 1).squeeze().permute((1, 2, 0))
        
        # Debugging functions
        if v.show_grad_img:
            grad_x, grad_y = torch.gradient(img_valid, dim=(0,1))
            grad_mean = 0.5 * (grad_x.abs().mean(dim=-1) + grad_y.abs().mean(dim=-1))
            grad_mean = torch.clip(5*grad_mean, 0, 1)
            img_valid = grad_mean.unsqueeze(-1).repeat_interleave(3, dim=-1)

        return img_valid

    # Get indices not currently skipped (via global or manual ranges)
    def get_active_indices(self):
        active = np.ones(vid.n_frames, dtype=np.int64)
        active[0:self.state.idx_s] = 0
        active[self.state.idx_e+1:] = 0

        # Parse ignore ranges (e.g. "1, 4, 5-7")
        if self.state.ignore_manual:
            for p in self.state.ignore_manual.split(','):
                s, *e = [int(v) for v in p.strip().split('-', 1)]
                e = s if not e else e[0]
                active[slice(s, e + 1)] = 0

        return np.argwhere(active > 0).reshape(-1)

    # Draw date str based on conditioning
    # Draw as overlay onto output image
    def draw_date_overlay(self, v, draw_list):
        s = self.ui_scale
        date = get_date(v.img_idx_shifted, v)
        date_str = datetime.strftime(date, r'%d-%b-%Y %H:%M') if date else 'Date parsing error'
        font_size = min(round(1.5*self.font_size), max(self.v._imgui_fonts.keys()))
        imgui.push_font(self.v._imgui_fonts[font_size])
        box_w, box_h = 183 * font_size / 22, 27 * font_size / 22
        draw_list.add_rect_filled(self.toolbar_width + 15*s, self.menu_bar_height + 8*s, self.toolbar_width + 15*s + box_w, self.menu_bar_height + 8*s + box_h, imgui.get_color_u32_rgba(0,0,0,1))
        draw_list.add_text(self.toolbar_width + 20*s, self.menu_bar_height + 10*s, imgui.get_color_u32_rgba(1,1,1,1), date_str)
        imgui.pop_font()

    # Manual align mode or debug: draw markers
    def draw_markers(self, v, draw_list, points, frame_idx=None):
        active_affine = get_current_transform(v, 1, 1, idx=frame_idx).numpy()

        for i, (x,y) in enumerate(points):
            colors = [
                (1.0, 0.0, 0.0, 1.0),
                (0.0, 1.0, 0.0, 1.0),
                (0.0, 0.0, 1.0, 1.0),
                (1.0, 1.0, 0.0, 1.0),
                (1.0, 0.0, 1.0, 1.0),
                (0.0, 1.0, 1.0, 1.0),
            ]
            color = imgui.get_color_u32_rgba(*colors[i % len(colors)])
            dims = self.output_pos_br - self.output_pos_tl

            
            pos_xform_norm = transform_markers((x, y), active_affine, inverse=False)[0]
            pos_xform_abs = self.output_pos_tl + pos_xform_norm*dims
            x, y = pos_xform_abs

            # Tick mark
            s = 0.6*self.font_size
            line_width = 0.3*self.font_size
            y -= 1.3*line_width/2 # fix tip at mouse pos

            # Transform with currently active affine
            pos_abs = [(x-s*0.7, y-s), (x, y), (x+s*0.7, y-s)]
            draw_list.add_polyline(pos_abs, color, closed=False, thickness=line_width)

    # extra UI elements under output
    def draw_output_extra(self):
        s = self.state

        # Frame selector
        drag_speed = 0.5
        s.img_idx = imgui.slider_int('##sl', s.img_idx, s.idx_s, s.idx_e, format='')[1]
        imgui.same_line()
        imgui.push_item_width(50 * self.ui_scale)
        s.img_idx = imgui.drag_int('##dr', s.img_idx, drag_speed, s.idx_s, s.idx_e)[1]
        s.img_idx = max(s.idx_s, min(s.idx_e, s.img_idx))
        s.img_idx = max(0, min(s.vid.n_frames-1, s.img_idx))
        imgui.pop_item_width()

        # Phase locking
        if s.cond is not None:
            imgui.same_line()
            changed, s.lock_phase = imgui.checkbox('Lock', s.lock_phase)
            if changed:
                if s.lock_phase:
                    # Lock phase to current value
                    s.cycle_phase = np.fmod(s.cond[s.img_idx], 1/s.cond_cycles)
                else:
                    # Avoid visible jump when locking is removed
                    s.img_idx = s.img_idx_shifted

    def draw_overlays(self, draw_list):
        s = self.state
        if s.show_date:
            self.draw_date_overlay(s, draw_list)

        if s.active_warp == 'Manual':
            active = np.searchsorted(s.manal.range_starts, s.img_idx_shifted, side='right') - 1
            points = s.manal.range_points_norm[active]
            self.draw_markers(s, draw_list, points)
        
        if s.debug_points:
            self.draw_markers(s, draw_list, s.debug_points)

    def jump_to_range(self, i):
        keyframes = self.state.manal.range_keyframes
        starts = self.state.manal.range_starts
        i = min(max(0, i), len(keyframes) - 1)
        self.state.img_idx = self.state.img_idx_shifted = max(keyframes[i], starts[i])
        self.state.lock_phase = False

    @property
    def active_range(self):
        return np.searchsorted(self.state.manal.range_starts, self.state.img_idx_shifted, side='right') - 1
    
    def add_range(self):
        i = self.state.img_idx_shifted
        if i in self.state.manal.range_starts:
            return
        
        ri = self.active_range
        self.state.manal.range_keyframes.insert(ri + 1, i)
        self.state.manal.range_points_norm.insert(ri + 1, [])
        self.state.manal.range_starts.insert(ri + 1, i)
        
        # Make sure keyframes are valid
        prev = self.state.manal.range_keyframes[ri]
        if prev >= i:
            self.state.manal.range_keyframes[ri] = i - 1
            self.state.manal.range_keyframes[ri + 1] = prev

    def draw_manual_align_controls(self, s):
        imgui.separator()
        imgui.text(f'Target: {s.manal.target_idx}')
        imgui.same_line()
        if imgui.button('Set'):
            s.manal.target_idx = s.img_idx_shifted

        active_range = self.active_range
        imgui.same_line()
        if imgui.button('Add'):
            self.add_range()
        
        # Add several evenly spaced ranges for manual alignment
        self.state.manal.n_jumps = imgui.input_int('##n_jumps', self.state.manal.n_jumps, 1, 100)[1]
        imgui.same_line()
        if imgui.button('Add range'):
            starts = np.linspace(s.idx_s, s.idx_e - 1, self.state.manal.n_jumps, dtype=np.int32)
            for st in starts:
                self.state.img_idx_shifted = st
                self.add_range()

        if imgui.button('Export warps##warps_manual'):
            xforms = get_manual_xforms(s)
            if xforms is not None:
                outfile = get_outdir(s.args.path) / 'warps_manual.npy'
                makedirs(outfile.parent, exist_ok=True)
                np.save(str(outfile), xforms.reshape(-1, 6))
                print('Exported to', outfile)

                # Update UI
                add_warps(outfile)
                s.active_warp = outfile.stem
            else:
                print('Not exported')

        # Scrollable list
        imgui.begin_child("range_widget_list", self.toolbar_width)
        
        starts = s.manal.range_starts
        ends = s.manal.range_starts[1:] + [vid.n_frames]
        for i, (si, ei) in enumerate(zip(starts, ends)):
            color = [1.0, 0.0, 0.0] if i == active_range else [0.8, 0.8, 0.8]
            n_pts = len(s.manal.range_points_norm[i])
            imgui.push_style_color(imgui.COLOR_TEXT, *color)
            imgui.text(f'[{si:<7}-{ei:>7}] ({n_pts}pts)')
            imgui.pop_style_color()

            imgui.same_line()
            if imgui.button(f'DEL##{i}') and len(starts) > 1:
                del s.manal.range_starts[i]
                del s.manal.range_keyframes[i]
                del s.manal.range_points_norm[i]
                s.manal.range_starts[0] = 0 # ensure initial zero is not deleted
                break

            imgui.same_line()
            if imgui.button(f'CLR##{i}') and n_pts > 0:
                s.manal.range_points_norm[i] = []

            imgui.same_line()
            if imgui.button(f'JMP##{i}'):
                self.jump_to_range(i)
        
        imgui.end_child()
        
        # Keyboard controls
        if self.v.keyhit(glfw.KEY_LEFT):
            self.jump_to_range(active_range - 1)
        if self.v.keyhit(glfw.KEY_RIGHT):
            self.jump_to_range(active_range + 1)
        if self.v.keyhit(glfw.KEY_UP):
            s.lock_phase = False
            s.img_idx = max(s.idx_s, min(s.img_idx + 1, s.idx_e))
        if self.v.keyhit(glfw.KEY_DOWN):
            s.lock_phase = False
            s.img_idx = max(s.idx_s, min(s.img_idx - 1, s.idx_e))
        if self.v.keyhit(glfw.KEY_SPACE):
            self.add_range()

        if self.mouse_over_image():
            # Add points
            active_affine = get_current_transform(s, 1, 1).numpy()
            pos_scr_norm = self.mouse_pos_img_norm
            
            if imgui.core.is_mouse_clicked(0): # left
                x, y = transform_markers(pos_scr_norm, active_affine, inverse=True)[0]
                s.manal.range_points_norm[active_range].append((x, y)) # normalized, relative to img content
                s.manal.range_keyframes[active_range] = s.img_idx_shifted # use current frame as keyframe
            if imgui.core.is_mouse_clicked(1): # right
                x, y = transform_markers(pos_scr_norm, active_affine, inverse=True)[0]
                sqr_dist = [(x-a)**2+(y-b)**2 for (a,b) in s.manal.range_points_norm[active_range]]
                if sqr_dist and min(sqr_dist) < (1/100)**2: # within 1% of img width
                    del s.manal.range_points_norm[active_range][np.argmin(sqr_dist)]
        
            # Pan view with mouse
            a, b = pos_scr_norm
            if imgui.core.is_mouse_clicked(2): # wheel down
                s.pan_start = (a, b)
            if imgui.core.is_mouse_down(2):
                s.pan_delta = (a - s.pan_start[0], b - s.pan_start[1])
            if imgui.core.is_mouse_released(2): # wheel up
                s.transl = tuple(s+d for s,d in zip(s.transl, s.pan_delta))
                s.pan_start = s.pan_delta = (0, 0)

    # Compute largest window that fits all xformed frames
    def fit_window_to_warps(self):
        s = self.state

        ids = self.get_active_indices()
        N = len(ids)

        xforms = np.eye(3)[None, ...].repeat(N, axis=0) # (N,3,3)
        xforms[:, 0:2, 0:3] = warps[s.active_warp]['raw'][ids]
        
        # Transformed corners for every frame
        tl = (xforms @ np.array([0, 0, 1])).T
        bl = (xforms @ np.array([0, 1, 1])).T
        tr = (xforms @ np.array([1, 0, 1])).T
        br = (xforms @ np.array([1, 1, 1])).T
        
        # Corners that affect every extent
        x0s = np.concatenate([tl[0, :], bl[0, :]]) # left edge (x-coord)
        y0s = np.concatenate([tl[1, :], tr[1, :]]) # top edge (y-coord)
        x1s = np.concatenate([tr[0, :], br[0, :]]) # right edge (x-coord)
        y1s = np.concatenate([bl[1, :], br[1, :]]) # bottom edge (y-coord)

        # Img indices that determine corners (for debugging)
        # In range (0, 2*N)
        ix0 = x0s.argmax(axis=-1); x0 = x0s[ix0]
        iy0 = y0s.argmax(axis=-1); y0 = y0s[iy0]
        ix1 = x1s.argmin(axis=-1); x1 = x1s[ix1]
        iy1 = y1s.argmin(axis=-1); y1 = y1s[iy1]

        # Inscribed axis-aligned box:
        W, H = (x1 - x0, y1 - y0)
        center = np.array([x0 + 0.5*W, y0 + 0.5*H])
        img_inds = [ids[i % N] for i in [ix0, iy1, ix1, iy0]]
        print('Limits (LBRT) defined by:', *img_inds)
        
        # Debugging
        #s.debug_points.clear()
        #s.debug_points.append((x0, y0))
        #s.debug_points.append((x0, y1))
        #s.debug_points.append((x1, y0))
        #s.debug_points.append((x1, y1))
        #s.debug_points.append((center[0], center[1]))

        # Update UI
        s.pan_delta = (0, 0)
        s.transl = (0.5 - center[0], 0.5 - center[1])
        s.crop = (0.5*(1-W), 0.5*(1-H)) # W,H
        s.rot_deg = 0
        s.zoom = 1

        return img_inds

    def draw_toolbar(self):
        s = self.state
        s.scale = imgui.slider_float('', s.scale, 0.05, 1.0, '%.5f')[1]; imgui.same_line(); imgui.text(f'{self.img_shape[2]}x{self.img_shape[1]}')
        s.resize_AA = imgui.checkbox('AA', s.resize_AA)[1]
        s.resize_interp = combo_box_vals('Resize interp', ['cubic', 'lanczos2', 'lanczos3', 'linear', 'box'], s.resize_interp)[1]
        s.resize_extra_AA = imgui.slider_float('Extra AA', s.resize_extra_AA, 1, 3)[1]
        
        s.rot_deg = imgui.slider_float('Rotation (deg)', s.rot_deg, -180, 180)[1]
        s.zoom = imgui.slider_float('Zoom', s.zoom, 0.2, 5.0)[1]
        s.transl = imgui.slider_float2('Transl. (X,Y)', *s.transl, -1, 1)[1]
        s.affine_interp = combo_box_vals('Affine interp', ['nearest', 'bilinear'], s.affine_interp)[1]
        changed, s.gl_linear = imgui.checkbox('GL_LINEAR', s.gl_linear)
        if changed:
            (self.v.set_interp_linear if s.gl_linear else self.v.set_interp_nearest)()
            
        s.vid.nvjpeg = imgui.checkbox('NVJpeg', s.vid.nvjpeg)[1]
        
        # Debugging
        imgui.separator()
        s.show_grad_img = imgui.checkbox('Gradient img', s.show_grad_img)[1]
        s.show_date = imgui.checkbox('Show date', s.show_date)[1]

        # Warps
        if warps is not None:
            s.active_warp = combo_box_vals('Warps', ['Off', 'Manual'] + list(warps.keys()), s.active_warp)[1]
            if s.active_warp == 'Manual':
                s.warp_mode = combo_box_vals('Mode', ['Translate (2-dof)', 'Affine (4-dof)', 'Affine (6-dof)'], s.warp_mode)[1]
            s.warp_smoothing = imgui.slider_int('Avg. window', s.warp_smoothing, 1, 100)[1]
            if s.active_warp in warps: 
                if imgui.button('Fit window'):
                    s.window_limit_lbrt = self.fit_window_to_warps()
                # Buttons for navigating to imgs that limit window size
                for lab, idx in zip(['L', 'B', 'R' ,'T'], s.window_limit_lbrt):
                    if imgui.button(f'{lab}: {idx}##jump_img_{lab}'):
                        s.img_idx = idx; s.lock_phase = False
                    if lab != 'T':
                        imgui.same_line()

        # Output cropping and trimming
        imgui.separator()
        s.idx_s, s.idx_e = imgui.slider_int2('Trim', s.idx_s, s.idx_e, 0, s.vid.n_frames - 1)[1]
        s.pad = imgui.slider_float2('Pad (W,H)', *s.pad, 0, 0.25, format='%.4f')[1]
        s.crop = imgui.slider_float2('Crop (W,H)', *s.crop, 0, 0.25, format='%.4f')[1]
        s.crop_even = imgui.checkbox('Force even size', s.crop_even)[1]

        # Manually remove frame ranges
        imgui.separator()
        s.ignore_manual = imgui.input_text('Skipped ranges', s.ignore_manual, 2048)[1]
        
        imgui.separator()
        if imgui.button('Reset'):
            s.scale = 1.0
            s.rot_deg = 0.0
            s.zoom = 1.0
            s.transl = [0, 0]
            s.crop = [0, 0]
            s.pad = [0, 0]

        if not s.export and imgui.button('Export frames'):
            s.export = True
        if s.export and imgui.button('Cancel'):
            s.export = False

        imgui.same_line()
        ch1, self.export_resume = imgui.checkbox('Resume', self.export_resume)
        imgui.same_line()
        ch2, self.export_restart = imgui.checkbox('Delete', self.export_restart)
        if ch1 and self.export_resume:
            self.export_restart = False
        if ch2 and self.export_restart:
            self.export_resume = False

        # Manual alignment settings
        if s.active_warp == 'Manual':
            self.draw_manual_align_controls(s)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Video preprocessing')
    parser.add_argument('path', type=str, help='Path to frames (AMOS data or output of get_video_frames.py)')
    parser.add_argument('--labels', type=str, default=None, help='Path to conditioning labels (if not parsable from filenames)')
    args = EasyDict(vars(parser.parse_args()))

    # Wrapper for frame collection
    vid = VideoFrames(args.path, torch=True)
    
    # Go fast
    torch.autograd.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True
    
    # Stay safe
    os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    viewer = VideoPreproc('seq_preproc')
    print('Done')