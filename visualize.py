# Copyright (c) 2022 Erik Härkönen, Aalto University
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

import os
import dnnlib
assert dnnlib.__file__ == os.path.abspath(os.path.join(__file__, '../dnnlib/__init__.py'))

import sys
import numpy as np
import threading
import imgui
import argparse
import time
import pickle
from io import BytesIO
from enum import Enum
from viewer import gl_viewer, utils
from viewer.utils import with_state, imgui_id, imgui_item_width, stack_to_global_t, open_prog
from pathlib import Path
from typing import Iterable
import dnnlib
from dnnlib import EasyDict
import torch
from viewer.utils import ParamCache

# Clear this dir if custom op compilation gets stuck
# assert 'TORCH_EXTENSIONS_DIR' in os.environ, 'Please set environment variable "TORCH_EXTENSIONS_DIR"'

###########
# UTILITIES
###########


pkl_cache = {}
def read_pkl_cached(pkl):
    global pkl_cache
    key = (str(pkl), Path(pkl).lstat().st_mtime)
    if key not in pkl_cache:
        pkl_cache.setdefault(key, open_prog(pkl, 'rb').read())
    return BytesIO(pkl_cache.get(key))

# Path to cached PCA components
def comp_path(pkl):
    p = Path(pkl)
    return p.parent / '.comp' / 'pca' / p.with_suffix('.npz').name

# Path to folder containing saved latent dirs
def dir_path(pkl):
    p = Path(pkl)
    return p.parent / '.comp' / 'dirs' / p.with_suffix('').name

def update_cff(v):
    if v.lat_expl.mode not in v.lat_expl.cff.modes:
        return

    s = v.lat_expl.cff.get('l_start', 1)
    e = v.lat_expl.cff.get('l_end', utils.cff_max_layer(v.net.G)) + 1 # end inclusive in UI
    def callback(msg):
        v.lat_expl.cff.status = msg
    v.lat_expl.cff.comp = utils.compute_cff(v.net.G, s - 1, e - 1, v.lat_expl.mode, callback)

def get_latent_edit(comp, std, coord):
    n_comp = min(comp.shape[0], len(std))
    coords = coord[:n_comp].reshape(n_comp, 1)
    offsets = comp[:n_comp, :] * std[:n_comp].reshape(n_comp, 1) * coords
    return np.sum(offsets, axis=0).reshape(1, -1)

# Apply conditioning edit specified by UI
# Edit linear c or Fourier stack (if using Fourier cond.)
# Can edit globally or per-layer
def apply_cond_edit(cs, v):
    if v.cond.mode == 'Linear' and v.net.has_strided_cond:
        cs = cs.unsqueeze(1).repeat_interleave(v.net.G.num_ws, dim=1)
        for ran, t in v.cond.ts:
            cs[:, ran[0]:ran[1], :] = t
        if hasattr(v.net.G, 'cond_xform'):
            cs = v.net.G.cond_xform(cs, broadcast=False)
    elif v.cond.mode == 'Fourier':
        num_f = v.net.G.cond_xform.num_f
        cs = cs.repeat_interleave(num_f, dim=-1).unsqueeze(1).repeat_interleave(v.net.G.num_ws, dim=1) # [B, #layers, #freq]

        for ran, freq in v.cond.fs:
            cs[:, ran[0]:ran[1], :] = torch.tensor(freq, dtype=torch.float32, requires_grad=False)
        
        cs = v.net.G.cond_xform(cs, broadcast=False)

    return cs

# Latent exploration
def apply_lat_edit(ws, v, scale=1.0):
    deltas = np.zeros((1, v.net.G.num_ws, v.net.G.z_dim), dtype=np.float32)
    for m in [v.lat_expl.pca, v.lat_expl.cff]:
        if m.comp:
            d = get_latent_edit(*m.comp, m.sliders*scale)
            s, e = (m.l_start, m.l_end)
            for l in range(s - 1, e): # 0-based indexing, end inclusive
                deltas[:, l:l+1, :] += d
    
    return ws + torch.tensor(deltas).to(ws.device)

def upscale_2x_box(img):
    bigger = np.zeros((2*img.shape[0], 2*img.shape[1], 3), dtype=np.float32)
    for (offs_x, offs_y) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        bigger[offs_x::2, offs_y::2] = img
    return bigger

def swap_model(v):
    setup_model(v)
    
    # Refresh latent dirs, training data viz
    utils.cff_cache.clear()
    v.lat_expl.pca.mark_stale() # show update button
    v.lat_expl.cff.comp = None # immediately recompute CFF (fast)
    v.training_data = None

    # Get cached PCA comps immediately if available
    if comp_path(pkl).is_file():
        v.lat_expl.pca.stale = False

    print('done')

def setup_model(v):
    f = read_pkl_cached(pkl)
    data = pickle.load(f)
    v.net.G = data['G_ema'].to('cuda')
    v.net.D = EasyDict(init_kwargs=data['D'].init_kwargs)
    dataset_args = data['training_set_kwargs'] or {}
    v.net.dataset_name = Path(dataset_args.get('path', pkl)).name
    
    # All own torch pickles support one c per layer
    if 'cond_args' in v.net.G.init_kwargs:
        v.net.has_strided_cond = True
        if v.net.G.init_kwargs.cond_args.type in ['fourier', 'f_concat']:
            # Show per frequency controls in UI
            v.cond.mode = 'Fourier'
            v.cond.fs = [[(0, v.net.G.num_ws), [v.cond.t]*len(v.net.G.cond_xform.get_frequencies())]]
        else:
            # Show linear c control in UI
            v.cond_mode = 'Linear'
            v.cond.fs = []
    else:
        v.cond.mode = 'Linear'
        v.net.has_strided_cond = False

    # Loading done
    v.state = State.IDLE

# Return closest training data img given timestamp
def get_gt_frame(v, t):
    # Default value used if no dataset is found
    res = v.net.G.img_resolution
    empty = np.zeros((res, res, 3), dtype=np.uint8)

    if not v.net.dataset_name:
        return empty

    # Load images into memory once
    if v.training_data is None:
        path = Path(v.net.dataset_path)

        # Dataset provided as cmd line arg directly
        if path.suffix == '.zip':
            assert path.is_file(), f'{path} not found'
        else:
            matches = list(path.glob(f'**/{v.net.dataset_name}'))
            if len(matches) == 0:
                # Give up
                print(f'Dataset {v.net.dataset_name} not found under {v.net.dataset_path}')
                v.training_data = EasyDict(meta={})
                return empty
                
            path = matches[0]

        # Read conditioning labels
        from training.dataset import ImageFolderDataset
        training_set = ImageFolderDataset(str(path), use_labels=True)
        labels = np.zeros(len(training_set))
        for idx in range(len(training_set)):
            label = training_set.get_details(idx).raw_label.flat[::-1]
            labels[idx] = label

        # Figure out sorted order
        sorting = np.argsort(labels.squeeze())
        labels = labels[sorting]
        if not all(sorting[1:] > sorting[:-1]):
            print('\nWARNING: dataset labels not sorted! (UTC-related?)\n')
        reverse_mapping = np.argsort(sorting)
        
        v.training_data = EasyDict(cond=labels, meta=training_set._get_meta())

        from preproc.tools import VideoFrames
        v.training_data.reader = VideoFrames(path, torch=False)

    # No dataset found, return empty image
    if not hasattr(v.training_data, 'reader'):
        return empty

    # Find closest match
    dists = np.abs(t - v.training_data.cond)
    best_i = np.argmin(dists)
    img = v.training_data.reader[best_i]

    return img

# Get date that best describes conditioning state
# Input: single t or stack of ts [t_lin, t_f1, ..., t_fn]
def get_date(t):
    if v.training_data is None:
        return ''

    days = v.training_data.meta.get('num_days', 1)
    ts0 = v.training_data.meta.get('date_start', 946684800) # 1.1.2000
    
    if isinstance(t, Iterable):
        fs = v.net.G.cond_xform.get_frequencies()
        t = stack_to_global_t(np.array(t).reshape(1, -1), fs).item()
    
    end = ts0 + t * days * 24 * 60 * 60
    
    from datetime import datetime
    return datetime.utcfromtimestamp(int(end)).strftime(r'%d.%m.%Y %H:%M:%S')


###############################
# GAN OUTPUT THREAD STATE FUNCS
###############################

def state_idle(v):
    timer = utils.Timer(n_avg=20)
    while not v.quit and v.state == State.IDLE:
        if is_visible('Output'):
            B = v.batch_size
            noise_mode = 'const' #'const', 'random', 'none'
            
            if not v.frozen_seed:
                v.seed = np.random.randint(np.iinfo(np.int32).max - B)

            # Per-layer conditioning
            cs = v.cond.t * torch.ones([B, v.net.G.c_dim], dtype=torch.float32, device='cuda')
            cs = apply_cond_edit(cs, v) # [B, n_layers, 6]

            # Generate latents
            latents = torch.from_numpy(utils.sample_latent(B, v.net.G.z_dim, v.seed)).pin_memory().cuda()
            ws = v.net.G.mapping(latents, cs[:, 0, :], truncation_psi=v.truncation, truncation_cutoff=v.truncation_cutoff)

            # PCA or CFF edit directions
            ws = apply_lat_edit(ws, v, scale=v.lat_expl.slider_total)

            # Saved dirs edit
            ws = ws + torch.tensor(v.lat_expl.saved_dirs_offset, device='cuda')

            # Randomize noise?
            if not v.frozen_noise:
                noise_mode = 'random'

            # Network output
            out_tensor = 0.5*(v.net.G.synthesis(ws, cs[:, :, 0:v.net.G.synthesis.c_dim], noise_mode=noise_mode) + 1)
            
            # Add training frame?
            if v.show_ref:
                ref_np = get_gt_frame(v, v.cond.t_gt)
                ref = torch.from_numpy(ref_np).cuda().permute(2, 0, 1).unsqueeze(0)
                out_tensor = torch.cat((out_tensor, ref.float() / 255.0), dim=0)

            # Crop padding if known
            if hasattr(v.net.G.synthesis, 'out_rect'):
                x1, y1, x2, y2 = v.net.G.synthesis.out_rect
                out_tensor = out_tensor[:, :, y1:y2, x1:x2]
            
            v.upload_image_torch('output', utils.reshape_grid_torch(out_tensor))

            # Lazily evaluated conditioning debug
            if is_visible('Conditioning debug'):
                utils.cond_viz(pkl, v, v.net.G)

            # Update CFF dirs
            if not v.lat_expl.cff.comp:
                update_cff(v)

            v.fps = 1.0 / timer.tick()
        else:
            time.sleep(1)

############
# UI WINDOWS
############

def get_window_flags(v):
    if v.windows_locked:
        return imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS
    else:
        return 0

def begin_window(v, name):
    if v.auto_layout:
        with imgui.styled(imgui.STYLE_WINDOW_ROUNDING, 0):
            imgui.begin(name,
                flags = \
                    imgui.WINDOW_NO_TITLE_BAR |
                    imgui.WINDOW_NO_RESIZE |
                    imgui.WINDOW_NO_MOVE |
                    imgui.WINDOW_NO_COLLAPSE |
                    imgui.WINDOW_NO_SAVED_SETTINGS
            )
    else:
        imgui.begin(name, flags=get_window_flags(v))

def end_window(v):
    if v.auto_layout:
        imgui.end()
    else:
        imgui.end()

def window_settings(v):
    begin_window(v, 'Settings')
    v.seed = imgui.input_int('Seed', v.seed, step_fast=v.batch_size)[1]
    v.batch_size = imgui.slider_int('Batch size', v.batch_size, 1, 16, '%d')[1]
    
    with imgui_item_width(v.ui_scale * 105):
        v.truncation = imgui.slider_float('Trunc.', v.truncation, 0.0, 1.0)[1]
        imgui.same_line()
        v.truncation_cutoff = imgui.slider_int('Cutoff', v.truncation_cutoff, 0, v.net.G.num_ws)[1]
    
    v.frozen_seed = imgui.checkbox('Freeze seed', v.frozen_seed)[1]
    imgui.same_line()
    v.frozen_noise = imgui.checkbox('Freeze noise', v.frozen_noise)[1]

    ch, v.show_ref = imgui.checkbox('Show training frame', v.show_ref)    
    if ch:
        toggle_window('Training data')
    end_window(v)

def window_output(v):
    begin_window(v, 'Output')
    imgui.text(pkl)
    if v.state == State.LOADING_MODEL:
        imgui.text('Loading...')
    imgui.text(f'FPS: {v.fps:.1f}')
    
    v.draw_image('output', width='fit', pad_v=15+v.ui_scale*130)
    changed, v.gl_linear = imgui.checkbox('GL_LINEAR', v.gl_linear)
    if changed:
        (v.set_interp_linear if v.gl_linear else v.set_interp_nearest)()
    end_window(v)

def pca_expl(v):
    # Wait for compute process to fill in comps
    if v.lat_expl.pca.stale:
        if imgui.button('Compute'):
            v.lat_expl.pca.comp = None
            v.lat_expl.pca.stale = False
            v.lat_expl.pca.status = 'Spawning process...'
        return

    if not v.lat_expl.pca.comp:
        imgui.text(v.lat_expl.pca.status)
        return

    l_max = utils.parse_n_styles(v.net.G)
    v.lat_expl.pca.l_start, v.lat_expl.pca.l_end = \
        utils.slider_range(v.lat_expl.pca.l_start, v.lat_expl.pca.l_end, 1, l_max)

    imgui.begin_child("pca_sliders")
    
    n_comp = min(150, len(v.lat_expl.pca.comp[0])) # show most meaningful
    for i in range(n_comp):
        v.lat_expl.pca.sliders[i] = imgui.slider_float(f'v{i}', v.lat_expl.pca.sliders[i], -8, 8)[1]

    imgui.end_child()

def cff_expl(v):
    # Wait for compute thread to fill in comps
    if not v.lat_expl.cff.comp:
        imgui.text(v.lat_expl.cff.status)
        return

    l_max = utils.parse_n_styles(v.net.G) - 1
    v.lat_expl.cff.l_start, v.lat_expl.cff.l_end = \
        utils.slider_range(v.lat_expl.cff.l_start, v.lat_expl.cff.l_end, 1, l_max)

    imgui.same_line()
    if imgui.button('Recompute'):
        v.lat_expl.cff.comp = None
        return

    imgui.begin_child("cff_sliders")
    
    n_comp = min(150, len(v.lat_expl.cff.comp[0])) # show most meaningful
    for i in range(n_comp):
        v.lat_expl.cff.sliders[i] = imgui.slider_float(f'v{i}', v.lat_expl.cff.sliders[i], -20, 20)[1]

    imgui.end_child()

@with_state(name='')
def window_latent_export_dialog(v, state):
    if not imgui.begin('Direction export', closable=True)[1]:
        toggle_window('Latent export')

    state.name = imgui.input_text('Name', state.name, 256)[1]
    outfile = (dir_path(pkl) / state.name).with_suffix('.pkl')
    
    button_active = state.name != '' and not outfile.is_file()
    imgui.push_style_var(imgui.STYLE_ALPHA, 1.0 if button_active else 0.4)
    if imgui.button('Save') and button_active:
        d = apply_lat_edit(torch.zeros(1), v).numpy().squeeze() # normalized to look OK in range t=[-1, 1]

        os.makedirs(outfile.parent, exist_ok=True)
        with open(outfile, 'wb') as f:
            pickle.dump(dict(
                name=state.name,
                dir=d,
                known_good_seed=v.seed,
                network=pkl,
            ), f)

        state.reset()
        refresh_saved_edits()
        toggle_window('Latent export')
    imgui.pop_style_var()

    imgui.end()

def window_latent_explorer(v):
    begin_window(v, 'Latent Explorer')
    changed, v.lat_expl.mode = utils.combo_box_vals('Mode',
        v.lat_expl.pca.modes + v.lat_expl.cff.modes, v.lat_expl.mode)

    # Single slider for combined effects with export button
    v.lat_expl.slider_total = imgui.slider_float('Sum', v.lat_expl.slider_total, -1, 1)[1]
    imgui.same_line()
    if imgui.button('Save'):
        toggle_window('Latent export')

    if v.lat_expl.mode in v.lat_expl.pca.modes:
        pca_expl(v)
    elif v.lat_expl.mode in v.lat_expl.cff.modes:
        if changed: # recompute
            v.lat_expl.cff.comp = None
        cff_expl(v)
    else:
        raise RuntimeError('Invalid latent exploration mode ' + v.lat_expl.mode)

    end_window(v)

# Apply previously saved edits
@with_state(refresh=True, sliders=np.zeros(0), dirs=np.zeros(0), names=[])
def window_saved_edits(v, state):
    begin_window(v, 'Saved dirs')
    
    if state.refresh:
        files = list(dir_path(pkl).glob('*.pkl'))
        if len(files) > 0:
            state.dirs = np.stack([pickle.loads(f.read_bytes())['dir'] for f in files], axis=0)
            state.names = [f.with_suffix('').name for f in files]
            state.sliders = np.zeros((len(state.dirs), 1, 1), dtype=np.float32)
        state.refresh = False
    
    for i in range(len(state.dirs)):
        state.sliders[i] = imgui.slider_float(state.names[i], state.sliders[i], -5, 5)[1]
    
    v.lat_expl.saved_dirs_offset = np.sum(state.sliders * state.dirs, axis=0).astype(np.float32)
    end_window(v)

def refresh_saved_edits():
    window_saved_edits.keywords['state'].reset()

def window_console(v):
    begin_window(v, 'Log')
    imgui.text_wrapped(sys.stdout.str)
    imgui.set_item_default_focus()
    curr = imgui.get_scroll_y()
    height = imgui.get_scroll_max_y()
    if v.log_updated and height - curr < 10: # almost at end
        imgui.set_scroll_here(1.0)
    v.log_updated = False
    end_window(v)

def window_G_info(v):
    import json
    begin_window(v, 'Networks')
    imgui.text('G: ' + json.dumps(v.net.G.init_kwargs, sort_keys=True, indent=4))
    imgui.text('D: ' + json.dumps(v.net.D.init_kwargs, sort_keys=True, indent=4))
    end_window(v)

def set_t_global(v, t):
    v.cond.t = t
    v.cond.ts = [[ran, t] for ran,_ in v.cond.ts]
    v.cond.fs = [[ran, [t] * len(ts)] for ran,ts in v.cond.fs]

# Show per-layer conditioning controls
def window_cond_controls(v):
    begin_window(v, 'Conditioning')

    # Mode toggle: linear or fourier
    if v.net.G.init_kwargs.get('cond_args', {}).get('type') in ['fourier', 'f_concat']:
        v.cond.mode = utils.combo_box_vals('Edit mode', ['Linear', 'Fourier'], v.cond.mode)[1]
    
    # Single global t always shown
    n_layers = utils.parse_n_styles(v.net.G)
    changed, newval = imgui.slider_float('Global t', v.cond.t, 0.0, 1.0, format="%.4f")

    if v.cond.mode == 'Linear':
        # Per-layer override
        if v.net.has_strided_cond:
            for i in range(len(v.cond.ts)):
                v.cond.ts[i][1] = imgui.slider_float(f'##{i}', v.cond.ts[i][1], 0.0, 1.0)[1]
                imgui.same_line()
                v.cond.ts[i][0] = utils.slider_range(*v.cond.ts[i][0], 0, n_layers, title=f'##{i}', width=100)
                imgui.same_line()
                if imgui.small_button(f'x##{i}'):
                    del v.cond.ts[i]; break
            if imgui.button('Add'):
                v.cond.ts.append([(0, n_layers), v.cond.t])
    else:
        freq = v.net.G.cond_xform.get_frequencies()
        names = ['Trend', 'Year', 'Day'] if len(freq) == 3 else ['Trend', 'Day']

        # Per-layer override
        for i, (ran, ts) in enumerate(v.cond.fs):
            with imgui_id(f'fourier_local_{i}'):
                # Overrides provide layer range, can be deleted
                if i > 0:
                    v.cond.fs[i][0] = utils.slider_range(*ran, 0, n_layers)
                    imgui.same_line()
                    if imgui.small_button('x'):
                        del v.cond.fs[i]; break
                for f_id, f in enumerate(freq):
                    if f < 1:
                        # Slider over whole range
                        f = abs(f) # explicit lin mode indicated with f=-1
                        v.cond.fs[i][1][f_id] = imgui.slider_float(names[f_id], ts[f_id] * f, 0.0, f, format='%.6f')[1] / f
                    else:
                        # Slider over 2 cycles locally
                        v.cond.fs[i][1][f_id] = imgui.slider_float(names[f_id], ts[f_id] * f, v.cond.t*f - 1, v.cond.t*f + 1, format='%.3f')[1] / f
                        imgui.same_line()
                        if imgui.small_button(f'-##{f_id}'):
                            newval = v.cond.t_gt - 1/f
                            changed = True
                        imgui.same_line()
                        if imgui.small_button(f'+##{f_id}'):
                            newval = v.cond.t_gt + 1/f
                            changed = True

        if imgui.button('Add'):
            v.cond.fs.append([(0, n_layers), [v.cond.t]*len(freq)])
    
    # Global slider overrides rest
    if changed:
        set_t_global(v, newval)

    # Get best-matching date of active override
    if v.cond.mode != 'Linear':
        fs = v.net.G.cond_xform.get_frequencies()
        #t1 = np.clip(t1, -0.9999, 0.9999) # strange jumps at exactly +-1 whole cycle
        #t2 = np.clip(t2, -0.9999, 0.9999)
        v.cond.t_gt = stack_to_global_t(np.array(v.cond.fs[-1][1]).reshape(1, -1), fs).item()
    elif len(v.cond.ts) > 0:
        v.cond.t_gt = v.cond.ts[-1][1]

    imgui.text(get_date(v.cond.t_gt))
    end_window(v)

# Show closest matching frame from training data
def window_training_data(v):
    begin_window(v, 'Training data')
    v.draw_image('gt_img', width='fit', pad_v=v.ui_scale*27)
    changed, t_out = imgui.slider_float('T', v.cond.t_gt, 0.0, 1.0, format="%.5f")
    if changed:
        set_t_global(v, t_out)
    end_window(v)

# pyimgui.readthedocs.io/en/latest/reference/imgui.core.html#imgui.core.begin_main_menu_bar
def window_toolbar(v):
    if imgui.begin_main_menu_bar():
        v.menu_bar_height = imgui.get_window_height()

        if imgui.begin_menu('View', True):
            for n, (f, active) in togglable_windows.items():
                clicked, _ = imgui.menu_item(n, selected=active)
                if clicked:
                    togglable_windows[n] = (f, not active)

            imgui.end_menu()

        # Right-aligned button for locking / unlocking UI
        T = 'L' if v.windows_locked else 'U'
        C = [0.8, 0.0, 0.0] if v.windows_locked else [0.0, 1.0, 0.0]

        s = v.ui_scale

        # UI scale slider
        if not v.windows_locked:
            imgui.same_line(position=imgui.get_window_width()-300-25*s)
            with imgui_item_width(300): # size not dependent on s => prevents slider drift
                ch, val = imgui.slider_float('', s, 0.5, 2.0)
            if ch:
                v.set_ui_scale(val)

        imgui.same_line(position=imgui.get_window_width()-25*s)
        imgui.push_style_color(imgui.COLOR_TEXT, *C)
        if imgui.button(T, width=20*s):
            v.windows_locked = not v.windows_locked
        imgui.pop_style_color()

        imgui.end_main_menu_bar()

######################
# USER INTERFACE STATE
######################

# State machine
class State(Enum):
    IDLE = 0
    LOADING_MODEL = 1
    PLAYING_ANIM = 2
    SWAPPING_MODEL = 3

# Name: (func, is_visible)
togglable_windows = {
    'Output': (window_output, True),
    'Settings': (window_settings, True),
    'Log': (window_console, True),
    'Info': (window_G_info, True),
    'Conditioning controls': (window_cond_controls, True),
    'Training data': (window_training_data, True),
    'Latent explorer': (window_latent_explorer, True),
    'Saved dirs': (window_saved_edits, True),
    'Latent export': (window_latent_export_dialog, False),
}

def is_visible(window_name):
    return togglable_windows.get(window_name, (None, False))[1]

def toggle_window(window_name):
    f, visible = togglable_windows[window_name]
    togglable_windows[window_name] = (f, not visible)

def handle_keys(v):
    import glfw
    
    if v.keyhit(glfw.KEY_F11):
        v.toggle_fullscreen()
    
    if v.keyhit(glfw.KEY_HOME):
        window_saved_edits.keywords['state'].sliders *= 0
        v.lat_expl.slider_total = 1.0
        if v.lat_expl.mode in v.lat_expl.pca.modes:
            v.lat_expl.pca.reset_sliders()
        if v.lat_expl.mode in v.lat_expl.cff.modes:
            v.lat_expl.cff.reset_sliders()

def file_drop_callback(window, paths):
    global pkl
    pickles = [p for p in paths if p.lower().endswith('.pkl')]
    if len(pickles) == 0:
        print('Please drop a .pkl pretrained model')
    elif len(pickles) > 1:
        print('Please drop only one pickle')
    else:
        pkl = pickles[0]
        v.state = State.SWAPPING_MODEL

#####################
# THREAD ENTRY POINTS
#####################

def _ui_loop_auto_layout(v):
    included = []
    
    def show(k, w, h, offset_x=0, offset_y=0):
        if is_visible(k):
            imgui.set_next_window_size(w, h)
            imgui.set_next_window_position(offset_x, v.menu_bar_height + offset_y)
            togglable_windows[k][0](v)
            included.append(k)

    col_offset_x = 0
    def create_column(names, weights, width, height):
        nonlocal col_offset_x
        assert len(names) == len(weights), 'Wrong number of window weights'
        
        heights = height * np.array(weights) / sum(weights)

        offset_y = 0
        for n, h in zip(names, heights):
            show(n, width, h, col_offset_x, offset_y)
            offset_y += h
        
        col_offset_x += width

    import glfw
    W, H = glfw.get_window_size(v._window)
    
    s = v.ui_scale
    w_left = 400 * s
    w_right = 400 * s
    w_center = W - w_left - w_right

    # left side toolbar
    create_column(
        ['Settings', 'Info', 'Log', 'Conditioning controls'],
        [2, 2, 2, 3],
        width=w_left,
        height=H-v.menu_bar_height
    )

    # center output
    create_column(
        ['Output'],
        [1],
        width=w_center,
        height=H-v.menu_bar_height
    )
    
    # right side toolbar
    h_gt = w_right + 25*s
    h_expl = (2/3)*(H - v.menu_bar_height - h_gt)
    h_saved = (1/3)*(H - v.menu_bar_height - h_gt)
    weights = [h_expl, h_saved] if v.show_ref else [h_expl, h_saved, h_gt]
    names = ['Latent explorer', 'Saved dirs'] if v.show_ref else ['Latent explorer', 'Saved dirs', 'Training data']
    create_column(
        names,
        weights,
        width=w_right,
        height=H-v.menu_bar_height
    )

    show('Latent export', 400*s, 300*s, (W-400*s)//2, (H-300*s)//2)

    # Sanity
    for k in togglable_windows.keys():
        assert k in included or not is_visible(k), f'Window {k} not handled'

def _ui_loop_manual_layout(v):
    for f, active in togglable_windows.values():
        if active:
            f(v)

# UI thread
def ui_loop(v):
    handle_keys(v)
    window_toolbar(v)
    
    if v.auto_layout:
        _ui_loop_auto_layout(v)
    else:
        _ui_loop_manual_layout(v)

    # Slowing down UI speeds up compute 'thread'...thanks Python!
    time.sleep(1/100)

# Dataset viz window compute thread
def dataset_viz_loop(v):
    cache = ParamCache()
    while not v.quit:
        if not is_visible('Training data'):
            time.sleep(1) # window hidden
        elif cache.update(t=v.cond.t_gt, args=v.net.G.init_kwargs, ds=v.net.dataset_name):
            img = get_gt_frame(v, v.cond.t_gt)
            v.upload_image('gt_img', img)
        else:
            time.sleep(1/120) # window in fg, stay snappy

# GAN output compute thread
def compute_loop(v):
    setup_model(v)

    # The state functions are blocking w/ internal loops
    while not v.quit:
        if v.state == State.IDLE:
            state_idle(v)
        elif v.state == State.SWAPPING_MODEL:
            swap_model(v)
            v.state = State.IDLE
        else:
            raise RuntimeError(f'Unknown state: {str(v.state)}')

# Compute PCA in separate process
def decomp_loop(v):
    while not v.quit:
        if not v.lat_expl.pca.comp and not v.lat_expl.pca.stale: # stale => waiting for button press
            cached = comp_path(pkl)

            # Try to load cached
            if cached.is_file():
                with np.load(cached) as d:
                    v.lat_expl.pca.comp = (d['pca_comp'], d['pca_std'])
            else:
                # Use separate process (not thread)
                from multiprocessing import Process, Pipe
                parent_conn, child_conn = Pipe()
                curr_pkl = pkl
                p = Process(target=utils.pca_w_process, args=(child_conn, curr_pkl, 100_000,))
                p.start()
                
                pca_comps = None
                while pca_comps is None:
                    # Abort if changing pickle
                    if v.quit or pkl != curr_pkl:
                        p.terminate()
                        break
                    data = parent_conn.recv()
                    if isinstance(data, str):
                        v.lat_expl.pca.status = data # progress percentage
                    else:
                        pca_comps = data
                        break
                p.join()

                if pca_comps:
                    os.makedirs(cached.parent, exist_ok=True)
                    np.savez(cached, pca_comp=pca_comps[0], pca_std=pca_comps[1])
                    v.lat_expl.pca.comp = pca_comps

        time.sleep(1.0)


#####################
# PROGRAM ENTRY POINT
#####################
    
def run_interactive():
    # For hot-swapping models by drag-and-drop
    import glfw
    glfw.set_drop_callback(v._window, file_drop_callback)

    # Mirror stdout in UI
    def refresh_cb(message):
        v.log_updated = True
    v.logger = utils.Logger(callback=refresh_cb)

    # Disable bilinear interpolation in GL
    v.set_interp_nearest()

    # Initialize any number of threads
    compute_thread = threading.Thread(target=compute_loop, args=[v])
    dataset_viz_thread = threading.Thread(target=dataset_viz_loop, args=[v])
    decomp_thread = threading.Thread(target=decomp_loop, args=[v])
    v.start(ui_loop, (compute_thread, decomp_thread, dataset_viz_thread))

####################
# CMD ARGS AND SETUP
####################

class LatDirs(EasyDict):
    def __init__(self, G):
        self.comp = None
        self.l_start = 1
        self.l_end = utils.parse_n_styles(G)
        self.sliders = np.zeros(utils.parse_z_dims(G))
        self.status = 'Loading...'
        self.stale = False # needs recompute

    def mark_stale(self):
        self.stale = True
        self.comp = None
        self.reset_sliders()

    def reset_sliders(self):
        self.sliders = np.zeros_like(self.sliders)

class PCADirs(LatDirs):
    def __init__(self, G):
        super().__init__(G)
        self.modes = ['IPCA']

class CFFDirs(LatDirs):
    def __init__(self, G):
        super().__init__(G)
        self.modes = ['SVD U', 'SeFa Unscaled', 'SeFa'] #, 'SVD V'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TLGAN visualizer')
    parser.add_argument('network_pickle', type=str, help='Path to model pickle')
    parser.add_argument('--seed', type=int, default=0, help='Initial seed for latent sampling')
    parser.add_argument('--dataset_root', type=str, default=os.environ.get('TLGAN_DATASET_ROOT', str(Path(__file__).parent)), help='Path to datasets (for showing GT frames)')
    parser.add_argument('-b', type=int, default=1, help='Batch size')
    args = parser.parse_args()

    pkl = args.network_pickle

    # State
    v = gl_viewer.viewer('TLGAN')
    v.net = EasyDict()
    data = pickle.load(read_pkl_cached(pkl))
    v.net.G = data['G_ema']
    v.net.D = data['D']
    v.net.dataset_name = None
    v.net.dataset_path = args.dataset_root
    v.net.has_strided_cond = False # only new pickles
    v.cond = EasyDict()
    v.cond.t = 0.5 # global time
    v.cond.t_gt = 0.5 # best-matching global time from per-f overrides
    v.cond.ts = [] # per-layer time:          [[(l_start, l_end), t], ...]
    v.cond.fs = [] # per-layer-and-freq time: [[(l_start, l_end), [t_freq0, t_freq1, ...]], ...]
    v.cond.mode = 'Linear' # 'Fourier': edit freqencies separately
    v.truncation = 1.0
    v.truncation_cutoff = utils.parse_n_styles(v.net.G)
    v.batch_size = args.b
    v.seed = args.seed
    v.frozen_seed = True
    v.frozen_noise = True
    v.gl_linear = False
    v.state = State.LOADING_MODEL
    v.show_ref = False
    v.lat_expl = EasyDict()
    v.lat_expl.pca = PCADirs(v.net.G)
    v.lat_expl.cff = CFFDirs(v.net.G)
    v.lat_expl.mode = v.lat_expl.pca.modes[0]
    v.lat_expl.slider_total = 1.0
    v.lat_expl.saved_dirs_offset = 0
    v.fps = 0
    v.log_updated = False
    v.training_data = None
    v.windows_locked = True
    v.auto_layout = True
    v.menu_bar_height = 0

    # Go fast
    torch.autograd.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True

    # Pseudo-float16 internally on Ampere (on by default)
    # float16:  1|5|10 = 16bit => 5bit e => must rescale
    # bfloat16: 1|8|7  = 16bit => 7bit m => precision issues
    # NV TF32:  1|8|10 = 19bit => 8bit e => no scale issues
    # float32:  1|8|23 = 32bit
    safe_mode = True
    if safe_mode:
        os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    run_interactive()
    print('Done')
