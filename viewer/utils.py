# Copyright (c) 2022 Erik Härkönen, Aalto University
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

from inspect import getmembers, isfunction
import sys
import time
import numpy as np
import torch
import imgui
import cachetools
import contextlib
import pickle
from cachetools.keys import hashkey
from tqdm import tqdm
from io import BytesIO
from pathlib import Path

import sys, os
sys.path += [os.path.abspath(os.path.dirname(__file__) + '/..')]
from dnnlib import EasyDict
from torch_utils.misc import named_params_and_buffers

# Decorator for adding static state to function
def with_state(**kwargs):
    class State(EasyDict):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            setattr(self, 'defaults', kwargs)

        def reset(self):
            for k,v in self.defaults.items():
                self[k] = v

    def decorate(func):
        from functools import partial
        return partial(func, state=State(**kwargs))
        
    return decorate

# Used to detect parameter changes for lazy recomputation
class ParamCache():
    def update(self, **kwargs):
        dirty = False
        for argname, val in kwargs.items():
            # Check pointer, then value
            current = getattr(self, argname, 0)
            if current is not val and pickle.dumps(current) != pickle.dumps(val):
                setattr(self, argname, val)
                dirty = True
        return dirty

# Redirect stdout, stderr to UI
class Logger(object):
    def __init__(self, capacity=20, callback=lambda _ : None):
        self.capacity = capacity
        self.callback = callback
        self.queue = []
        self.str = ''

        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = self
        sys.stderr = self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def flush(self):
        self.stdout.flush()

    def close(self):
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

    # Print calls write for each token
    def write(self, message):
        if isinstance(message, bytes):
            message = message.decode()
        
        # Workaround for a bug in VSCode debugger:
        # sys.stdout.write(''); sys.stdout.flush() => crash
        if len(message) == 0:
            return
        
        self.stdout.write(message)
        self.update_str(message)
        self.callback(message)

    def update_str(self, message):
        # Capacity ignored for now
        self.queue.append(message)
        self.str = ''.join(self.queue)

    def __getattr__(self, attr):
        return getattr(self.stdout, attr)

class Timer():
    def __init__(self, n_avg=1):
        self.len = n_avg
        self.reset()

    def reset(self):
        self.tlast = time.time()
        self.deltas = [0.0] * self.len # circular buffer
        self.idx = 0
    
    def tick(self):
        t = time.time()
        self.deltas[self.idx] = t - self.tlast
        self.idx = (self.idx + 1) % self.len
        self.tlast = t

        return sum(self.deltas) / self.len

# with-block for item id
@contextlib.contextmanager
def imgui_id(id: str):
    imgui.push_id(id)
    yield
    imgui.pop_id()

# with-block for item width
@contextlib.contextmanager
def imgui_item_width(size):
    imgui.push_item_width(size)
    yield
    imgui.pop_item_width()

# Full screen imgui window
def begin_inline(name):
    with imgui.styled(imgui.STYLE_WINDOW_ROUNDING, 0):
        imgui.begin(name,
            flags = \
                imgui.WINDOW_NO_TITLE_BAR |
                imgui.WINDOW_NO_RESIZE |
                imgui.WINDOW_NO_MOVE |
                imgui.WINDOW_NO_COLLAPSE |
                imgui.WINDOW_NO_SCROLLBAR |
                imgui.WINDOW_NO_SAVED_SETTINGS
        )

# Recursive getattr
def rgetattr(obj, key, default=None):
    head = obj
    while '.' in key:
        bot, key = key.split('.', maxsplit=1)
        head = getattr(head, bot, {})
    return getattr(head, key, default)

def dict_diff(d1, d2):
    return _dict_diff_impl(d1, d2, {}, {}, {})

# Compare two dicts, return partition of unique values
def _dict_diff_impl(d1, d2, left, right, changed): # cannot use default args => remembered between calls!
    only_left = set(d1.keys()) - set(d2.keys())
    only_right = set(d2.keys()) - set(d1.keys())
    both = set(d1.keys()).intersection(d2.keys())

    for k in only_left:
        left[k] = d1[k]

    for k in only_right:
        right[k] = d2[k]

    for k in both:
        if isinstance(d1[k], (dict, EasyDict)):
            for d in [left, right, changed]:
                d[k] = {}
            _dict_diff_impl(d1[k], d2[k], left[k], right[k], changed[k])
            for d in [left, right, changed]:
                if d[k] == {}:
                    del d[k]
        elif d1[k] != d2[k]:
            changed[k] = (d1[k], d2[k])

    return (left, right, changed)

# Combo box that returns value, not index
def combo_box_vals(title, values, current, height_in_items=-1, to_str=lambda v: v):
    curr_idx = 0 if current not in values else values.index(current)
    changed, ind = imgui.combo(title, curr_idx, [to_str(v) for v in values], height_in_items)
    return changed, values[ind]

# Int2 slider that prevents overlap
def slider_range(v1, v2, vmin, vmax, push=False, title='', width=0.0):
    imgui.push_item_width(width)
    s, e = imgui.slider_int2(title, v1, v2, vmin, vmax)[1]
    imgui.pop_item_width()

    if push:
        return (min(s, e), max(s, e))
    elif s != v1:
        return (min(s, e), e)
    elif e != v2:
        return (s, max(s, e))
    else:
        return (s, e)

def parse_res(res_str):
    res_confs = {
        '256x256':   (2, 2, 7),
        '384x256':   (3, 2, 7),
        '640x384':   (5, 3, 7),
        '512x512':   (4, 4, 7),
        '512x640':   (4, 5, 7),
        '1024x1024': (4, 4, 8),
        '1280x768':  (5, 3, 8),
    }
    assert res_str in res_confs, f'Unknown resolution {res_str}'
    return res_confs[res_str]

# Shape batch as square if possible
def get_grid_dims(B):
    if B == 0:
        return (0, 0)
    
    S = int(B**0.5 + 0.5)
    while B % S != 0:
        S -= 1
    return (B // S, S) # (W, H)

def reshape_grid_np(img_batch):
    if isinstance(img_batch, list):
        img_batch = np.concatenate(img_batch, axis=0) # along batch dim
    
    B, C, H, W = img_batch.shape
    cols, rows = get_grid_dims(B)

    img_batch = np.reshape(img_batch, [rows, cols, C, H, W])
    img_batch = np.transpose(img_batch, [0, 3, 1, 4, 2])
    img_batch = np.reshape(img_batch, [rows * H, cols * W, C])

    return img_batch

def reshape_grid_torch(img_batch):
    if isinstance(img_batch, list):
        img_batch = torch.cat(img_batch, axis=0) # along batch dim
    
    B, C, H, W = img_batch.shape
    cols, rows = get_grid_dims(B)

    img_batch = img_batch.reshape(rows, cols, C, H, W)
    img_batch = img_batch.permute(0, 3, 1, 4, 2)
    img_batch = img_batch.reshape(rows * H, cols * W, C)

    return img_batch

def reshape_grid(batch):
    return reshape_grid_torch(batch) if torch.is_tensor(batch) else reshape_grid_np(batch)

def sample_seeds(N, base=None):
    if base is None:
        base = np.random.randint(np.iinfo(np.int32).max - N)
    return [(base + s) for s in range(N)]

def sample_latent(B, n_dims=512, seed=None):
    seeds = sample_seeds(B, base=seed)
    return seeds_to_latents(seeds, n_dims)

def seeds_to_latents(seeds, n_dims=512):
    latents = np.zeros((len(seeds), n_dims), dtype=np.float32)
    for i, seed in enumerate(seeds):
        rng = np.random.RandomState(seed)
        latents[i] = rng.standard_normal(n_dims)
    
    return latents

# Map stack of per-frequency ts to single global t
def stack_to_global_t(ts, fs):
    assert isinstance(ts, np.ndarray) and (ts.size % fs.size == 0), \
        'ts must be numpy array of shape (B, n_freq) or (n_freq)'

    ts = ts.reshape(-1, fs.shape[-1])

    B, num_f = ts.shape
    assert num_f == len(fs), 'Len of ts and fs must match'

    T, *ts = ts.T.reshape(num_f, B, 1) # (B, 1) for every frequency
    f0, *fs = fs
    assert f0 < 0.1, 'f0 is not linear'
    assert all(f2 >= f1 for f2,f1 in zip(fs[1:], fs[:-1])), 'fs not sorted'

    for f, t in zip(fs, ts):
        # Remove offset w.r.t frequency,
        # override with provided offset
        T = T - np.fmod(T, 1/f) + np.fmod(t, 1/f)

    return T

def parse_n_styles(net):
    return net.num_ws

def parse_z_dims(net):
    return net.z_dim

def parse_cond_type(net):
    return rgetattr(net, 'init_kwargs.cond_args.type', 'none')

def pca_sanity(X, transformer):
    # Statistics
    #total_var = X.var(axis=0).sum() # total variance
    #mean = X.mean(axis=0, keepdims=True) # mean
    stdev = np.dot(transformer.components_, X.T).std(axis=1) # projected stdev

    # Sort components based on explained variance
    idx = np.argsort(stdev)[::-1]

    # Components should be sorted by default
    assert all(idx[1:] > idx[:-1]), 'PCA produced non-sorted basis'

    # Check orthogonality
    from itertools import combinations
    dotps = [np.dot(*transformer.components_[[i, j]])
        for (i, j) in combinations(range(X.shape[1]), 2)]
    if not np.allclose(dotps, 0, atol=1e-4):
        print('PCA components not orghogonal, max dot', np.abs(dotps).max())

# Wrapper for computing pca in separate process
def pca_w_process(pipe, pkl, N=1_000_000, B=10_000):
    import dnnlib
    import pickle
    
    with dnnlib.util.open_url(pkl) as f:
        G = pickle.load(f)['G_ema']
        G = G.to('cpu') # don't need GPU for just mapping network
    
    comps = pca_w(G, N, B, lambda d : pipe.send(d))
    pipe.send(comps)
    pipe.close()

# Run incremental PCA on W space
def pca_w(G, N=1_000_000, B=10_000, progress_callback=lambda t: None):
    np.random.seed(0)
    torch.random.manual_seed(0)
    
    N = (((N - 1) // B) + 1) * B

    # Result of PCA depends on conditioning?
    if G.mapping.c_dim > 0:
        print('WARNING: PCA dependent on t, fixing to t=0.5')

    # Run PCA
    from sklearn.decomposition import IncrementalPCA
    transformer = IncrementalPCA(150, whiten=False, batch_size=2*G.w_dim)

    G.mapping.num_ws = 1 # don't broadcast result
    
    with torch.no_grad():
        for b in range(0, N, B):
            progress_callback(f'Computing PCA ({100*b//N}%)')
            cs = 0.5 * torch.ones(B, G.mapping.c_dim) # no variation w.r.t. cond
            z = torch.randn(B, G.z_dim)
            w = G.mapping(z, cs, truncation_psi=1.0).reshape(-1, G.w_dim)
            transformer.partial_fit(w.numpy())

    # Check orthogonality etc.
    transformer.components_ = transformer.components_.astype(np.float32)
    stdev = np.sqrt(transformer.explained_variance_).astype(np.float32)
    #pca_sanity(X, transformer)

    return (transformer.components_, stdev)

def cff_max_layer(G):
    return len(_get_cff_layers(G))

def _get_cff_layers(G):
    include = ['affine', 'weight']
    exclude = ['rgb', 'affine_c', 'input']
    
    names = []
    for n, _ in named_params_and_buffers(G):
        incl = all(k in n.lower() for k in include)
        excl = any(k in n.lower() for k in exclude)
        if incl and not excl:
            names.append(n)
    
    return names

# l_start, l_end: zero-based style indices
cff_cache = cachetools.LRUCache(100)
@cachetools.cached(cache=cff_cache, key=lambda G, s, e, m, c: hashkey(s, e, m)) # ignore uncachable vals
def compute_cff(G, l_start=0, l_end=None, mode='SVD U', progress_callback=lambda t: None):
    from scipy.linalg import svd

    progress_callback('Collecting weight matrices')
    names = _get_cff_layers(G)
    n_styles = parse_n_styles(G)

    assert len(names) > 0, 'No modulation weights in model, cannot compute CFF'
    if len(names) != n_styles - 1:
        print(f'WARN (CFF): number of mod_weights ({len(names)}) does not match layer count ({n_styles}) - 1')
    
    # Compute SVD of all chosen layers at once
    # => finds dirs that activate whole range at once
    #   => all layers: close to PCA-W behavior
    #   => single layer: localized changes
    s, e = (l_start, l_end or len(names))
    mats = [t.data.cpu().numpy().T for n,t in named_params_and_buffers(G) if n in names]
    weight_full = np.concatenate(mats[s:e], axis=1).astype(np.float32) # [512, ~5000]
    
    comp, stdev = (None, None)

    progress_callback('Computing decomposition...')

    # Left-singular SVD vectors
    if mode == 'SVD U':
        U, sigma, V = svd(weight_full, lapack_driver='gesvd') # more accurate triangular solver
        comp, stdev = (U.T, np.sqrt(sigma))
    elif mode == 'SVD V':
        # This mode makes no sense, can be e.g. shape [32, 32] for only last layer
        # ...unless applied not to w, but to s (i.e. after affine)?
        U, sigma, V = svd(weight_full, lapack_driver='gesvd')
        comp, stdev = (V.T, np.sqrt(sigma))
    elif mode == 'SeFa Unscaled':
        eigval, comps = np.linalg.eig(weight_full.dot(weight_full.T)) # WW^T = 512x512
        comp, stdev = (comps.T, eigval)
    elif mode == 'SeFa':
        weight_full = weight_full / np.linalg.norm(weight_full, axis=0, keepdims=True)
        eigval, comps = np.linalg.eig(weight_full.dot(weight_full.T)) # WW^T = 512x512
        comp, stdev = (comps.T, eigval)
    else:
        raise RuntimeError('Unknown CFF mode ' + mode)

    # Normalize, just to be sure
    comp /= np.linalg.norm(comp, axis=-1, keepdims=True) # [n_comp, w_dims]
    stdev = np.ones_like(stdev) # use ones instead for now

    progress_callback('')
    return (comp, stdev)

# File copy with progress bar
# For slow network drives etc.
def copy_with_progress(pth_from, pth_to):
    os.makedirs(pth_to.parent, exist_ok=True)
    size = int(os.path.getsize(pth_from))
    fin = open(pth_from, 'rb')
    fout = open(pth_to, 'ab')

    try:
        with tqdm(ncols=80, total=size, bar_format=pth_from.name + ' {l_bar}{bar} | Remaining: {remaining}') as pbar:
            while True:
                buf = fin.read(4*2**20) # 4 MiB
                if len(buf) == 0:
                    break
                fout.write(buf)
                pbar.update(len(buf))
    except Exception as e:
        print(f'File copy failed: {e}')
    finally:
        fin.close()
        fout.close()

# File open with progress bar
# For slow network drives etc.
# Supports context manager
def open_prog(pth, mode):
    size = int(os.path.getsize(pth))
    fin = open(pth, 'rb')

    assert mode == 'rb', 'Only rb supported'
    fout = BytesIO()

    try:
        with tqdm(ncols=80, total=size, bar_format=Path(pth).name + ' {l_bar}{bar}| Remaining: {remaining}') as pbar:
            while True:
                buf = fin.read(4*2**20) # 4 MiB
                if len(buf) == 0:
                    break
                fout.write(buf)
                pbar.update(len(buf))
    except Exception as e:
        print(f'File copy failed: {e}')
    finally:
        fin.close()
        fout.seek(0)

    return fout