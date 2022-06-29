# Copyright (c) 2022 Erik Härkönen, Aalto University
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

from collections import defaultdict
from io import BytesIO
import shutil
import os
import numpy as np
import torch
from PIL import Image
from pathlib import Path
import subprocess
from numbers import Integral
from packaging import version
import zipfile
from tqdm import trange
import imageio

import sys
sys.path += [str(Path(__file__).parents[1])]
from ext import resize_right # type: ignore
from ext.get_image_size import get_image_size_from_bytesio

class StrictDataclass:
    # Check for missing type annotations (break by-value comparisons)
    # OK:  val1: int = 5
    # OK:  val1: int = None
    # ERR: val1 = 5
    
    # Should not get called if decorator is present
    def __init__(self):
        if not hasattr(self, '__dataclass_fields__'):
            raise RuntimeError('[ERR] @dataclass decorator missing in subclass')
    
    # Should only get called if decorator is present
    def __post_init__(self):
        for name in dir(self):
            if name.startswith('__'):
                continue
            if name not in self.__dataclass_fields__:
                raise RuntimeError(f'[ERR] Unannotated field: {name}')

# Simulate ZipFile interface for folders
class DirHandle:
    def __init__(self, path):
        self.path = Path(path)

    def namelist(self):
        names = list(self.path.glob('**/*.jpg')) + list(self.path.glob('**/*.png'))
        return [str(p) for p in sorted(names)]
    
    def read(self, name):
        return Path(name).read_bytes()

class VideoFrames:
    def __init__(self, path, torch=False, nvjpeg=True, to_byte=True, batched_resize=False, exceptions=False, out_res='max'):
        self.path = Path(path)
        self.names = []
        self.handles = []
        self.end_idx = [] # ending idx for every handle
        self.exceptions = exceptions
        
        if zipfile.is_zipfile(self.path):
            self._add_zip(self.path)
        elif self.path.is_dir():
            # Add zips from directory
            zips = sorted(list(self.path.glob('*.zip')))
            for zip in zips:
                self._add_zip(zip)
            
            # If no zips: add dir itself
            if len(zips) == 0:
                self._add_dir(self.path)
        else:
            print(f'Input {path} is neither an archive nor a directory')

        self.n_frames = len(self.names)
        assert self.n_frames > 0, f'No frames found in {path}'
        #self.frame_prefix = names[0].name[:-9] # .jpg + 5 digits

        self.error_color = (0, 0, 0)
        self.to_byte = to_byte
        self.torch = torch
        self.nvjpeg = self.torch and nvjpeg
        self.batched_resize = batched_resize
        self.versions_checked = False
        
        # Create static resize layers per inres-outres pair (for torch tensors)
        self.resize_layers = {}

        if out_res in ['max', 'min']:
            inds = np.unique(np.linspace(0, self.n_frames - 1, 1_000).astype(int)) # sparse sampling
            sizes = []

            # PIL: 6.3s / 10kimg, from_bytesio: 3.8s / 10kimg
            from ext.get_image_size import UnknownImageFormat
            for i in inds:
                try:
                    bytes = self._get_bytes(i)
                    sizes.append(get_image_size_from_bytesio(BytesIO(bytes), len(bytes)))
                except UnknownImageFormat as e:
                    print(f'Could not get size of file {i} ({self.names[i]})')

            sizes = np.array(sizes)
            selector = np.argmax if out_res == 'max' else np.argmin
            self.out_res = tuple(sizes[selector(sizes.prod(axis=-1)), :].tolist())
        elif out_res == 'orig':
            self.out_res = None
        elif isinstance(out_res, tuple):
            self.out_res = out_res # (W, H)
        else:
            raise RuntimeError('Unknown resoution mode')

    def _add_zip(self, path):
        handle = zipfile.ZipFile(path, 'r')
        assert handle.compression == zipfile.ZIP_STORED and handle._seekable, \
            'Archive compressed or not seekable'
        self.handles.append(handle)
        names = [f for f in handle.namelist() if f.split('.')[-1].lower() in ['jpg', 'png']]
        self.names += sorted(names)
        self.end_idx.append(len(self.names))
        print(f'Added {path.name} ({len(self.names)//1000}k total)')

    def _add_dir(self, path):
        handle = DirHandle(path)
        self.handles.append(handle)
        self.names += handle.namelist()
        self.end_idx.append(len(self.names))
        print(f'Added {path.name} ({len(self.names)//1000}k total)')

    def _assert_versions(self):
        if self.torch and not self.versions_checked:
            self.versions_checked = True

            import torch, torchvision
            
            # Make sure all DLLs are visible
            # Same logic is performed by pycuda.driver
            if hasattr(os, 'add_dll_directory'):
                # Py3.8+ on Windows, https://github.com/inducer/pycuda/issues/213
                import shutil, warnings
                cuda_path_spec = os.environ.get('CUDA_PATH_V{}_{}'.format(*torch.version.cuda.split('.')))
                cuda_path_gen = os.environ.get("CUDA_PATH")
                nvcc_path = shutil.which('nvcc.exe')
                if cuda_path_spec is not None:
                    os.add_dll_directory(os.path.join(cuda_path_spec, "bin"))
                elif cuda_path_gen is not None:
                    os.add_dll_directory(os.path.join(cuda_path_gen, "bin"))
                elif nvcc_path is not None:
                    os.add_dll_directory(os.path.dirname(nvcc_path))
                else:
                    warnings.warn(
                        "Unable to discover CUDA installation directory "
                        "while attempting to add it to Python's DLL path. "
                        "Either set the 'CUDA_PATH' environment variable "
                        "or ensure that 'nvcc.exe' is on the path."
                    )

            if 11000 <= torchvision.version.cuda < 11060:
                print('NVJPEG memory leak in CUDA 11.0 - 11.5, disabling GPU decoding...')
                print('More info: https://github.com/pytorch/vision/issues/4378#issuecomment-1044957322')
                raise AttributeError('NVJPEG not available')
            
            if version.parse(torchvision.__version__) < version.parse("0.10.0a0"):
                print('GPU decoding requires PyTorch 1.9+, torchvision >= 0.10.0 built from source with NVJPEG support')
                raise AttributeError('NVJPEG not available')

    def show(self, i: int):
        Image.open(self._get_bytes_io(i)).show()

    def _get_bytes(self, i: int):
        handle_id = 0
        while self.end_idx[handle_id] <= i:
            handle_id += 1
        return self.handles[handle_id].read(self.names[i])

    def _get_bytes_io(self, i: int):
        return BytesIO(self._get_bytes(i))

    def _get_filename(self, i: int):
        return self.names[i]

    def _cast_torch(self, arr):
        if self.to_byte:
            return arr.clip(0, 255).to(torch.uint8)
        else:
            return arr.float().clip(0, 255) # in range [~0.0, ~255.0] -> [0.0, 255.0]

    def _cast_np(self, arr):
        if self.to_byte:
            return np.uint8(np.clip(arr, 0, 255))
        else:
            return np.clip(np.float32(arr), 0, 255) # in range [0.0, 255.0]

    def _cast(self, arr):
        return self._cast_torch(arr) if self.torch else self._cast_np(arr)

    def _resize(self, arr):
        if self.out_res is None:
            return arr
        
        if self.torch:
            # Use nn.Module which computes weights only once
            key = (tuple(arr.shape[-3:]), tuple(self.out_res)) # batch size can be ignored
            if key not in self.resize_layers:
                layer = resize_right.ResizeLayer(arr.shape, out_shape=(self.out_res[1], self.out_res[0], 3),
                    interp_method=resize_right.interp_methods.lanczos3, antialiasing=True, device=torch.device('cuda'))
                self.resize_layers[key] = layer
            
            return self.resize_layers[key](arr)
        else:
            # Use dynamic mode
            B = [] if arr.ndim == 3 else [arr.shape[0]]
            return resize_right.resize(arr, out_shape=B+[self.out_res[1], self.out_res[0], 3],
                interp_method=resize_right.interp_methods.lanczos3, antialiasing=True)

    # Decode, cast and resize on CPU
    def _decode_CPU(self, i):
        return np.array(Image.open(self._get_bytes_io(i)).convert("RGB"))

    # Decode on CPU, cast and resize on GPU
    def _decode_CPU_GPU(self, i):
        img = Image.open(self._get_bytes_io(i)).convert("RGB")
        img = torch.tensor(np.array(img), dtype=torch.uint8).cuda() # [H, W, 3]
        return img

    # Decode, cast and resize on GPU
    def _decode_GPU(self, i):
        self._assert_versions()

        if not self.names[i].lower().endswith('.jpg'):
            return self._decode_CPU_GPU(i)
        else:
            import torchvision # might be called from new thread => import again
            data = torch.tensor(bytearray(self._get_bytes(i)), dtype=torch.uint8)
            img = torchvision.io.image.decode_jpeg(data, device='cuda',
                mode=torchvision.io.image.ImageReadMode.RGB) # decoded image in on GPU; [3, H, W] as uint8
            
            return img.permute((1, 2, 0))

    # In case of decoding failure
    def _error_img(self):
        if self.exceptions:
            raise RuntimeError('Decoding failed')
        else:
            img = np.tile(np.array(self.error_color), (self.out_res[1], self.out_res[0], 1)) # magenta
            return torch.from_numpy(img).cuda() if self.torch else img

    def _get_frame_single(self, i: int):
        assert isinstance(i, Integral), 'Only integer indexing supported'
        
        # Fast path
        if self.nvjpeg and self.torch:
            try:
                return self._decode_GPU(i)
            except AttributeError as e:
                self.nvjpeg = False # CUDA / Torchvision version incompatible
            except RuntimeError as e:
                print(f'\nERROR: NJPEG decoding failed for idx {i} ({self._get_filename(i)}):\n{e}\n')
                if str(e) == 'No such operator image::read_file':
                    print('Make sure torchvision is imported right before calling "torchvision.io.image.read_file".\n' \
                        + 'Also only sems to work from within a separate thread...?!')
        
        # NVJpeg unavailable or disabled
        try:
            return self._decode_CPU_GPU(i) if self.torch else self._decode_CPU(i)
        except Exception as e:
            print('CPU decode failed:', e)
            return self._error_img()

    # Zero-based global indexing before trimming
    # Resize all of same size on single batch
    def __getitem__(self, key):
        if isinstance(key, slice) and self.batched_resize:
            # Batched resize: faster, but uses more memory (scales with input size)
            assert self.out_res, 'Cannot use batched resize with varying output resolution'
            fw = torch if self.torch else np
            frames = [self._get_frame_single(i) for i in range(self.n_frames)[key]] # decoded individually
            shapes = [list(f.shape) for f in frames]
            unique_shapes = np.unique(shapes, axis=0).tolist()
            
            out_shape = [len(frames), self.out_res[1], self.out_res[0], 3]
            if unique_shapes == [out_shape[1:]]:
                return self._cast(fw.stack(frames, 0))

            resized = torch.zeros(out_shape, device='cuda') if self.torch else np.zeros(out_shape, dtype=np.float32)
            for shape in unique_shapes:
                inds = np.argwhere([s == shape for s in shapes]).flatten()
                batch = fw.stack([frames[i] for i in inds], 0)
                resized[inds] = self._resize(batch)

            return self._cast(resized)
        elif isinstance(key, slice):
            frames = [self._resize(self._get_frame_single(i)) for i in range(self.n_frames)[key]]
            return self._cast(torch.stack(frames, 0) if self.torch else np.stack(frames, 0))
        else:
            img = self._get_frame_single(key)
            return self._cast(self._resize(img))

def shell_cmd(cmd):
    return os.popen(cmd).read().strip()

def get_timings(path):
    out = shell_cmd(f"ffprobe -loglevel error -select_streams v:0 \
        -show_entries packet=pts_time,flags -of csv=print_section=0 {path}")

    # Frames labeled K_ are keyframes
    frames = [l.split(',') for l in out.splitlines()]
    return frames

def get_n_frames(path):
    cmd = f'ffprobe -v error -select_streams v:0 -count_packets \
        -show_entries stream=nb_read_packets -of csv=p=0 {path}'
    ret = subprocess.check_output(cmd, shell=True)
    return int(ret)

# inputs: filenames of numpy arrays
# crop: 'W:H:X:Y'
def _make_video(basedir, i_start, i_end, outname, target_fps, target_len_s, out_width=None, crop='in_w:in_h:0:0', format=None):
    vid = VideoFrames(basedir, torch=False, to_byte=True, out_res='max')

    if out_width == None:
        out_width = vid.out_res[0]
    else:
        w, h = vid.out_res
        vid.out_res = [out_width, int(h*(out_width / w))]

    # Support ratios
    if isinstance(i_start, float):
        i_start = int(i_start * vid.n_frames)
    if isinstance(i_end, float):
        i_end = max(1, int(i_end * vid.n_frames))

    assert vid.n_frames > 0
    assert 0 <= i_start < i_end <= vid.n_frames, 'Invalid frame range'
    inputs = list(range(i_start, i_end))

    shutil.rmtree('frames_gif_tmp', ignore_errors=True)
    os.makedirs('frames_gif_tmp')
    
    target_len = int(np.ceil(target_len_s * target_fps))
    n_input = len(inputs)

    skip = 1
    if n_input < target_len:
        # Not enough frames: adjust fps
        target_fps = n_input / target_len_s
    else:
        # Enough frames: adjust skip
        skip = n_input // target_len

    chosen = [inputs[i] for i in range(0, n_input, skip)][:target_len]

    # More parameters:
    # http://blog.pkh.me/p/21-high-quality-gif-with-ffmpeg.html

    outname = Path(outname).with_suffix(f'.{format}')
    pattern = os.path.abspath('./frames_gif_tmp/img%05d.jpg')
    palette = os.path.abspath('./frames_gif_tmp/palette.png')

    if format == 'gif':
        for i in trange(len(chosen), desc='Exporting frames'):
            img_idx = chosen[i]
            out = f'frames_gif_tmp/img{i:05d}.jpg'
            Image.fromarray(vid[img_idx]).convert('RGB').save(out, quality=95)

        palette_cmd = 'ffmpeg -hide_banner -loglevel error -framerate {} -i "{}" -vf crop={},scale={}:-1:flags=lanczos,palettegen {} -y'.format(target_fps, pattern, crop, out_width, palette)
        gif_cmd = 'ffmpeg -hide_banner -loglevel error -framerate {} -i "{}" -i {} -filter_complex "crop={},scale={}:-1:flags=lanczos[x];[x][1:v] paletteuse" {} -y'.format(target_fps, pattern, palette, crop, out_width, outname)
        os.system(palette_cmd)
        os.system(gif_cmd)
    elif format == 'mp4':
        assert crop == 'in_w:in_h:0:0', 'Streaming mp4 writer doesn\'t support crop'
        video = imageio.get_writer(outname, mode='I', fps=target_fps, codec='libx264', bitrate='7M')
        for i in range(len(chosen)): #, desc='Creating video'):
            img_idx = chosen[i]
            video.append_data(vid[img_idx])
        video.close()
    else:
        raise RuntimeError(f'Unknown output format {format}')
    
    size_mb = outname.stat().st_size / 1e6
    print('Created', outname.absolute(), f'({len(chosen) / target_fps}s, {target_fps:.2f}fps, {size_mb:.1f}MB)')

    shutil.rmtree('frames_gif_tmp')


def make_mp4(*args, **kwargs):
    return _make_video(*args, **kwargs, format='mp4')

def make_gif(*args, **kwargs):
    return _make_video(*args, **kwargs, format='gif')