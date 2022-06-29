# Copyright (c) 2022 Erik Härkönen, Aalto University
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

"""
Split video into frames for preprocessing.
HW-accelerated video decoding unreliable to use directly...
"""

import argparse
import os
from pathlib import Path
from tools import shell_cmd, get_n_frames

parser = argparse.ArgumentParser(description='Video frame splitter')
parser.add_argument('path', type=str, help='Path to input video')
args = parser.parse_args()

outdir = Path(args.path).with_suffix('.frames')
os.makedirs(outdir, exist_ok=True)

# Split into frames
n_frames = get_n_frames(args.path)
print('Generating', n_frames, 'frames')
frame_fmt = outdir / 'img%05d.jpg'
shell_cmd(f'ffmpeg -i {args.path} -qscale:v 2 -start_number 0 {frame_fmt}') # zero-based indexing, jpeg quality 2 (range 2-31)

print('Saved', n_frames, 'frames to', str(outdir))