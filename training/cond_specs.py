# Copyright (c) 2022 Erik Härkönen, Aalto University
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

import dnnlib

def_comb = dnnlib.EasyDict(type='mult', degree=-1)
specs = { d['type']: d for d in [
    dict(type='none',     dequant='gauss', noise=0.0,                             dims=0, lr=1, lin_lr=1e-2),
    dict(type='concat',   dequant='gauss', noise=0.0, noise_f_int=[],             dims=1, lr=1, lin_lr=1e-2), # 1 frame overlap in conditioning
    dict(type='fourier',  dequant='gauss', noise=0.0, noise_f_int=[], noise_f=[], dims=2, lr=1, lin_lr=1e-2, f_manual=[], include_lin=True),
    dict(type='f_concat', dequant='gauss', noise=0.0, noise_f_int=[], noise_f=[], dims=2, lr=1, lin_lr=1e-2, f_manual=[], include_lin=True), # fourier features given to concat
]}