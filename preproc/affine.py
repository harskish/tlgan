# Copyright (c) 2022 Erik Härkönen, Aalto University
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

import numpy as np

def WarpMatrix(ps):
    ps = ps.reshape(-1, 6, 1)
    
    return np.concatenate([
        1.0+ps[:, 0],     ps[:, 2], ps[:, 4],
            ps[:, 1], 1.0+ps[:, 3], ps[:, 5]
    ], axis=-1).reshape(-1, 2, 3)

def PfromWarpMatrix(m):
    assert m.ndim == 2, 'Not tested for batched inputs'
    m = m.reshape(-1, 2, 3)

    p = np.array([
        m[:, 0, 0] - 1, m[:, 1, 0], m[:, 0, 1], m[:, 1, 1] - 1, m[:, 0, 2], m[:, 1, 2]
    ]).reshape(-1, 6, 1)
    
    return p

# Decomposition that allows interpolation
# https://math.stackexchange.com/a/3521141
def AffineDecomp(M):
    assert M.ndim == 3 and M.shape[-2:] == (2, 3) # (N, 2, 3)

    T = M[:, :, 2] # translation
    A = M[:, :, 0:2] # rotation
    A11 = A[:, 0, 0]; A12 = A[:, 0, 1]; A21 = A[:, 1, 0]; A22 = A[:, 1, 1]
    sx = np.sqrt(A11**2 + A21**2)
    th = np.arctan2(A21, A11)
    msy = A12*np.cos(th) + A22*np.sin(th)
    with np.errstate(invalid='ignore'): # both branches evaluated => ignore zero division warning
        sy = np.where(np.abs(th) < 1e-8, (A22-msy*np.sin(th))/np.cos(th), (msy*np.cos(th)-A12)/np.sin(th))
    assert np.isfinite(sy).all(), 'Inf or nan in affine decomp'
    m = msy / sy
    total = np.stack([T[:, 0], T[:, 1], sx, sy, m, th], axis=-1)

    return total

def DecompToAffine(Tx, Ty, sx, sy, m, th):
    rot = np.array([
        np.cos(th), -np.sin(th),
        np.sin(th),  np.cos(th)
    ]).reshape((2, 2))
    shear = np.array([
        1, m,
        0, 1
    ]).reshape((2, 2))
    scale = np.array([
        sx, 0,
        0, sy
    ]).reshape((2, 2))
    M = rot @ shear @ scale
    A = np.array([
        M[0, 0], M[0, 1], Tx,
        M[1, 0], M[1, 1], Ty,
    ]).reshape((2, 3))
    
    return A