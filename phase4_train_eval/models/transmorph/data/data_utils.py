import random
import pickle
import numpy as np
import torch

M = 2 ** 32 - 1


def init_fn(worker):
    seed = torch.LongTensor(1).random_().item()
    seed = (seed + worker) % M
    np.random.seed(seed)
    random.seed(seed)


def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def npzload(fname):
    """Load volume (and optionally seg) from .npz file."""
    d = np.load(fname)
    vol = d['vol']
    seg = d['seg'] if 'seg' in d else None
    return vol, seg
