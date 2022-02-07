import jax.numpy as np


def unpack_state(state):
    T = state[..., 0:1]
    zeta = state[...,  1:2]
    eta = state[..., 2:]
    return T, zeta, eta