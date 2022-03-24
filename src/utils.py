import jax.numpy as np


def unpack_state(state):
    zeta = state[...,  0:1]
    eta = state[..., 1:]
    return zeta, eta