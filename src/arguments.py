import numpy as onp
import jax
import jax.numpy as np
import argparse
import sys
import numpy as onp
import matplotlib.pyplot as plt
from jax.config import config
import torch

torch.manual_seed(0)

# Set numpy printing format
onp.random.seed(0)
onp.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True)
onp.set_printoptions(precision=10)

# np.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True)
# np.set_printoptions(precision=5)

# Manage arguments
parser = argparse.ArgumentParser()
parser.add_argument('--num_oris', type=int, default=20)
parser.add_argument('--num_grains', type=int, default=20000)
parser.add_argument('--dim', type=int, default=3)
parser.add_argument('--domain_height', type=float, default=0.1)
parser.add_argument('--domain_width', type=float, default=0.2)
parser.add_argument('--domain_length', type=float, default=1.)
parser.add_argument('--T_melt', type=float, default=1500.)
args = parser.parse_args()

# Latex style plot
# plt.rcParams.update({
#     "text.latex.preamble": r"\usepackage{amsmath}",
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica"]})


