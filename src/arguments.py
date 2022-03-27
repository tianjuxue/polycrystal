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
parser.add_argument('--domain_height', type=float, help='Unit: mm', default=0.1)
parser.add_argument('--domain_width', type=float, help='Unit: mm', default=0.2)
parser.add_argument('--domain_length', type=float, help='Unit: mm', default=1.)
parser.add_argument('--T_melt', type=float, help='Unit: K', default=1700.)
parser.add_argument('--rho', type=float, help='Unit: kg/mm^3', default=8.e-6)
parser.add_argument('--c_h', type=float, help='Unit: J/(kg*K)', default=770.)
parser.add_argument('--power', type=float, help='Unit: W', default=100.)
parser.add_argument('--r_beam', type=float, help='Unit: mm', default=0.05)
parser.add_argument('--h_depth', type=float, help='Unit: mm', default=0.1)
parser.add_argument('--emissivity', type=float, help='Unit:', default=0.2)
parser.add_argument('--SB_constant', type=float, help='Unit: W/(mm^2*K*4)', default=5.67e-14)
parser.add_argument('--kappa', type=float, help='Unit: W/(mm*K)', default=1.5e-2)
parser.add_argument('--gas_const', type=float, help='Unit: J/(Mol*K)', default=8.3)
parser.add_argument('--Qg', type=float, help='Unit: J/Mol', default=1.4e5)
parser.add_argument('--L0', type=float, help='Unit: mm^4/(J*s)', default=3.5e12)




args = parser.parse_args()

# Latex style plot
# plt.rcParams.update({
#     "text.latex.preamble": r"\usepackage{amsmath}",
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica"]})


