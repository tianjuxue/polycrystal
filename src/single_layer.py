import jax
import jax.numpy as np
import numpy as onp
import os
from src.utils import read_path, obj_to_vtu
from src.arguments import args
from src.allen_cahn import polycrystal_gn, polycrystal_fd, build_graph, phase_field, odeint, explicit_euler


def neper_domain():
    os.system(f'neper -T -n {args.num_grains} -id 1 -regularization 0 -domain "cube({args.domain_length},{args.domain_width},{args.domain_height})" \
                -o data/neper/single_layer/domain -format tess,obj,ori')
    os.system(f'neper -T -loadtess data/neper/single_layer/domain.tess -statcell x,y,z,vol,facelist -statface x,y,z,area')
    os.system(f'neper -M -rcl 1 -elttype hex -faset faces data/neper/single_layer/domain.tess')


def default_initialization(poly_sim):
    num_nodes = len(poly_sim.centroids)
    T = args.T_ambient*np.ones(num_nodes)
    zeta = np.ones(num_nodes)
    eta = np.zeros((num_nodes, args.num_oris))
    eta = eta.at[np.arange(num_nodes), poly_sim.cell_ori_inds].set(1)
    # shape of state: (num_nodes, 1 + 1 + args.num_oris)
    y0 = np.hstack((T[:, None], zeta[:, None], eta))
    melt = np.zeros(len(y0), dtype=bool)
    return y0, melt

 
def simulate(ts, xs, ys, ps, func):
    polycrystal, mesh = func()
    y0, melt = default_initialization(polycrystal)
    graph = build_graph(polycrystal, y0)
    state_rhs = phase_field(graph, polycrystal)
    odeint(polycrystal, mesh, None, explicit_euler, state_rhs, y0, melt, ts, xs, ys, ps)


def run_gn():
    args.case = 'gn_single_layer'
    ts, xs, ys, ps = read_path(f'data/txt/single_track.txt')
    simulate(ts, xs, ys, ps, polycrystal_gn)


def run_fd():
    args.case = 'fd_single_layer'
    ts, xs, ys, ps = read_path(f'data/txt/single_track.txt')
    simulate(ts, xs, ys, ps, polycrystal_fd)


if __name__ == "__main__":
    # neper_domain()
    run_gn()
    run_fd()

