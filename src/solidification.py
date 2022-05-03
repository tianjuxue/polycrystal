import os
import jax
import jax.numpy as np
import numpy as onp
from src.arguments import args
from src.allen_cahn import polycrystal_fd, build_graph, phase_field, odeint, explicit_euler
from src.utils import unpack_state, walltime, read_path


def set_params():
    args.case = 'fd_solidification'
    args.num_grains = 10000
    args.domain_length = 1.
    args.domain_width = 0.01
    args.domain_height = 1.
    args.write_sol_interval = 1000

    # The following parameter controls the anisotropy level, see Yan paper Eq. (12)
    # If set to be zero, then isotropic grain growth is considered.
    # Default value is used if not explict set here.
    # args.anisotropy = 0.


def neper_domain():
    set_params()
    os.system(f'neper -T -n {args.num_grains} -id 1 -regularization 0 -domain "cube({args.domain_length},{args.domain_width},{args.domain_height})" \
                -o data/neper/solidification/domain -format tess,obj,ori')
    os.system(f'neper -T -loadtess data/neper/solidification/domain.tess -statcell x,y,z,vol,facelist -statface x,y,z,area')
    os.system(f'neper -M -rcl 1 -elttype hex -faset faces data/neper/solidification/domain.tess')


def solidification_initialization(poly_sim):
    num_nodes = len(poly_sim.centroids)
    T = args.T_ambient*np.ones(num_nodes)
    zeta = np.zeros(num_nodes)
    eta = np.zeros((num_nodes, args.num_oris))
    # shape of state: (num_nodes, 1 + 1 + args.num_oris)
    eta = eta.at[np.arange(num_nodes), poly_sim.cell_ori_inds].set(1)
    y0 = np.hstack((T[:, None], zeta[:, None], eta))
    melt = np.zeros(len(y0), dtype=bool)
    return y0, melt


def get_T(centroids, t):
    '''
    Given spatial coordinates and t, we prescribe the value of T.
    '''
    z = centroids[:, 2]
    vel = 200.
    thermal_grad = 500.
    cooling_rate = thermal_grad * vel
    t_total = args.domain_height / vel
    T = args.T_melt + thermal_grad * z - cooling_rate * t
    return T[:, None]


@jax.jit
def overwrite_T(y, centroids, t):
    '''
    We overwrite T if T is prescribed.
    '''
    T, zeta, eta = unpack_state(y) 
    T = get_T(centroids, t)
    return np.hstack((T, zeta, eta))


def run():
    set_params()
    ts, xs, ys, ps = read_path(f'data/txt/solidification.txt')
    polycrystal, mesh = polycrystal_fd('solidification')
    y0, melt = solidification_initialization(polycrystal)
    graph = build_graph(polycrystal, y0)
    state_rhs = phase_field(graph, polycrystal)
    odeint(polycrystal, mesh, None, explicit_euler, state_rhs, y0, melt, ts, xs, ys, ps, overwrite_T)


if __name__ == "__main__":
    # neper_domain()
    run()
