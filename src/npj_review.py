import jax
import jax.numpy as np
import numpy as onp
import os
import matplotlib.pyplot as plt
from src.utils import read_path, obj_to_vtu
from src.arguments import args
from src.allen_cahn import polycrystal_gn, polycrystal_fd, build_graph, phase_field, odeint, odeint_no_output, explicit_euler

  
def debug(): 
    neper_mesh = 'debug'
    morpho = 'gg'
    # args.num_grains = 1000
    os.system(f'neper -T -n 10000 -morpho centroidal -morphooptistop itermax=50 -domain "cube({args.domain_length},{args.domain_width},{args.domain_height})" \
                -o data/neper/{neper_mesh}/domain -format tess,obj,ori')
    # os.system(f'neper -T -n 100 -domain "cube({args.domain_length},{args.domain_width},{args.domain_height})" \
    #             -o data/neper/{neper_mesh}/domain -format tess,obj,ori')

    # os.system(f'neper -T -loadtess data/neper/{neper_mesh}/domain.tess -statcell x,y,z,vol,facelist -statface x,y,z,area')
    # os.system(f'neper -M -rcl 1 -elttype hex -faset faces data/neper/{neper_mesh}/domain.tess')
    write_vtu_files(neper_mesh)


def write_vtu_files(neper_mesh):
    args.case = 'gn_' + neper_mesh
    poly_mesh = obj_to_vtu(neper_mesh)
    vtk_folder = f'data/vtk/{args.case}/mesh/'
    if not os.path.exists(vtk_folder):
        os.makedirs(vtk_folder)
    poly_mesh.write(f'data/vtk/{args.case}/mesh/poly_mesh.vtu')


def neper_domain(neper_mesh, morpho): 
    itermax = 50 if morpho == 'centroidal' else 1000000
    # TODO: Will itermax=1000000 cause a bug for voronoi?
    os.system(f'neper -T -n {args.num_grains} -morpho {morpho} -morphooptistop itermax={itermax} -domain "cube({args.domain_length},{args.domain_width},{args.domain_height})" \
                -o data/neper/{neper_mesh}/domain -format tess,obj,ori')
    os.system(f'neper -T -loadtess data/neper/{neper_mesh}/domain.tess -statcell x,y,z,vol,facelist -statface x,y,z,area')
    os.system(f'neper -M -rcl 1 -elttype hex -faset faces data/neper/{neper_mesh}/domain.tess')
    write_vtu_files(neper_mesh)


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

 
def simulate(func, neper_mesh, just_measure_time=False):
    print(f"Running case {args.case}")
    polycrystal, mesh = func(neper_mesh)
    y0, melt = default_initialization(polycrystal)
    graph = build_graph(polycrystal, y0)
    state_rhs = phase_field(graph, polycrystal)
    ts, xs, ys, ps = read_path(f'data/txt/single_track.txt')
    if just_measure_time:
        odeint_no_output(polycrystal, mesh, None, explicit_euler, state_rhs, y0, melt, ts, xs, ys, ps)
    else:
        odeint(polycrystal, mesh, None, explicit_euler, state_rhs, y0, melt, ts, xs, ys, ps)


def run_voronoi():
    neper_mesh = 'npj_review_voronoi'
    # neper_domain(neper_mesh, voronoi)
    args.num_oris = 20
    args.num_grains = 40000

    args.case = 'gn_npj_review_voronoi'
    simulate(polycrystal_gn, neper_mesh)
    # simulate(polycrystal_gn, neper_mesh, True)

    args.case = 'fd_npj_review_voronoi'
    simulate(polycrystal_fd, neper_mesh)
    # simulate(polycrystal_fd, neper_mesh, True)


def run_voronoi_more_oris():
    neper_mesh = 'npj_review_voronoi'
    args.num_oris = 40
    args.num_grains = 40000

    args.case = 'gn_npj_review_voronoi_more_oris'
    # simulate(polycrystal_gn, neper_mesh)
    simulate(polycrystal_gn, neper_mesh, True)

    args.case = 'fd_npj_review_voronoi_more_oris'
    # simulate(polycrystal_fd, neper_mesh)
    simulate(polycrystal_fd, neper_mesh, True)


def run_voronoi_less_oris():
    neper_mesh = 'npj_review_voronoi'
    args.num_oris = 10
    args.num_grains = 40000

    args.case = 'gn_npj_review_voronoi_less_oris'
    # simulate(polycrystal_gn, neper_mesh)
    simulate(polycrystal_gn, neper_mesh, True)

    args.case = 'fd_npj_review_voronoi_less_oris'
    # simulate(polycrystal_fd, neper_mesh)
    simulate(polycrystal_fd, neper_mesh, True)


def run_voronoi_fine():
    neper_mesh = 'npj_review_voronoi_fine'
    args.num_oris = 20
    args.num_grains = 80000
    # neper_domain(neper_mesh, 'voronoi')

    args.case = 'gn_npj_review_voronoi_fine'
    # simulate(polycrystal_gn, neper_mesh)
    simulate(polycrystal_gn, neper_mesh, True)

    args.case = 'fd_npj_review_voronoi_fine'
    # simulate(polycrystal_fd, neper_mesh)
    simulate(polycrystal_fd, neper_mesh, True)


def run_voronoi_coarse():
    neper_mesh = 'npj_review_voronoi_coarse'
    args.num_oris = 20
    args.num_grains = 20000

    # neper_domain(neper_mesh, 'voronoi')

    args.case = 'gn_npj_review_voronoi_coarse'
    # simulate(polycrystal_gn, neper_mesh)
    simulate(polycrystal_gn, neper_mesh, True)

    args.case = 'fd_npj_review_voronoi_coarse'
    # simulate(polycrystal_fd, neper_mesh)
    simulate(polycrystal_fd, neper_mesh, True)


def run_centroidal():
    neper_mesh = 'npj_review_centroidal'
    args.num_oris = 20
    args.num_grains = 40000

    # neper_domain(neper_mesh, 'centroidal')

    args.case = 'gn_npj_review_centroidal'
    # simulate(polycrystal_gn, neper_mesh)
    simulate(polycrystal_gn, neper_mesh, True)

    args.case = 'fd_npj_review_centroidal'
    # simulate(polycrystal_fd, neper_mesh)
    simulate(polycrystal_fd, neper_mesh, True)


def run_voronoi_laser_150():
    neper_mesh = 'npj_review_voronoi'
    args.power = 150.

    args.case = 'gn_npj_review_voronoi_laser_150'
    simulate(polycrystal_gn, neper_mesh)

    args.case = 'fd_npj_review_voronoi_laser_150'
    simulate(polycrystal_fd, neper_mesh)


def run_voronoi_laser_250():
    neper_mesh = 'npj_review_voronoi'
    args.power = 250.

    args.case = 'gn_npj_review_voronoi_laser_250'
    simulate(polycrystal_gn, neper_mesh)

    args.case = 'fd_npj_review_voronoi_laser_250'
    simulate(polycrystal_fd, neper_mesh)


def run_voronoi_laser_100():
    neper_mesh = 'npj_review_voronoi'
    args.power = 100.

    args.case = 'gn_npj_review_voronoi_laser_100'
    simulate(polycrystal_gn, neper_mesh)

    args.case = 'fd_npj_review_voronoi_laser_100'
    simulate(polycrystal_fd, neper_mesh)


def npj_review_initial_size_distribution():
    args.case = 'none'
    poly_voronoi, _ = polycrystal_gn('npj_review_voronoi')
    voronoi_vols = poly_voronoi.volumes*1e9
    poly_centroidal, _ = polycrystal_gn('npj_review_centroidal')
    centroidal_vols = poly_centroidal.volumes*1e9

    colors = ['red', 'blue']
    labels = ['Voronoi', 'Centroidal']
    fig = plt.figure(figsize=(8, 6))
    plt.hist([voronoi_vols, centroidal_vols], bins=onp.linspace(0, 3*1e3, 13), color=colors, label=labels)
    plt.legend(fontsize=20, frameon=False) 
    plt.xlabel(r'Grain volume [$\mu$m$^3$]', fontsize=20)
    plt.ylabel(r'Count', fontsize=20)
    plt.tick_params(labelsize=18)
    plt.grid(False)
    plt.savefig(f'data/pdf/initial_size_distribution.pdf', bbox_inches='tight')


if __name__ == "__main__":
    # run_voronoi()
    # run_voronoi_more_oris()
    # run_voronoi_less_oris()
    # run_voronoi_fine()
    # run_voronoi_coarse()
    # run_centroidal()
    # run_voronoi_small_laser()
    # run_voronoi_big_laser()
    # run_voronoi_laser_100()

    npj_review_initial_size_distribution()
    # plt.show()