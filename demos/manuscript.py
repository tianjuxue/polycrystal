import jax.numpy as np
import jax
import numpy as onp
import os
import meshio
import matplotlib.pyplot as plt
from src.arguments import args
from src.utils import *
from src.allen_cahn import *


def generate_demo_graph():
    args.num_grains = 10
    args.domain_length = 1.
    args.domain_width = 1.
    args.domain_height = 1.
    # os.system(f'neper -T -n {args.num_grains}  -domain "cube({args.domain_length},{args.domain_width},{args.domain_height})" \
    #     -o data/neper/graph/domain -format tess,obj')

    os.system(f'neper -T -n {args.num_grains} -periodic 1 -domain "cube({args.domain_length},{args.domain_width},{args.domain_height})" \
        -o data/neper/graph/domain -format tess,obj')

    os.system(f'neper -T -loadtess data/neper/graph/domain.tess -statcell x,y,z,vol,facelist -statface x,y,z,area')   
    mesh = obj_to_vtu(domain_name='graph')
    num = len(mesh.cells_dict['polyhedron'])
    mesh.cell_data['color'] = [onp.hstack((onp.random.uniform(low=0., high=1., size=(num, 3)), onp.ones((num, 1))))]
    mesh.cell_data['id'] = [onp.arange(num)]
    mesh.write(f'data/vtk/graph/demo.vtu')

    # poly, _ = polycrystal_gn(domain_name='graph')
    # print(poly.edges)


def fd_gn_compare():
    # args.num_grains = 40000
    # args.domain_width = 0.4
    # args.r_beam = 0.1
    # args.power = 200
    ts, xs, ys, ps = read_path(f'data/txt/single_track.txt')

    args.case = 'gn'
    simulate(ts, xs, ys, ps, polycrystal_gn)

    args.case = 'fd'
    simulate(ts, xs, ys, ps, polycrystal_fd)


if __name__ == "__main__":
    # generate_demo_graph()
    fd_gn_compare()
    plt.show()
