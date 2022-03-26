import jax.numpy as np
import meshio
import jax
import jax.numpy as np
import numpy as onp
import orix
import matplotlib.pyplot as plt
from orix import plot, sampling
from orix.crystal_map import Phase
from orix.quaternion import Orientation, symmetry
from orix.vector import Vector3d
from src.arguments import args


def unpack_state(state):
    zeta = state[...,  0:1]
    eta = state[..., 1:]
    return zeta, eta


def get_unique_ori_colors():
    print(f"Debug info: args.num_oris = {args.num_oris}")
    # unique_oris = R.random(args.num_oris, random_state=0).as_euler('zxz', degrees=True)
    ori2 = Orientation.random(args.num_oris)
    ipfkey = plot.IPFColorKeyTSL(symmetry.Oh)
    ori2.symmetry = symmetry.Oh
    rgb_z = ipfkey.orientation2color(ori2)
    # ori2.scatter("ipf", c=rgb_z, direction=ipfkey.direction)
    return rgb_z


# def orix_exp():
#     # We'll want our plots to look a bit larger than the default size
#     new_params = {
#         "figure.facecolor": "w",
#         "figure.figsize": (20, 7),
#         "lines.markersize": 10,
#         "font.size": 15,
#         "axes.grid": True,
#     }
#     plt.rcParams.update(new_params)
#     pg = symmetry.Oh
#     ori2 = Orientation.random(1000)
#     ipfkey = plot.IPFColorKeyTSL(pg)
#     ori2.symmetry = ipfkey.symmetry
#     rgb_z = ipfkey.orientation2color(ori2)
#     ori2.scatter("ipf", c=rgb_z, direction=ipfkey.direction)



def obj_to_vtu():
    filepath = f'data/neper/domain.obj'
    file = open(filepath, 'r')
    lines = file.readlines()
    points = []
    cells_inds = []

    for i, line in enumerate(lines):
        l = line.split()
        if l[0] == 'v':
            points.append([float(l[1]), float(l[2]), float(l[3])])
        if l[0] == 'g':
            cells_inds.append([])
        if l[0] == 'f':
            cells_inds[-1].append([int(pt_ind) - 1 for pt_ind in l[1:]])

    cells = [('polyhedron', cells_inds)]

    # cell_data = {'u': [onp.ones(len(cells_inds), dtype=onp.float32)]}
    # cell_data = {'u': [onp.random.rand(len(cells_inds), 3)]}
    # cell_data = {'u': [onp.hstack((onp.zeros((len(cells_inds), 2)), onp.ones((len(cells_inds), 1))))]}
    # mesh = meshio.Mesh(points, cells, cell_data=cell_data)

    mesh = meshio.Mesh(points, cells)
    return mesh


def vtk_convert_from_server():
    filepath = f'data/vtk/sols/u_final.vtk'
    mesh = meshio.read(filepath)
    mesh.write(filepath)


if __name__ == "__main__":
    get_unique_ori_colors()
    plt.show()
