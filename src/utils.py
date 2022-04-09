import jax.numpy as np
import meshio
import jax
import jax.numpy as np
import numpy as onp
import orix
import time
import os
import matplotlib.pyplot as plt
from orix import plot, sampling
from orix.crystal_map import Phase
from orix.quaternion import Orientation, symmetry
from orix.vector import Vector3d
from src.arguments import args
from sklearn.decomposition import PCA


def unpack_state(state):
    T = state[..., 0:1]
    zeta = state[...,  1:2]
    eta = state[..., 2:]
    return T, zeta, eta


def get_unique_ori_colors():
    # unique_oris = R.random(args.num_oris, random_state=0).as_euler('zxz', degrees=True)
    ori2 = Orientation.random(args.num_oris)

    v = Vector3d((0, 1, 0))

    ipfkey = plot.IPFColorKeyTSL(symmetry.Oh, v)
    ori2.symmetry = symmetry.Oh
    rgb_z = ipfkey.orientation2color(ori2)
    ori2.scatter("ipf", c=rgb_z, direction=ipfkey.direction)

    onp.save(f"data/numpy/quat.npy", ori2.data)

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


def make_video():
    # The command -pix_fmt yuv420p is to ensure preview of video on Mac OS is enabled
    # https://apple.stackexchange.com/questions/166553/why-wont-video-from-ffmpeg-show-in-quicktime-imovie-or-quick-preview
    # The command -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" is to solve the following "not-divisible-by-2" problem
    # https://stackoverflow.com/questions/20847674/ffmpeg-libx264-height-not-divisible-by-2
    # -y means always overwrite
    os.system('ffmpeg -y -framerate 10 -i data/png/tmp/u.%04d.png -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" data/mp4/test.mp4')


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
    mesh = meshio.Mesh(points, cells)
    return mesh


def vtk_convert_from_server():
    args.case = 'fd'
    def vtk_convert_from_server_helper(number):
        filepath = f'data/vtk/sols/{args.case}/u{number}.vtu'
        mesh = meshio.read(filepath)
        mesh.write(filepath)

    vtk_convert_from_server_helper(0)
    vtk_convert_from_server_helper(120)


def walltime(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        time_elapsed = end_time - start_time
        print(f"Time elapsed {time_elapsed} on platform {jax.lib.xla_bridge.get_backend().platform}") 
        return time_elapsed
    return wrapper


def compute_stats():
    def compute_stats_helper():
        # edges = onp.load(f"data/numpy/{case}/edges.npy")
        # volumes = onp.load(f"data/numpy/{case}/vols.npy")
        # centroids = onp.load(f"data/numpy/{case}/centroids.npy")
        # cell_ori_inds = onp.load(f"data/numpy/{case}/cell_ori_inds_{tick}.npy")
        # melt = onp.load(f"data/numpy/{case}/melt_{tick}.npy")
        # num_nodes = len(volumes)

        edges = onp.load(f"data/numpy/fd/info/edges.npy")
        volumes = onp.load(f"data/numpy/fd/info/vols.npy")
        centroids = onp.load(f"data/numpy/fd/info/centroids.npy")

        if case == 'fd':
            cell_ori_inds =onp.load(f"data/numpy/{case}/sols/cell_ori_inds_{step:03d}.npy")
            melt = onp.load(f"data/numpy/{case}/sols/melt_{step:03d}.npy")           
        else:
            grain_oris_inds = onp.load(f"data/numpy/{case}/sols/cell_ori_inds_{step:03d}.npy")
            grain_melt = onp.load(f"data/numpy/{case}/sols/melt_{step:03d}.npy")
            cell_grain_inds = onp.load(f"data/numpy/fd/info/cell_grain_inds.npy")
            cell_ori_inds = onp.take(grain_oris_inds, cell_grain_inds - 1, axis=0)
            melt = onp.take(grain_melt, cell_grain_inds - 1, axis=0)


        num_nodes = len(volumes)
        edges_in_order = [[] for _ in range(num_nodes)]

        print(f"Re-ordering edges...")
        for edge in edges:
            node1 = edge[0]
            node2 = edge[1]
            edges_in_order[node1].append(node2)
            edges_in_order[node2].append(node1)

        print(f"BFS...")
        visited = onp.zeros(num_nodes)
        grains = [[] for _ in range(args.num_oris)]
        for i in range(len(visited)):
            if visited[i] == 0 and melt[i]:
                oris_index = cell_ori_inds[i]
                grains[oris_index].append([])
                queue = [i]
                visited[i] = 1
                while queue:
                    s = queue.pop(0) 
                    grains[oris_index][-1].append(s)
                    connected_nodes = edges_in_order[s]
                    for cn in connected_nodes:
                        if visited[cn] == 0 and cell_ori_inds[cn] == oris_index and melt[cn]:
                            queue.append(cn)
                            visited[cn] = 1

        def compute_aspect_ratios(grain):
            vol = onp.sum(onp.array([volumes[g] for g in grain]))
            if len(grain) < 3:
                return 1., vol

            directions = onp.array([centroids[g] for g in grain])
            weighted_directions = onp.array([volumes[g]*centroids[g] for g in grain])
            vols = onp.array([volumes[g] for g in grain])

            # weighted_directions = weighted_directions - onp.mean(weighted_directions, axis=0)[None, :]
            pca.fit(weighted_directions)
            components = pca.components_
            ev = pca.explained_variance_
            lengths = onp.sqrt(ev)
            aspect_ratio = 2*lengths[0]/(lengths[1] + lengths[2])
            return aspect_ratio, vol

        pca = PCA(n_components=3)
        print(f"Compute vols...")
        grain_vols = []
        aspect_ratios = []
        for i in range(len(grains)):
            grains_oris = grains[i] 
            for j in range(len(grains_oris)):
                grain = grains_oris[j]
                aspect_ratio, vol = compute_aspect_ratios(grain)
                aspect_ratios.append(aspect_ratio)
                grain_vols.append(vol)

        grain_vols = onp.array(grain_vols)
        aspect_ratios = onp.array(aspect_ratios)
        onp.save(f"data/numpy/{case}/post-processing/post_vols_{step:03d}.npy", grain_vols)
        onp.save(f"data/numpy/{case}/post-processing/post_aspect_ratios_{step:03d}.npy", aspect_ratios)

    cases = ['gn', 'fd']
    steps = [20]
    for case in cases:
        for step in steps:
            compute_stats_helper()

def hist_plot():
    step = 20

    fd_vols = onp.load(f"data/numpy/fd/post-processing/post_vols_{step:03d}.npy")
    gn_vols = onp.load(f"data/numpy/gn/post-processing/post_vols_{step:03d}.npy")
    fd_aspect_ratios = onp.load(f"data/numpy/fd/post-processing/post_aspect_ratios_{step:03d}.npy")
    gn_aspect_ratios = onp.load(f"data/numpy/gn/post-processing/post_aspect_ratios_{step:03d}.npy")

    # print(onp.mean()
    # print(onp.mean(gn_vols[gn_vols > 1e-7]))
    # val = onp.min(gn_vols)
    # val = 1e-7

    fd_volumes = onp.load(f"data/numpy/fd/info/vols.npy")
    domain_vol = args.domain_length*args.domain_width*args.domain_height
    avg_cell_vol = domain_vol / len(fd_volumes)
    avg_grain_vol = domain_vol / args.num_grains
    print(f"avg fd cell_vol = {avg_cell_vol}")
    print(f"avg grain vol = {avg_grain_vol}")

    # val = 1e-7
    val = 1e-6

    # print(fd_vols[fd_vols < val])
    # print(gn_vols[gn_vols < val])
    print("\n")
    print(f"fd mean vol = {onp.mean(fd_vols[fd_vols > val])}")
    print(f"gn mean vol = {onp.mean(gn_vols[gn_vols > val])}")
    print(f"fd num of grains = {onp.sum(fd_vols > val)}")
    print(f"gn num of grains = {onp.sum(gn_vols > val)}")
    
    print("\n")
    print(f"fd mean aspect_ratio = {onp.mean(fd_aspect_ratios[fd_vols > val])}")
    print(f"gn mean aspect_ratio = {onp.mean(gn_aspect_ratios[gn_vols > val])}")
 
 
    colors = ['blue', 'red']
    labels = ['DNS', 'ROM']

    fig = plt.figure()
    plt.hist([fd_vols[fd_vols > val], gn_vols[gn_vols > val]], color=colors, bins=onp.linspace(0., 1e-5, 6), label=labels)
    plt.legend(fontsize=14, frameon=False) 
    plt.xlabel(f'Grain volume [mm$^3$]', fontsize=16)
    plt.ylabel(f'Count', fontsize=16)
    plt.tick_params(labelsize=14)

    fig = plt.figure()
    plt.hist([fd_aspect_ratios[fd_vols > val], gn_aspect_ratios[gn_vols > val]], color=colors, bins=onp.linspace(1, 4, 13), label=labels)
    plt.legend(fontsize=14, frameon=False) 
    plt.xlabel(f'Aspect ratio', fontsize=16)
    plt.ylabel(f'Count', fontsize=16)
    plt.tick_params(labelsize=14)


if __name__ == "__main__":
    # vtk_convert_from_server()
    # get_unique_ori_colors()
    make_video()
    # compute_stats()
    # hist_plot()
    plt.show()
