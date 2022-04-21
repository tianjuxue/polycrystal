import jax.numpy as np
import jax
import numpy as onp
import orix
import meshio
import pickle
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
    onp.random.seed(0)
    ori2 = Orientation.random(args.num_oris)
    v = Vector3d((0, 1, 0))
    ipfkey = plot.IPFColorKeyTSL(symmetry.Oh, v)
    ori2.symmetry = symmetry.Oh
    rgb_y = ipfkey.orientation2color(ori2)
    ori2.scatter("ipf", c=rgb_y, direction=ipfkey.direction)
    onp.save(f"data/numpy/quat.npy", ori2.data)
    return rgb_y


def ipf_logo():
    new_params = {
        "figure.facecolor": "w",
        "figure.figsize": (6, 3),
        "lines.markersize": 10,
        "font.size": 25,
        "axes.grid": True,
    }
    plt.rcParams.update(new_params)
    plot.IPFColorKeyTSL(symmetry.Oh).plot()
    plt.savefig(f'data/pdf/ipf.pdf', bbox_inches='tight')


def make_video():
    # The command -pix_fmt yuv420p is to ensure preview of video on Mac OS is enabled
    # https://apple.stackexchange.com/questions/166553/why-wont-video-from-ffmpeg-show-in-quicktime-imovie-or-quick-preview
    # The command -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" is to solve the following "not-divisible-by-2" problem
    # https://stackoverflow.com/questions/20847674/ffmpeg-libx264-height-not-divisible-by-2
    # -y means always overwrite
    os.system('ffmpeg -y -framerate 10 -i data/png/tmp/u.%04d.png -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" data/mp4/test.mp4')


def obj_to_vtu(domain_name='domain'):
    filepath=f'data/neper/{domain_name}/domain.obj'
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


def walltime(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        time_elapsed = end_time - start_time
        print(f"Time elapsed {time_elapsed} on platform {jax.lib.xla_bridge.get_backend().platform}") 
        return time_elapsed
    return wrapper


def read_path(path):
    path_info = onp.loadtxt(path)
    traveled_time =  path_info[:, 0]
    x_corners = path_info[:, 1]
    y_corners = path_info[:, 2]
    power_control = path_info[:-1, 3]
    ts, xs, ys, ps = [], [], [], []
    for i in range(len(traveled_time) - 1):
        ts_seg = onp.arange(traveled_time[i], traveled_time[i + 1], args.dt)
        xs_seg = onp.linspace(x_corners[i], x_corners[i + 1], len(ts_seg))
        ys_seg = onp.linspace(y_corners[i], y_corners[i + 1], len(ts_seg))
        ps_seg = onp.linspace(power_control[i], power_control[i], len(ts_seg))
        ts.append(ts_seg)
        xs.append(xs_seg)
        ys.append(ys_seg)
        ps.append(ps_seg)

    ts, xs, ys, ps = onp.hstack(ts), onp.hstack(xs), onp.hstack(ys), onp.hstack(ps)  
    print(f"Total number of time steps = {len(ts)}")
    return ts, xs, ys, ps


def fd_helper(num_fd_nodes):
    domain_vol = args.domain_length*args.domain_width*args.domain_height
    avg_cell_vol = domain_vol / num_fd_nodes
    avg_cell_len = avg_cell_vol**(1/3)
    avg_grain_vol = domain_vol / args.num_grains
    print(f"avg fd cell_vol = {avg_cell_vol}")
    print(f"avg grain vol = {avg_grain_vol}")
    return avg_cell_vol, avg_cell_len


def compute_stats():
    edges = onp.load(f"data/numpy/fd/info/edges.npy")
    volumes = onp.load(f"data/numpy/fd/info/vols.npy")
    centroids = onp.load(f"data/numpy/fd/info/centroids.npy")
    cell_grain_inds = onp.load(f"data/numpy/fd/info/cell_grain_inds.npy")
    num_fd_nodes = len(volumes)
    avg_cell_vol, avg_cell_len = fd_helper(num_fd_nodes)

    def compute_stats_helper():
        if case == 'fd':
            cell_ori_inds =onp.load(f"data/numpy/{case}/sols/cell_ori_inds_{step:03d}.npy")
            melt = onp.load(f"data/numpy/{case}/sols/melt_{step:03d}.npy")        
            T = onp.load(f"data/numpy/{case}/sols/T_{step:03d}.npy")   
            zeta = onp.load(f"data/numpy/{case}/sols/zeta_{step:03d}.npy") 
        else:
            grain_oris_inds = onp.load(f"data/numpy/{case}/sols/cell_ori_inds_{step:03d}.npy")
            grain_melt = onp.load(f"data/numpy/{case}/sols/melt_{step:03d}.npy")
            grain_T = onp.load(f"data/numpy/{case}/sols/T_{step:03d}.npy")
            zeta_T = onp.load(f"data/numpy/{case}/sols/zeta_{step:03d}.npy") 
            cell_ori_inds = onp.take(grain_oris_inds, cell_grain_inds, axis=0)
            melt = onp.take(grain_melt, cell_grain_inds, axis=0)
            T = onp.take(grain_T, cell_grain_inds, axis=0)
            zeta = onp.take(zeta_T, cell_grain_inds, axis=0)

        return T, zeta, melt, cell_ori_inds

    def process_T():
        sampling_depth = 5
        sampling_width = 5
        avg_length = 8
        sampling_section = sampling_depth*sampling_width*2

        inds = onp.argwhere((centroids[:, 2] > args.domain_height - sampling_depth*avg_cell_len) & 
                            (centroids[:, 2] < args.domain_height) &
                            (centroids[:, 1] > args.domain_width/2 - sampling_width*avg_cell_len) & 
                            (centroids[:, 1] < args.domain_width/2 + sampling_width*avg_cell_len))[:, 0]

        T_sampled = T[inds].reshape(sampling_section, -1)
        T_sampled_len = T_sampled.shape[1]
        T_sampled = T_sampled[:, :T_sampled_len//avg_length*avg_length].T
        T_sampled = T_sampled.reshape(-1, sampling_section*avg_length)
        T_sampled = onp.mean(T_sampled, axis=1)

        return T_sampled

    def process_zeta():
        inds_melt_pool = onp.argwhere(zeta < 0.5)[:, 0]

        if len(inds_melt_pool) == 0:
            return onp.zeros(4)

        centroids_melt_pool = onp.take(centroids, inds_melt_pool, axis=0)
        length_melt_pool = onp.max(centroids_melt_pool[:, 0]) - onp.min(centroids_melt_pool[:, 0])
        width_melt_pool = onp.max(centroids_melt_pool[:, 1]) - onp.min(centroids_melt_pool[:, 1])
        height_melt_pool = onp.max(centroids_melt_pool[:, 2]) - onp.min(centroids_melt_pool[:, 2])
        volume_melt_pool = avg_cell_vol*len(inds_melt_pool)
        characteristics = onp.array([length_melt_pool, width_melt_pool, height_melt_pool, volume_melt_pool])

        return characteristics

    def process_eta():
        edges_in_order = [[] for _ in range(num_fd_nodes)]

        print(f"Re-ordering edges...")
        for edge in edges:
            node1 = edge[0]
            node2 = edge[1]
            edges_in_order[node1].append(node2)
            edges_in_order[node2].append(node1)

        print(f"BFS...")
        visited = onp.zeros(num_fd_nodes)
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

        return [grain_vols, aspect_ratios]
 
    # cases = ['gn', 'fd']
    # steps = [20]
    cases = ['gn', 'fd']
    steps = [i for i in range(31)]
    for case in cases:
        T_collect = []
        zeta_collect = []
        eta_collect = []
        for step in steps:
            print(f"step = {step}, case = {case}")
            T, zeta, melt, cell_ori_inds = compute_stats_helper()
            T_results = process_T()
            zeta_results = process_zeta()
            eta_results = process_eta()
            T_collect.append(T_results)
            zeta_collect.append(zeta_results)
            eta_collect.append(eta_results)
        onp.save(f"data/numpy/{case}/post-processing/T_collect.npy", onp.array(T_collect))
        onp.save(f"data/numpy/{case}/post-processing/zeta_collect.npy", onp.array(zeta_collect))
        onp.save(f"data/numpy/{case}/post-processing/eta_collect.npy", onp.array(eta_collect, dtype=object))


def produce_figures():
    ts, xs, ys, ps = read_path(f'data/txt/single_track.txt')
    ts = ts[::args.write_sol_interval]*1e6

    def T_plot():
        T_results_fd = onp.load(f"data/numpy/fd/post-processing/T_collect.npy")
        T_results_gn = onp.load(f"data/numpy/gn/post-processing/T_collect.npy")

        step = 12
        T_select_fd = T_results_fd[step]
        T_select_gn = T_results_gn[step]
        x = onp.linspace(0., args.domain_length, len(T_select_fd))*1e3

        fig = plt.figure(figsize=(8, 6))
        plt.plot(x, T_select_fd, label='DNS', color='blue', marker='o', markersize=8, linestyle="-", linewidth=2)
        plt.plot(x, T_select_gn, label='PIGN', color='red', marker='o', markersize=8, linestyle="-", linewidth=2)
        plt.xlabel(r'x-axis [$\mu$m]', fontsize=20)
        plt.ylabel(r'Temperature [K]', fontsize=20)
        plt.tick_params(labelsize=18)
        plt.legend(fontsize=20, frameon=False)
        plt.savefig(f'data/pdf/T_scanning_line.pdf', bbox_inches='tight')

        ind_T = T_results_fd.shape[1]//2
        fig = plt.figure(figsize=(8, 6))
        plt.plot(ts, T_results_fd[:, ind_T], label='DNS', color='blue', marker='o', markersize=8, linestyle="-", linewidth=2)
        plt.plot(ts, T_results_gn[:, ind_T], label='PIGN', color='red', marker='o', markersize=8, linestyle="-", linewidth=2)
        plt.xlabel(r'Time [$\mu$s]', fontsize=20)
        plt.ylabel(r'Temperature [K]', fontsize=20)
        plt.tick_params(labelsize=18)
        plt.legend(fontsize=20, frameon=False)
        plt.savefig(f'data/pdf/T_center.pdf', bbox_inches='tight')


    def zeta_plot():
        zeta_results_fd = onp.load(f"data/numpy/fd/post-processing/zeta_collect.npy")
        zeta_results_gn = onp.load(f"data/numpy/gn/post-processing/zeta_collect.npy")
        labels = ['Melt pool length [mm]', 'Melt pool width [mm]', 'Melt pool height [mm]', 'Melt pool volume [mm$^3$]']
        names = ['melt_pool_length', 'melt_pool_width', 'melt_pool_height', 'melt_pool_volume']
        for i in range(4):
            fig = plt.figure(figsize=(8, 6))
            plt.plot(ts, zeta_results_fd[:, i], label='DNS', color='blue', marker='o', markersize=8, linestyle="-", linewidth=2)
            plt.plot(ts, zeta_results_gn[:, i], label='PIGN', color='red', marker='o', markersize=8, linestyle="-", linewidth=2)
            plt.xlabel(r'Time [$\mu$s]', fontsize=20)
            plt.ylabel(labels[i], fontsize=20)
            plt.tick_params(labelsize=18)
            plt.legend(fontsize=20, frameon=False)
            plt.savefig(f'data/pdf/{names[i]}.pdf', bbox_inches='tight')


    def eta_plot():
        eta_results_fd = onp.load(f"data/numpy/fd/post-processing/eta_collect.npy", allow_pickle=True)
        eta_results_gn = onp.load(f"data/numpy/gn/post-processing/eta_collect.npy", allow_pickle=True)

        val = 1e-7
        # val = 1e-6

        def eta_helper(eta_results):
            vols_filtered = []
            aspect_ratios_filtered = []
            num_vols = []
            avg_vol = []
            for item in eta_results:
                grain_vols, aspect_ratios = item
                grain_vols = onp.array(grain_vols)
                inds = onp.argwhere(grain_vols > val)[:, 0]
                grain_vols = grain_vols[inds]*1e9
                aspect_ratios = onp.array(aspect_ratios)
                aspect_ratios = aspect_ratios[inds]
                num_vols.append(len(grain_vols))
                avg_vol.append(onp.mean(grain_vols))
                vols_filtered.append(grain_vols)
                aspect_ratios_filtered.append(aspect_ratios)
            return num_vols, avg_vol, vols_filtered, aspect_ratios_filtered

        num_vols_fd, avg_vol_fd, vols_filtered_fd, aspect_ratios_filtered_fd = eta_helper(eta_results_fd)
        num_vols_gn, avg_vol_gn, vols_filtered_gn, aspect_ratios_filtered_gn = eta_helper(eta_results_gn)

        fig = plt.figure(figsize=(8, 6))
        plt.plot(ts, num_vols_fd, label='DNS', color='blue', marker='o', markersize=8, linestyle="-", linewidth=2)
        plt.plot(ts, num_vols_gn, label='PIGN', color='red', marker='o', markersize=8, linestyle="-", linewidth=2)
        plt.xlabel(r'Time [$\mu$s]', fontsize=20)
        plt.ylabel(r'Number of grains', fontsize=20)
        plt.tick_params(labelsize=18)
        plt.legend(fontsize=20, frameon=False)
        plt.savefig(f'data/pdf/num_grains.pdf', bbox_inches='tight')

        fig = plt.figure(figsize=(8, 6))
        plt.plot(ts, avg_vol_fd, label='DNS', color='blue', marker='o', markersize=8, linestyle="-", linewidth=2)
        plt.plot(ts, avg_vol_gn, label='PIGN', color='red', marker='o', markersize=8, linestyle="-", linewidth=2)
        plt.xlabel(r'Time [$\mu$s]', fontsize=20)
        plt.ylabel(r'Average grain volume [$\mu$m$^3$]', fontsize=20)
        plt.tick_params(labelsize=18)
        plt.legend(fontsize=20, frameon=False)
        plt.savefig(f'data/pdf/grain_vol.pdf', bbox_inches='tight')

        step = 30
        assert len(aspect_ratios_filtered_fd) == 31
        assert len(vols_filtered_fd) == len(aspect_ratios_filtered_gn)

        fd_vols = vols_filtered_fd[step]
        gn_vols = vols_filtered_gn[step]
        fd_aspect_ratios = aspect_ratios_filtered_fd[step]
        gn_aspect_ratios = aspect_ratios_filtered_gn[step]

        print("\n")
        print(f"fd mean vol = {onp.mean(fd_vols)}")
        print(f"gn mean vol = {onp.mean(gn_vols)}")

        print("\n")
        print(f"fd mean aspect_ratio = {onp.mean(fd_aspect_ratios)}")
        print(f"gn mean aspect_ratio = {onp.mean(gn_aspect_ratios)}")

        colors = ['blue', 'red']
        labels = ['DNS', 'PIGN']

        fig = plt.figure(figsize=(8, 6))
        plt.hist([fd_vols, gn_vols], color=colors, bins=onp.linspace(0., 1e4, 6), label=labels)
        plt.legend(fontsize=20, frameon=False) 
        plt.xlabel(f'Grain volume [mm$^3$]', fontsize=20)
        plt.ylabel(f'Count', fontsize=20)
        plt.tick_params(labelsize=18)
        plt.savefig(f'data/pdf/vol_distribution.pdf', bbox_inches='tight')

        fig = plt.figure(figsize=(8, 6))
        plt.hist([fd_aspect_ratios, gn_aspect_ratios], color=colors, bins=onp.linspace(1, 4, 13), label=labels)
        plt.legend(fontsize=20, frameon=False) 
        plt.xlabel(f'Aspect ratio', fontsize=20)
        plt.ylabel(f'Count', fontsize=20)
        plt.tick_params(labelsize=18)
        plt.savefig(f'data/pdf/aspect_distribution.pdf', bbox_inches='tight')

    T_plot()
    zeta_plot()
    eta_plot()


if __name__ == "__main__":
    # vtk_convert_from_server()
    # get_unique_ori_colors()
    ipf_logo()
    # make_video()
    # compute_stats()
    # produce_figures()
    # post_results()
    # plt.show()
 