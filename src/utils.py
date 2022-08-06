'''
The file produces figures in the manuscript.
It also has some post-processing functions.
'''
import jax.numpy as np
import jax
import numpy as onp
import orix
import meshio
import pickle
import time
import os
import glob
import matplotlib.pyplot as plt
from orix import plot, sampling
from orix.crystal_map import Phase
from orix.quaternion import Orientation, symmetry
from orix.vector import Vector3d
from src.arguments import args
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R
from src.fit_ellipsoid import EllipsoidTool
 

# Latex style plot
plt.rcParams.update({
    "text.latex.preamble": r"\usepackage{amsmath}",
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})


def unpack_state(state):
    T = state[..., 0:1]
    zeta = state[...,  1:2]
    eta = state[..., 2:]
    return T, zeta, eta


def get_unique_ori_colors():
    onp.random.seed(1)

    if args.case == 'fd_solidification':

        # axes = onp.array([[1., 0., 0.], 
        #                   [1., 1., 0.],
        #                   [1., 1., 1.],
        #                   [1., 1., 0.],
        #                   [1., 0., 0.], 
        #                   [1., -1., 0.]])
        # angles = onp.array([0., 
        #                     onp.pi/8,
        #                     onp.pi/4,
        #                     onp.pi/4, 
        #                     onp.pi/4,
        #                     onp.pi/2 - onp.arccos(onp.sqrt(2)/onp.sqrt(3))])

        axes = onp.array([[1., 0., 0.], 
                          [1., 0., 0.], 
                          [1., -1., 0.]])
        angles = onp.array([0., 
                            onp.pi/4,
                            onp.pi/2 - onp.arccos(onp.sqrt(2)/onp.sqrt(3))])

        args.num_oris = len(axes)
        ori2 = Orientation.from_axes_angles(axes, angles)
    else:
        ori2 = Orientation.random(args.num_oris)        

    vx = Vector3d((1, 0, 0))
    vy = Vector3d((0, 1, 0))
    vz = Vector3d((0, 0, 1))
    ipfkey_x = plot.IPFColorKeyTSL(symmetry.Oh, vx)
    rgb_x = ipfkey_x.orientation2color(ori2)
    ipfkey_y = plot.IPFColorKeyTSL(symmetry.Oh, vy)
    rgb_y = ipfkey_y.orientation2color(ori2)
    ipfkey_z = plot.IPFColorKeyTSL(symmetry.Oh, vz)
    rgb_z = ipfkey_z.orientation2color(ori2)
    rgb = onp.stack((rgb_x, rgb_y, rgb_z))

    onp.save(f"data/numpy/quat_{args.num_oris:03d}.npy", ori2.data)
    dx = onp.array([1., 0., 0.])
    dy = onp.array([0., 1., 0.])
    dz = onp.array([0., 0., 1.])
    scipy_quat = onp.concatenate((ori2.data[:, 1:], ori2.data[:, :1]), axis=1)
    r = R.from_quat(scipy_quat)
    grain_directions = onp.stack((r.apply(dx), r.apply(dy), r.apply(dz)))

    save_ipf = False
    if save_ipf:
        # Plot IPF for those orientations
        new_params = {
            "figure.facecolor": "w",
            "figure.figsize": (6, 3),
            "lines.markersize": 10,
            "font.size": 20,
            "axes.grid": True,
        }
        plt.rcParams.update(new_params)
        ori2.symmetry = symmetry.Oh
        ori2.scatter("ipf", c=rgb_x, direction=ipfkey_x.direction)
        # plt.savefig(f'data/pdf/ipf_x.pdf', bbox_inches='tight')
        ori2.scatter("ipf", c=rgb_y, direction=ipfkey_y.direction)
        # plt.savefig(f'data/pdf/ipf_y.pdf', bbox_inches='tight')
        ori2.scatter("ipf", c=rgb_z, direction=ipfkey_z.direction)
        # plt.savefig(f'data/pdf/ipf_z.pdf', bbox_inches='tight')

    return rgb, grain_directions


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
    plt.savefig(f'data/pdf/ipf_legend.pdf', bbox_inches='tight')


def generate_demo_graph():
    '''
    Produce the grain graph in Fig. 1 in the manuscript
    '''
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


def make_video():
    # The command -pix_fmt yuv420p is to ensure preview of video on Mac OS is enabled
    # https://apple.stackexchange.com/questions/166553/why-wont-video-from-ffmpeg-show-in-quicktime-imovie-or-quick-preview
    # The command -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" is to solve the following "not-divisible-by-2" problem
    # https://stackoverflow.com/questions/20847674/ffmpeg-libx264-height-not-divisible-by-2
    # -y means always overwrite
    os.system('ffmpeg -y -framerate 10 -i data/png/tmp/u.%04d.png -pix_fmt yuv420p -vf "crop=trunc(iw/2)*2:trunc(ih/2)*2" data/mp4/test.mp4')


def obj_to_vtu(domain_name):
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
    def wrapper(*list_args, **keyword_args):
        start_time = time.time()
        return_values = func(*list_args, **keyword_args)
        end_time = time.time()
        time_elapsed = end_time - start_time
        platform = jax.lib.xla_bridge.get_backend().platform
        print(f"Time elapsed {time_elapsed} on platform {platform}") 
        with open(f'data/txt/walltime_{platform}_{args.case}_{args.layer:03d}.txt', 'w') as f:
            f.write(f'{start_time}, {end_time}, {time_elapsed}\n')
        return return_values
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


def get_edges_and_face_in_order(edges, face_areas, num_graph_nodes):
    edges_in_order = [[] for _ in range(num_graph_nodes)]
    face_areas_in_order = [[] for _ in range(num_graph_nodes)]

    assert len(edges) == len(face_areas)

    print(f"Re-ordering edges and face_areas...")
    for i, edge in enumerate(edges):
        node1 = edge[0]
        node2 = edge[1]
        edges_in_order[node1].append(node2)
        edges_in_order[node2].append(node1)  
        face_areas_in_order[node1].append(face_areas[i])
        face_areas_in_order[node2].append(face_areas[i])

    return edges_in_order, face_areas_in_order


def get_edges_in_order(edges, num_graph_nodes):
    edges_in_order = [[] for _ in range(num_graph_nodes)]
    print(f"Re-ordering edges...")
    for i, edge in enumerate(edges):
        node1 = edge[0]
        node2 = edge[1]
        edges_in_order[node1].append(node2)
        edges_in_order[node2].append(node1)  
    return edges_in_order


def BFS(edges_in_order, melt, cell_ori_inds, combined=True):
    num_graph_nodes = len(melt)
    print(f"BFS...")
    visited = onp.zeros(num_graph_nodes)
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

    grains_combined = []
    for i in range(len(grains)):
        grains_oris = grains[i] 
        for j in range(len(grains_oris)):
            grains_combined.append(grains_oris[j])

    if combined:
        return grains_combined
    else:
        return grains


def get_aspect_ratio_inputs_single_track(grains_combined, volumes, centroids):
    grain_vols = []
    grain_centroids = []
    for i in range(len(grains_combined)):
        grain = grains_combined[i]
        grain_vol = onp.array([volumes[g] for g in grain])
        grain_centroid = onp.take(centroids, grain, axis=0)
        assert grain_centroid.shape == (len(grain_vol), 3)
        grain_vols.append(grain_vol)
        grain_centroids.append(grain_centroid)

    return grain_vols, grain_centroids


def compute_aspect_ratios_and_vols(grain_vols, grain_centroids):
    pca = PCA(n_components=3)
    print(f"Call compute_aspect_ratios_and_vols")
    grain_sum_vols = []
    grain_sum_aspect_ratios = []

    for i in range(len(grain_vols)):
        grain_vol = grain_vols[i]
        sum_vol = onp.sum(grain_vol)
     
        if len(grain_vol) < 3:
            grain_sum_aspect_ratios.append(1.)
        else:
            directions = grain_centroids[i]
            weighted_directions = directions * grain_vol[:, None]
            # weighted_directions = weighted_directions - onp.mean(weighted_directions, axis=0)[None, :]
            pca.fit(weighted_directions)
            components = pca.components_
            ev = pca.explained_variance_
            lengths = onp.sqrt(ev)
            aspect_ratio = 2*lengths[0]/(lengths[1] + lengths[2])
            grain_sum_aspect_ratios.append(aspect_ratio)

        grain_sum_vols.append(sum_vol)
  
    return [grain_sum_vols, grain_sum_aspect_ratios]


def compute_stats_multi_layer():
    args.case = 'gn_multi_layer_scan_1'
    args.num_total_layers = 10

    grain_oris_inds = []
    melt = []
    for i in range(args.num_total_layers):
        grain_ori_inds_bottom = onp.load(f"data/numpy/{args.case}/sols/layer_{i + 1:03d}/cell_ori_inds_bottom.npy")
        melt_final_bottom = onp.load(f'data/numpy/{args.case}/sols/layer_{i + 1:03d}/melt_final_bottom.npy')
        assert grain_ori_inds_bottom.shape == melt_final_bottom.shape
        grain_oris_inds.append(grain_ori_inds_bottom)
        melt.append(melt_final_bottom)

    melt = onp.hstack(melt)
    grain_oris_inds = onp.hstack(grain_oris_inds)

    edges = onp.load(f"data/numpy/{args.case}/info/edges.npy")
    volumes = onp.load(f"data/numpy/{args.case}/info/vols.npy")
    centroids = onp.load(f"data/numpy/{args.case}/info/centroids.npy")

    assert melt.shape == volumes.shape

    grains_combined = BFS(edges, melt, grain_oris_inds)

    grain_sum_vols = []
    for i in range(len(grains_combined)):
        grain = grains_combined[i]
        grain_vol = onp.sum(onp.array([volumes[g] for g in grain]))
        grain_sum_vols.append(grain_vol)

    grain_sum_vols = onp.array(grain_sum_vols)

    val = 0.
    inds = onp.argwhere(grain_sum_vols > val)[:, 0]
    grain_sum_vols = grain_sum_vols[inds]*1e9
    # grain_sum_aspect_ratios = grain_sum_aspect_ratios[inds]

    onp.save(f"data/numpy/{args.case}/post-processing/grain_sum_vols.npy", grain_sum_vols)

    return grain_sum_vols


def produce_figures_multi_layer():
    grain_sum_vols_scan1 = onp.load(f"data/numpy/gn_multi_layer_scan_1/post-processing/grain_sum_vols.npy")
    grain_sum_vols_scan2 = onp.load(f"data/numpy/gn_multi_layer_scan_2/post-processing/grain_sum_vols.npy")

    colors = ['blue', 'red']
    labels = ['Scan 1', 'Scan 2']

    print(f"total vol of scan 1 = {onp.sum(grain_sum_vols_scan1)}, mean = {onp.mean(grain_sum_vols_scan1)}")
    print(f"total vol of scan 2 = {onp.sum(grain_sum_vols_scan2)}, mean = {onp.mean(grain_sum_vols_scan2)}")
    print(f"total number of grains for scan 1 {len(grain_sum_vols_scan1)}")
    print(f"total number of grains for scan 2 {len(grain_sum_vols_scan2)}")

    log_grain_sum_vols_scan1 = onp.log10(grain_sum_vols_scan1)
    log_grain_sum_vols_scan2 = onp.log10(grain_sum_vols_scan2)

    bins = onp.linspace(1e2, 1e7, 25)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))

    fig = plt.figure(figsize=(8, 6))
    plt.hist([grain_sum_vols_scan1, grain_sum_vols_scan2], bins=logbins, color=colors, label=labels)
 
    plt.xscale('log')
    plt.xlabel(r'Grain volume [$\mu$m$^3$]', fontsize=20)
    plt.ylabel(r'Count', fontsize=20)
    plt.tick_params(labelsize=18)
    plt.legend(fontsize=20, frameon=False) 
    # plt.savefig(f'data/pdf/multi_layer_vol.pdf', bbox_inches='tight')


def compute_stats_single_layer(neper_mesh):
    edges = onp.load(f"data/numpy/fd_{neper_mesh}/info/edges.npy")
    volumes = onp.load(f"data/numpy/fd_{neper_mesh}/info/vols.npy")
    centroids = onp.load(f"data/numpy/fd_{neper_mesh}/info/centroids.npy")
    cell_grain_inds = onp.load(f"data/numpy/fd_{neper_mesh}/info/cell_grain_inds.npy")
    num_fd_nodes = len(volumes)
    avg_cell_vol, avg_cell_len = fd_helper(num_fd_nodes)

    def compute_stats_helper():
        if case.startswith('fd'):
            cell_ori_inds = onp.load(f"data/numpy/{case}/sols/cell_ori_inds_{step:03d}.npy")
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

        # More reasonable: This is NOT what's currently in paper
        # melt = onp.logical_and(melt, zeta > 0.5)

        return T, zeta, melt, cell_ori_inds

    def process_T():
        sampling_depth = 5
        sampling_width = 5
        avg_length = 8
        sampling_section = sampling_depth*sampling_width*2

        bias = avg_cell_len/2. if neper_mesh == 'npj_review_voronoi_coarse' else 0.
        inds = onp.argwhere((centroids[:, 2] > args.domain_height - sampling_depth*avg_cell_len) & 
                            (centroids[:, 2] < args.domain_height) &
                            (centroids[:, 1] > args.domain_width/2 + bias - sampling_width*avg_cell_len) & 
                            (centroids[:, 1] < args.domain_width/2 + bias + sampling_width*avg_cell_len))[:, 0]

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
        grains_combined = BFS(edges_in_order, melt, cell_ori_inds)
        grain_vols, grain_centroids = get_aspect_ratio_inputs_single_track(grains_combined, volumes, centroids)
        eta_results = compute_aspect_ratios_and_vols(grain_vols, grain_centroids)
        return eta_results

    edges_in_order = get_edges_in_order(edges, len(centroids))


    # cases = ['gn', 'fd']
    # steps = [20]
    cases = [f'gn_{neper_mesh}', f'fd_{neper_mesh}']

    for case in cases:
        numpy_folder = f"data/numpy/{case}/post-processing"
        if not os.path.exists(numpy_folder):
            os.makedirs(numpy_folder)

        T_collect = []
        zeta_collect = []
        eta_collect = []
        for step in range(31):
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


def produce_figures_single_layer(neper_mesh, additional_info=None):
    pdf_folder = f"data/pdf/{neper_mesh}"
    if not os.path.exists(pdf_folder):
        os.makedirs(pdf_folder)

    ts, xs, ys, ps = read_path(f'data/txt/single_track.txt')
    ts = ts[::args.write_sol_interval]*1e6

    volumes = onp.load(f"data/numpy/fd_{neper_mesh}/info/vols.npy")
    num_fd_nodes = len(volumes)
    avg_cell_vol, avg_cell_len = fd_helper(num_fd_nodes)

    def T_plot():
        T_results_fd = onp.load(f"data/numpy/fd_{neper_mesh}/post-processing/T_collect.npy")
        T_results_gn = onp.load(f"data/numpy/gn_{neper_mesh}/post-processing/T_collect.npy")

        step = 12
        T_select_fd = T_results_fd[step]
        T_select_gn = T_results_gn[step]
        x = onp.linspace(0., args.domain_length, len(T_select_fd))*1e3

        fig = plt.figure(figsize=(8, 6))
        plt.plot(x, T_select_fd, label='DNS', color='blue', marker='o', markersize=8, linestyle="-", linewidth=2)
        plt.plot(x, T_select_gn, label='PEGN', color='red', marker='o', markersize=8, linestyle="-", linewidth=2)
        plt.xlabel(r'x-axis [$\mu$m]', fontsize=20)
        plt.ylabel(r'Temperature [K]', fontsize=20)
        plt.tick_params(labelsize=18)
        plt.legend(fontsize=20, frameon=False)
        plt.savefig(f'data/pdf/{neper_mesh}/T_scanning_line.pdf', bbox_inches='tight')

        ind_T = T_results_fd.shape[1]//2
        fig = plt.figure(figsize=(8, 6))
        plt.plot(ts, T_results_fd[:, ind_T], label='DNS', color='blue', marker='o', markersize=8, linestyle="-", linewidth=2)
        plt.plot(ts, T_results_gn[:, ind_T], label='PEGN', color='red', marker='o', markersize=8, linestyle="-", linewidth=2)
        plt.xlabel(r'Time [$\mu$s]', fontsize=20)
        plt.ylabel(r'Temperature [K]', fontsize=20)
        plt.tick_params(labelsize=18)
        plt.legend(fontsize=20, frameon=False)
        plt.savefig(f'data/pdf/{neper_mesh}/T_center.pdf', bbox_inches='tight')


    def zeta_plot():
        zeta_results_fd = onp.load(f"data/numpy/fd_{neper_mesh}/post-processing/zeta_collect.npy")
        zeta_results_gn = onp.load(f"data/numpy/gn_{neper_mesh}/post-processing/zeta_collect.npy")
        labels = ['Melt pool length [mm]', 'Melt pool width [mm]', 'Melt pool height [mm]', 'Melt pool volume [mm$^3$]']
        names = ['melt_pool_length', 'melt_pool_width', 'melt_pool_height', 'melt_pool_volume']
        for i in range(4):
            fig = plt.figure(figsize=(8, 6))
            plt.plot(ts, zeta_results_fd[:, i], label='DNS', color='blue', marker='o', markersize=8, linestyle="-", linewidth=2)
            plt.plot(ts, zeta_results_gn[:, i], label='PEGN', color='red', marker='o', markersize=8, linestyle="-", linewidth=2)
            plt.xlabel(r'Time [$\mu$s]', fontsize=20)
            plt.ylabel(labels[i], fontsize=20)
            plt.tick_params(labelsize=18)
            plt.legend(fontsize=20, frameon=False)
            plt.savefig(f'data/pdf/{neper_mesh}/{names[i]}.pdf', bbox_inches='tight')


    def eta_plot(neper_mesh):
        eta_results_fd = onp.load(f"data/numpy/fd_{neper_mesh}/post-processing/eta_collect.npy", allow_pickle=True)
        eta_results_gn = onp.load(f"data/numpy/gn_{neper_mesh}/post-processing/eta_collect.npy", allow_pickle=True)

        # val = 1e-7 is used before we consider anisotropy
        # val = 1.6*1e-7 is used after we consider anisotropy
        if neper_mesh == 'npj_review_voronoi_fine':
            val = 0.8*1e-7
        elif neper_mesh == 'npj_review_voronoi_coarse':
            val = 3.2*1e-7
        else:
            val = 1.6*1e-7

        if additional_info == 'npj_review_centroidal_big_grain':
            neper_mesh = additional_info
            val = 1e-5
 
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
        plt.plot(ts, num_vols_gn, label='PEGN', color='red', marker='o', markersize=8, linestyle="-", linewidth=2)
        plt.xlabel(r'Time [$\mu$s]', fontsize=20)
        plt.ylabel(r'Number of grains', fontsize=20)
        plt.tick_params(labelsize=18)
        plt.legend(fontsize=20, frameon=False)
        plt.savefig(f'data/pdf/{neper_mesh}/num_grains.pdf', bbox_inches='tight')

        fig = plt.figure(figsize=(8, 6))
        plt.plot(ts, avg_vol_fd, label='DNS', color='blue', marker='o', markersize=8, linestyle="-", linewidth=2)
        plt.plot(ts, avg_vol_gn, label='PEGN', color='red', marker='o', markersize=8, linestyle="-", linewidth=2)
        plt.xlabel(r'Time [$\mu$s]', fontsize=20)
        plt.ylabel(r'Average grain volume [$\mu$m$^3$]', fontsize=20)
        plt.tick_params(labelsize=18)
        plt.legend(fontsize=20, frameon=False)
        plt.savefig(f'data/pdf/{neper_mesh}/grain_vol.pdf', bbox_inches='tight')

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
        print(f"fd median aspect_ratio = {onp.median(fd_aspect_ratios)}")
        print(f"gn median aspect_ratio = {onp.median(gn_aspect_ratios)}")

        colors = ['blue', 'red']
        labels = ['DNS', 'PEGN']

        fig = plt.figure(figsize=(8, 6))
        plt.hist([fd_vols, gn_vols], color=colors, bins=onp.linspace(0., 1e4, 6), label=labels)
        plt.legend(fontsize=20, frameon=False) 
        plt.xlabel(r'Grain volume [$\mu$m$^3$]', fontsize=20)
        plt.ylabel(r'Count', fontsize=20)
        plt.tick_params(labelsize=18)
        plt.savefig(f'data/pdf/{neper_mesh}/vol_distribution.pdf', bbox_inches='tight')

        fig = plt.figure(figsize=(8, 6))
        plt.hist([fd_aspect_ratios, gn_aspect_ratios], color=colors, bins=onp.linspace(1, 4, 13), label=labels)
        plt.legend(fontsize=20, frameon=False) 
        plt.xlabel(r'Aspect ratio', fontsize=20)
        plt.ylabel(r'Count', fontsize=20)
        plt.tick_params(labelsize=18)
        plt.savefig(f'data/pdf/{neper_mesh}/aspect_distribution.pdf', bbox_inches='tight')


        # print(num_vols_fd)
        # print(num_vols_gn)

    T_plot()
    # zeta_plot()
    # eta_plot(neper_mesh)


def compute_vol_and_area(grain, volumes, centroids, face_areas_in_order, edges_in_order):
    vol = onp.sum(onp.take(volumes, grain))
    cen = onp.mean(onp.take(centroids, grain, axis=0), axis=0)
    hash_table = set(grain)
    # print(f"Total number of g = {len(grain)}")
    area = 0.
    for g in grain:
        count = 0
        for i, f_area in enumerate(face_areas_in_order[g]):
            if edges_in_order[g][i] not in hash_table:
                area += f_area
            else:
                count += 1
        # print(f"Found {count} neighbor")
    # print(f"Total number of neighbors found = {count}")
    return vol, area, cen
       

def grain_nodes_and_edges(grain, edges_in_order):
    hash_table = set(grain)
    count = 0
    for g in grain:
        for e in edges_in_order[g]:
            if e in hash_table:
                count += 1
    return len(grain), count//2


def npj_review_grain_growth():
    neper_mesh = 'npj_review_voronoi'

    cases = ['gn_npj_review_voronoi']
    # cases = [f'gn_{neper_mesh}', f'fd_{neper_mesh}']

    for case in cases:
        args.case = case
        args.num_oris = 20
        args.num_grains = 40000

        compute = True
        if compute:
            files_vtk = glob.glob(f"data/vtk/{args.case}/single_grain/*")
            for f in files_vtk:
                os.remove(f)
            unique_oris_rgb, unique_grain_directions = get_unique_ori_colors()
            edges = onp.load(f"data/numpy/{args.case}/info/edges.npy")
            volumes = onp.load(f"data/numpy/{args.case}/info/vols.npy")
            centroids = onp.load(f"data/numpy/{args.case}/info/centroids.npy")
            face_areas = onp.load(f"data/numpy/{args.case}/info/face_areas.npy")

            edges_in_order, face_areas_in_order = get_edges_and_face_in_order(edges, face_areas, len(centroids))

            grain_geo = []
            for step in range(15, 31, 5):
                print(f"step = {step}, case = {args.case}")
                oris_inds = onp.load(f"data/numpy/{args.case}/sols/cell_ori_inds_{step:03d}.npy")
                melt = onp.load(f"data/numpy/{args.case}/sols/melt_{step:03d}.npy")
                zeta = onp.load(f"data/numpy/{args.case}/sols/zeta_{step:03d}.npy") 
                melt = onp.logical_and(melt, zeta > 0.5)
                ipf_z = onp.take(unique_oris_rgb[2], oris_inds, axis=0)

                grains = BFS(edges_in_order, melt, oris_inds, combined=False)
                grains_combined = BFS(edges_in_order, melt, oris_inds, combined=True)

                # Very ad-hoc
                if args.case.startswith('gn'):
                    selected_grain_id = 17980
                else:
                    selected_grain_id = 2876108

                # To answer reviewer 1 Q5
                if step == 30 and args.case.startswith('gn'):
                    nums_nodes = []
                    nums_edges = []
                    for grain in grains_combined:
                        num_nodes, num_edges = grain_nodes_and_edges(grain, edges_in_order)
                        nums_nodes.append(num_nodes)
                        nums_edges.append(num_edges)
                    nums_nodes = onp.array(nums_nodes)
                    nums_edges = onp.array(nums_edges)
                    print(f"len(nums_nodes) = {len(nums_nodes)}")
                    print(f"max nums_nodes = {onp.max(nums_nodes)}, min nums_nodes = {onp.min(nums_nodes)}")
                    print(f"mean nums_nodes = {onp.mean(nums_nodes)}, std nums_nodes = {onp.std(nums_nodes)}")
                    print(f"max nums_edges = {onp.max(nums_edges)}, min nums_edges = {onp.min(nums_edges)}")
                    print(f"mean nums_edges = {onp.mean(nums_edges)}, std nums_edges = {onp.std(nums_edges)}")
                    print(f"onp.argmax(nums_nodes) = {onp.argmax(nums_nodes)}")
                    print(f"onp.argmax(nums_edges) = {onp.argmax(nums_edges)}")


                grains_same_ori = []
                idx = 11
                # 11: pink color
                for i, g in enumerate(grains[idx]):
                    vol, area, cen = compute_vol_and_area(onp.array(g), volumes, centroids, face_areas_in_order, edges_in_order)
                    grains_same_ori += g
                    # print(f"vol = {vol}, area = {area}")
                    # if cen[0] > args.domain_length/2. and cen[1] < args.domain_width:
                    #     print(f"cen = {cen}, i = {i}, g = {g}")
                    if selected_grain_id in g:
                        single_grain_idx = g
                        grain_geo.append([vol, area])
         
 
                def plot_some_grains(grain_ids, name):
                    if args.case.startswith('gn'):
                        mesh = obj_to_vtu(neper_mesh)
                        cells = [('polyhedron', onp.take(mesh.cells_dict['polyhedron'], grain_ids, axis=0))]
                    else:
                        mesh = meshio.read(f"data/vtk/{args.case}/sols/u000.vtu")
                        cells = [('hexahedron', onp.take(mesh.cells_dict['hexahedron'], grain_ids, axis=0))]

                    new_mesh = meshio.Mesh(mesh.points, cells)
                    new_mesh.cell_data['ipf_z'] = [onp.take(ipf_z, grain_ids, axis=0)] 
                    new_mesh.write(f'data/vtk/{args.case}/single_grain/{name}_u{step:03d}.vtu') 

                plot_some_grains(grains_same_ori, 'same_color')
                plot_some_grains(single_grain_idx, 'single_grain')


            onp.save(f"data/numpy/{args.case}/post-processing/grain_geo.npy", onp.array(grain_geo))  
 

    fd_grain_geo = onp.load(f"data/numpy/fd_{neper_mesh}/post-processing/grain_geo.npy") 
    gn_grain_geo = onp.load(f"data/numpy/gn_{neper_mesh}/post-processing/grain_geo.npy") 

    ts, xs, ys, ps = read_path(f'data/txt/single_track.txt')
    ts = ts[::args.write_sol_interval]*1e6
    ts = ts[15:31:5]

    fig = plt.figure(figsize=(8, 6)) 
    plt.plot(ts, fd_grain_geo[:, 0]*1e9, label='DNS', color='blue', marker='o', markersize=8, linestyle="-", linewidth=2)
    plt.plot(ts, gn_grain_geo[:, 0]*1e9, label='PEGN', color='red', marker='o', markersize=8, linestyle="-", linewidth=2)    
    plt.xlabel(r'Time [$\mu$s]', fontsize=20)
    plt.ylabel(r'Grain volume [$\mu$m$^3$]', fontsize=20)
    plt.tick_params(labelsize=18)
    plt.legend(fontsize=20, frameon=False)
    # plt.savefig(f'data/pdf/npj_review_grain_growth/grain_vol.pdf', bbox_inches='tight')            

    fig = plt.figure(figsize=(8, 6)) 
    plt.plot(ts, fd_grain_geo[:, 1]*1e6, label='DNS', color='blue', marker='o', markersize=8, linestyle="-", linewidth=2)
    plt.plot(ts, gn_grain_geo[:, 1]*1e6, label='PEGN', color='red', marker='o', markersize=8, linestyle="-", linewidth=2)    
    plt.xlabel(r'Time [$\mu$s]', fontsize=20)
    plt.ylabel(r'Surface area [$\mu$m$^2$]', fontsize=20)
    plt.tick_params(labelsize=18)
    plt.legend(fontsize=20, frameon=False)
    # plt.savefig(f'data/pdf/npj_review_grain_growth/grain_area.pdf', bbox_inches='tight')            


def single_layer():
    neper_mesh = 'single_layer'
    compute_stats_single_layer(neper_mesh)
    produce_figures_single_layer(neper_mesh)
    

def npj_review_voronoi():
    args.num_oris = 20
    args.num_grains = 40000
    neper_mesh = 'npj_review_voronoi'
    # compute_stats_single_layer(neper_mesh)
    produce_figures_single_layer(neper_mesh)


def npj_review_voronoi_more_oris():
    args.num_oris = 40
    args.num_grains = 40000
    neper_mesh = 'npj_review_voronoi_more_oris'
    # compute_stats_single_layer(neper_mesh)
    produce_figures_single_layer(neper_mesh)


def npj_review_voronoi_less_oris():
    args.num_oris = 10
    args.num_grains = 40000
    neper_mesh = 'npj_review_voronoi_less_oris'
    # compute_stats_single_layer(neper_mesh)
    produce_figures_single_layer(neper_mesh)


def npj_review_voronoi_fine():
    args.num_oris = 20
    args.num_grains = 80000
    neper_mesh = 'npj_review_voronoi_fine'
    # compute_stats_single_layer(neper_mesh)
    produce_figures_single_layer(neper_mesh)


def npj_review_voronoi_coarse():
    args.num_oris = 20
    args.num_grains = 20000
    neper_mesh = 'npj_review_voronoi_coarse'
    # compute_stats_single_layer(neper_mesh)
    produce_figures_single_layer(neper_mesh)


def npj_review_centroidal():
    args.num_oris = 20
    args.num_grains = 40000
    neper_mesh = 'npj_review_centroidal'
    # compute_stats_single_layer(neper_mesh)
    produce_figures_single_layer(neper_mesh)


def npj_review_centroidal_big_grain():
    args.num_oris = 20
    args.num_grains = 40000
    neper_mesh = 'npj_review_centroidal'
    produce_figures_single_layer(neper_mesh, 'npj_review_centroidal_big_grain')


def npj_review_laser_150():
    args.power = 150.
    neper_mesh = 'npj_review_voronoi_laser_150'
    # compute_stats_single_layer(neper_mesh)
    produce_figures_single_layer(neper_mesh)


def npj_review_laser_250():
    args.power = 250.
    neper_mesh = 'npj_review_voronoi_laser_250'
    # compute_stats_single_layer(neper_mesh)
    produce_figures_single_layer(neper_mesh)


def npj_review_laser_100():
    args.power = 100.
    neper_mesh = 'npj_review_voronoi_laser_100'
    # compute_stats_single_layer(neper_mesh)
    produce_figures_single_layer(neper_mesh)


if __name__ == "__main__":
    # generate_demo_graph()
    # vtk_convert_from_server()
    # get_unique_ori_colors()
    # ipf_logo()
    # make_video()
    # compute_stats_multi_layer()
    # produce_figures_multi_layer()

    npj_review_voronoi()
    npj_review_voronoi_more_oris()
    npj_review_voronoi_less_oris()
    npj_review_voronoi_fine()
    npj_review_voronoi_coarse()
    npj_review_centroidal()
    # npj_review_centroidal_big_grain()
    npj_review_laser_100()

    # npj_review_grain_growth()
    # plt.show()
