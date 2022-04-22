import jraph
import jax
import jax.numpy as np
import numpy as onp
import meshio
import os
import glob
import pickle
from functools import partial
from scipy.spatial.transform import Rotation as R
from collections import namedtuple
from matplotlib import pyplot as plt
from src.arguments import args
from src.plots import save_animation
from src.utils import unpack_state, get_unique_ori_colors, obj_to_vtu, walltime, read_path
 

# TODO: unique_oris_rgb and unique_grain_directions should be a class property, not an instance property
PolyCrystal = namedtuple('PolyCrystal', ['edges', 'ch_len', 'centroids', 'volumes', 'unique_oris_rgb', 
    'unique_grain_directions', 'cell_ori_inds', 'boundary_face_areas', 'boundary_face_centroids', 'meta_info'])


@partial(jax.jit, static_argnums=(2,))
def rk4(state, t_crt, f, *ode_params):
    '''
    Fourth order Runge-Kutta method
    We probably don't need this one.
    '''
    y_prev, t_prev = state
    h = t_crt - t_prev
    k1 = h * f(y_prev, t_prev, *ode_params)
    k2 = h * f(y_prev + k1/2., t_prev + h/2., *ode_params)
    k3 = h * f(y_prev + k2/2., t_prev + h/2., *ode_params)
    k4 = h * f(y_prev + k3, t_prev + h, *ode_params)
    y_crt = y_prev + 1./6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return (y_crt, t_crt), y_crt


@partial(jax.jit, static_argnums=(2,))
def explicit_euler(state, t_crt, f, *ode_params):
    '''
    Explict Euler method
    '''
    y_prev, t_prev = state
    h = t_crt - t_prev
    y_crt = y_prev + h * f(y_prev, t_prev, *ode_params)
    return (y_crt, t_crt), y_crt


@jax.jit
def force_eta_zero_in_liquid(y):
    '''
    In liquid zone, set all eta to be zero.
    '''
    T, zeta, eta = unpack_state(y)
    eta = np.where(zeta < 0.5, 0., eta)
    return np.hstack((T, zeta, eta))


def odeint(polycrystal, mesh, mesh_bottom_layer, stepper, f, y0, melt, ts, xs, ys, ps):
    '''
    ODE integrator. 
    '''
    clean_sols()
    state = (y0, ts[0])
    write_sols(polycrystal, mesh, y0, melt, 0)
    for (i, t_crt) in enumerate(ts[1:]):
        state, y = stepper(state, t_crt, f, xs[i + 1], ys[i + 1], ps[i + 1])
        state = (force_eta_zero_in_liquid(y), t_crt)
        melt = np.logical_or(melt, y[:, 1] < 0.5)
        if (i + 1) % 20 == 0:
            print(f"step {i + 1}")
            # print(y[:10, :5])
            inspect_sol(y, y0)
            if not np.all(np.isfinite(y)):          
                raise ValueError(f"Found np.inf or np.nan in y - stop the program")
        write_sol_interval = args.write_sol_interval
        if (i + 1) % write_sol_interval == 0:
            write_sols(polycrystal, mesh, y, melt, (i + 1) // write_sol_interval)

    write_final_sols(polycrystal, mesh_bottom_layer, y, melt)
    write_info(polycrystal)

 
def inspect_sol(y, y0):
    '''
    While running simulations, print out some useful information.
    '''
    T = y[:, 0]
    zeta = y[:, 1]
    change_zeta = np.where(zeta < 0.5, 1, 0)
    eta0 = np.argmax(y0[:, 2:], axis=1)
    eta = np.argmax(y[:, 2:], axis=1)
    change_eta = np.where(eta0 == eta, 0, 1)
    change_T = np.where(T >= args.T_melt, 1, 0)
    print(f"percet of change of zeta (liquid) = {np.sum(change_zeta)/len(change_zeta)*100}%")
    print(f"percent of change of orientations = {np.sum(change_eta)/len(change_eta)*100}%")
    print(f"percet of T >= T_melt = {np.sum(change_T)/len(change_T)*100}%")
    print(f"max T = {np.max(T)}")
 

def clean_sols():
    '''
    Clean the data folder.
    '''
    if args.case == 'fd' or args.case == 'gn':
        vtk_folder = f"data/vtk/{args.case}/sols"
        numpy_folder = f"data/numpy/{args.case}/sols"
    else:
        vtk_folder = f"data/vtk/{args.case}/sols/layer_{args.layer:03d}"
        if not os.path.exists(vtk_folder):
            os.makedirs(vtk_folder)
        numpy_folder = f'data/numpy/{args.case}/sols/layer_{args.layer:03d}'
        if not os.path.exists(numpy_folder):
            os.makedirs(numpy_folder)

    files_vtk = glob.glob(vtk_folder + f"/*")
    files_numpy = glob.glob(numpy_folder + f"/*")
    files = files_vtk + files_numpy
    for f in files:
        os.remove(f)


def write_info(polycrystal):
    '''
    Mostly for post-processing. E.g., compute grain volume, aspect ratios, etc.
    '''
    if args.case == 'fd' or args.case == 'gn':
        onp.save(f"data/numpy/{args.case}/info/edges.npy", polycrystal.edges)
        onp.save(f"data/numpy/{args.case}/info/vols.npy", polycrystal.volumes)
        onp.save(f"data/numpy/{args.case}/info/centroids.npy", polycrystal.centroids)


def write_sols_heper(polycrystal, mesh, y, melt):
    T = y[:, 0]
    zeta = y[:, 1]
    eta = y[:, 2:]
    cell_ori_inds = onp.argmax(eta, axis=1)
    ipf_x = onp.take(polycrystal.unique_oris_rgb[0], cell_ori_inds, axis=0)
    ipf_y = onp.take(polycrystal.unique_oris_rgb[1], cell_ori_inds, axis=0)
    ipf_z = onp.take(polycrystal.unique_oris_rgb[2], cell_ori_inds, axis=0)
    mesh.cell_data['T'] = [onp.array(T, dtype=onp.float32)]
    mesh.cell_data['zeta'] = [onp.array(zeta, dtype=onp.float32)]
    mesh.cell_data['ipf_x'] = [ipf_x]
    mesh.cell_data['ipf_y'] = [ipf_y]
    mesh.cell_data['ipf_z'] = [ipf_z]
    mesh.cell_data['melt'] = [onp.array(melt, dtype=onp.float32)]
    cell_ori_inds = onp.array(cell_ori_inds, dtype=onp.int32)
    mesh.cell_data['ori_inds'] = [cell_ori_inds]

    return T, zeta, cell_ori_inds


def write_sols(polycrystal, mesh, y, melt, step):
    '''
    Use Paraview to open .vtu files for visualization of:
    1. Temeperature field (T)
    2. Liquid/Solid phase (zeta)
    3. Grain orientations (eta)
    '''
    print(f"Write sols to file...")
    T, zeta, cell_ori_inds = write_sols_heper(polycrystal, mesh, y, melt)
    if args.case == 'fd' or args.case == 'gn':
        onp.save(f"data/numpy/{args.case}/sols/T_{step:03d}.npy", T)
        onp.save(f"data/numpy/{args.case}/sols/zeta_{step:03d}.npy", zeta)
        onp.save(f"data/numpy/{args.case}/sols/cell_ori_inds_{step:03d}.npy", cell_ori_inds)
        onp.save(f"data/numpy/{args.case}/sols/melt_{step:03d}.npy", melt)
        mesh.write(f"data/vtk/{args.case}/sols/u{step:03d}.vtu")
    else:
        if args.layer < 6:
            mesh.write(f"data/vtk/{args.case}/sols/layer_{args.layer:03d}/u{step:03d}.vtu")


def write_final_sols(polycrystal, mesh_bottom_layer, y, melt):
    if args.case == 'part':
        y_to_save = onp.array(y[args.layer_num_dofs:, :])
        y_to_save[:, 0] = args.T_ambient
        y_to_save[:, 1] = 1.
        np.save(f'data/numpy/{args.case}/sols/layer_{args.layer:03d}/y_final.npy', y_to_save)
        np.save(f'data/numpy/{args.case}/sols/layer_{args.layer:03d}/melt_final.npy', melt[args.layer_num_dofs:])
        write_sols_heper(polycrystal, mesh_bottom_layer, y[:args.layer_num_dofs, :], melt[:args.layer_num_dofs])
        mesh_bottom_layer.write(f"data/vtk/{args.case}/sols/group/sol_bottom_layer_{args.layer:03d}.vtu")


def polycrystal_gn(domain_name='domain_big'):
    '''
    Prepare graph information for reduced-order modeling
    '''
    unique_oris_rgb, unique_grain_directions = get_unique_ori_colors()
    grain_oris_inds = onp.random.randint(args.num_oris, size=args.num_grains)
    cell_ori_inds = grain_oris_inds
    mesh = obj_to_vtu(domain_name)

    stface = onp.loadtxt(f'data/neper/{domain_name}/domain.stface')
    face_centroids = stface[:, :3]
    face_areas = stface[:, 3]

    edges = [[] for i in range(len(face_areas))]
    centroids = []
    volumes = []
 
    file = open(f'data/neper/{domain_name}/domain.stcell', 'r')
    lines = file.readlines()

    assert args.num_grains == len(lines)
 
    boundary_face_areas = onp.zeros((args.num_grains, 6))
    boundary_face_centroids = onp.zeros((args.num_grains, 6, args.dim))

    for i, line in enumerate(lines):
        l = line.split()
        centroids.append([float(l[0]), float(l[1]), float(l[2])])
        volumes.append(float(l[3]))
        l = l[4:]
        num_nb_faces = len(l)
        for j in range(num_nb_faces):
            edges[int(l[j]) - 1].append(i)

    centroids = onp.array(centroids)
    volumes = onp.array(volumes)
 
    new_face_areas = []
    new_edges = []

    def face_centroids_to_boundary_index(face_centroid):
        domain_measures = [args.domain_length, args.domain_width, args.domain_height]
        for i, domain_measure in enumerate(domain_measures):
            if onp.isclose(face_centroid[i], 0., atol=1e-08):
                return 2*i
            if onp.isclose(face_centroid[i], domain_measure, atol=1e-08):
                return 2*i + 1
        raise ValueError(f"Expect a boundary face, got centroid {face_centroid} that is not on any boundary.")

    for i, edge in enumerate(edges):
        if len(edge) == 1:
            grain_index = edge[0]
            boundary_index = face_centroids_to_boundary_index(face_centroids[i])
            face_area = face_areas[i]
            face_centroid = face_centroids[i]
            boundary_face_areas[grain_index, boundary_index] = face_area
            boundary_face_centroids[grain_index, boundary_index] = face_centroid
        elif len(edge) == 2:
            new_edges.append(edge)
            new_face_areas.append(face_areas[i])
        else:
            raise ValueError(f"Number of connected grains for any face must be 1 or 2, got {len(edge)}.")

    new_edges = onp.array(new_edges)
    new_face_areas = onp.array(new_face_areas)

    centroids_1 = onp.take(centroids, new_edges[:, 0], axis=0)
    centroids_2 = onp.take(centroids, new_edges[:, 1], axis=0)
    grain_distances = onp.sqrt(onp.sum((centroids_1 - centroids_2)**2, axis=1))
 
    ch_len = new_face_areas / grain_distances

    # domain_vol = args.domain_length*args.domain_width*args.domain_height
    # ch_len_avg = (domain_vol / args.num_grains)**(1./3.) * onp.ones(len(new_face_areas))

    meta_info = onp.array([0., 0., 0., args.domain_length, args.domain_width, args.domain_height])
    polycrystal = PolyCrystal(new_edges, ch_len, centroids, volumes, unique_oris_rgb, unique_grain_directions,
                              cell_ori_inds, boundary_face_areas, boundary_face_centroids, meta_info)

    return polycrystal, mesh


def polycrystal_fd(domain_name='domain_big'):
    '''
    Prepare graph information for finite difference method
    '''
    filepath=f'data/neper/{domain_name}/domain.msh'
    mesh = meshio.read(filepath)
    points = mesh.points
    cells =  mesh.cells_dict['hexahedron']
    cell_grain_inds = mesh.cell_data['gmsh:physical'][0] - 1
    onp.save(f"data/numpy/fd/info/cell_grain_inds.npy", cell_grain_inds)
    assert args.num_grains == onp.max(cell_grain_inds) + 1

    unique_oris_rgb, unique_grain_directions = get_unique_ori_colors()
    grain_oris_inds = onp.random.randint(args.num_oris, size=args.num_grains)
    cell_ori_inds = onp.take(grain_oris_inds, cell_grain_inds, axis=0)

    Nx = round(args.domain_length / points[1, 0])
    Ny = round(args.domain_width / points[Nx + 1, 1])
    Nz = round(args.domain_height / points[(Nx + 1)*(Ny + 1), 2])
    args.Nx = Nx
    args.Ny = Ny
    args.Nz = Nz

    print(f"Total num of grains = {args.num_grains}")
    print(f"Total num of orientations = {args.num_oris}")
    print(f"Total num of finite difference cells = {len(cells)}")
    assert Nx*Ny*Nz == len(cells)

    edges = []
    for i in range(Nx):
        if i % 100 == 0:
            print(f"i = {i}")
        for j in range(Ny):
            for k in range(Nz):
                crt_ind = i + j * Nx + k * Nx * Ny
                if i != Nx - 1:
                    edges.append([crt_ind, (i + 1) + j * Nx + k * Nx * Ny])
                if j != Ny - 1:
                    edges.append([crt_ind, i + (j + 1) * Nx + k * Nx * Ny])
                if k != Nz - 1:
                    edges.append([crt_ind, i + j * Nx + (k + 1) * Nx * Ny])

    edges = onp.array(edges)
    cell_points = onp.take(points, cells, axis=0)
    centroids = onp.mean(cell_points, axis=1)
    domain_vol = args.domain_length*args.domain_width*args.domain_height
    volumes = domain_vol / (Nx*Ny*Nz) * onp.ones(len(cells))
    ch_len = (domain_vol / len(cells))**(1./3.) * onp.ones(len(edges))

    face_inds = [[0, 3, 4, 7], [1, 2, 5, 6], [0, 1, 4, 5], [2, 3, 6, 7], [0, 1, 2, 3], [4, 5, 6, 7]]
    boundary_face_centroids = onp.transpose(onp.stack([onp.mean(onp.take(cell_points, face_ind, axis=1), axis=1) 
        for face_ind in face_inds]), axes=(1, 0, 2))
    
    boundary_face_areas = []
    domain_measures = [args.domain_length, args.domain_width, args.domain_height]
    face_cell_nums = [Ny*Nz, Nx*Nz, Nx*Ny]
    for i, domain_measure in enumerate(domain_measures):
        cell_area = domain_vol/domain_measure/face_cell_nums[i]
        boundary_face_area1 = onp.where(onp.isclose(boundary_face_centroids[:, 2*i, i], 0., atol=1e-08), cell_area, 0.)
        boundary_face_area2 = onp.where(onp.isclose(boundary_face_centroids[:, 2*i + 1, i], domain_measure, atol=1e-08), cell_area, 0.)
        boundary_face_areas += [boundary_face_area1, boundary_face_area2]

    boundary_face_areas = onp.transpose(onp.stack(boundary_face_areas))

    meta_info = onp.array([0., 0., 0., args.domain_length, args.domain_width, args.domain_height])
    polycrystal = PolyCrystal(edges, ch_len, centroids, volumes, unique_oris_rgb, unique_grain_directions,
                              cell_ori_inds, boundary_face_areas, boundary_face_centroids, meta_info)


    # centroids_reshape = onp.reshape(centroids, (Nz, Ny, Nx, 3))
    # print(centroids_reshape[Nz - 1, 0, Nx - 1])
    # print(centroids_reshape[Nz - 1, Ny - 1, Nx - 1])
    # print(centroids_reshape[0, 0, Nx - 1])
    # print(centroids_reshape[0, Ny - 1, Nx - 1])
    # exit()

    return polycrystal, mesh


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


def layered_initialization(poly_top_layer):
    y_top, melt_top = default_initialization(poly_top_layer)
    y_down = np.load(f'data/numpy/{args.case}/sols/layer_{args.layer - 1:03d}/y_final.npy')
    melt_down = np.load(f'data/numpy/{args.case}/sols/layer_{args.layer - 1:03d}/melt_final.npy')
    return np.vstack((y_down, y_top)), np.hstack((melt_down, melt_top))


def build_graph(polycrystal, y0):
    '''
    Initialize graph using JAX library Jraph
    https://github.com/deepmind/jraph
    '''
    print(f"Build graph...")
    num_nodes = len(polycrystal.centroids)
    senders = polycrystal.edges[:, 0]
    receivers = polycrystal.edges[:, 1]
    n_node = np.array([num_nodes])
    n_edge = np.array([len(senders)])
    senders = np.array(senders)
    receivers = np.array(receivers)

    print(f"Total number nodes = {n_node[0]}, total number of edges = {n_edge[0]}")

    node_features = {'state':y0, 
                     'centroids': polycrystal.centroids,
                     'volumes': polycrystal.volumes[:, None],
                     'boundary_face_areas': polycrystal.boundary_face_areas, 
                     'boundary_face_centroids': polycrystal.boundary_face_centroids}

    edge_features = {'ch_len': polycrystal.ch_len[:, None],
                     'anisotropy': np.ones((n_edge[0], args.num_oris))}

    graph = jraph.GraphsTuple(nodes=node_features, edges=edge_features, senders=senders, receivers=receivers,
        n_node=n_node, n_edge=n_edge, globals={})

    return graph


def update_graph():
    '''
    With the help of Jraph, we can compute both grad_energy and local_energy easily.
    Note that grad_energy should be understood as stored in edges, while local_energy stored in nodes.
    '''
    # TODO: Don't do sum here. Let Jraph do sum by defining global energy.
    def update_edge_fn(edges, senders, receivers, globals_):
        '''
        Compute grad_energy for T, zeta, eta
        '''
        del globals_
        sender_T, sender_zeta, sender_eta = unpack_state(senders['state'])
        receiver_T, receiver_zeta, receiver_eta = unpack_state(receivers['state'])
        ch_len = edges['ch_len']
        anisotropy = edges['anisotropy']
        assert anisotropy.shape == sender_eta.shape
        grad_energy_T = args.kappa_T * 0.5 * np.sum((sender_T - receiver_T)**2 * ch_len)
        grad_energy_zeta = args.kappa_p * 0.5 * np.sum((sender_zeta - receiver_zeta)**2 * ch_len)
        grad_energy_eta = args.kappa_g * 0.5 * np.sum((sender_eta - receiver_eta)**2 * ch_len * anisotropy)
        grad_energy = (grad_energy_zeta + grad_energy_eta) * args.ad_hoc + grad_energy_T
 
        return {'grad_energy': grad_energy}

    def update_node_fn(nodes, sent_edges, received_edges, globals_):
        '''
        Compute local_energy for zeta and eta
        '''
        del sent_edges, received_edges

        T, zeta, eta = unpack_state(nodes['state'])
        assert T.shape == zeta.shape
        phi = 0.5 * (1 - np.tanh(1e2*(T/args.T_melt - 1)))
        phase_energy = args.m_p * np.sum(((1 - zeta)**2 * phi + zeta**2 * (1 - phi)))
        gamma = 1
        vmap_outer = jax.vmap(np.outer, in_axes=(0, 0))
        grain_energy_1 = np.sum((eta**4/4. - eta**2/2.))
        graph_energy_2 = gamma * (np.sum(np.sum(vmap_outer(eta, eta)**2, axis=(1, 2))[:, None]) - np.sum(eta**4))
        graph_energy_3 = np.sum((1 - zeta.reshape(-1))**2 * np.sum(eta**2, axis=1).reshape(-1))
        grain_energy = args.m_g * (grain_energy_1 +  graph_energy_2 + graph_energy_3)

        local_energy = phase_energy + grain_energy
        local_energy = local_energy / args.ad_hoc

        return {'local_energy': local_energy}

    def update_global_fn(nodes, edges, globals_):
        del globals_
        total_energy = edges['grad_energy'] + nodes['local_energy']
        return {'total_energy': total_energy}

    net_fn = jraph.GraphNetwork(update_edge_fn=update_edge_fn,
                                update_node_fn=update_node_fn,
                                update_global_fn=update_global_fn)

    return net_fn


def phase_field(graph, polycrystal):
    net_fn = update_graph()
    volumes = graph.nodes['volumes']
    centroids = graph.nodes['centroids']

    def heat_source(y, t, *ode_params):
        '''
        Using a boundary heat source with following reference:
        Lian, Yanping, et al. "A cellular automaton finite volume method for microstructure evolution during 
        additive manufacturing." Materials & Design 169 (2019): 107672.
        The heat source only acts on the top surface.
        Also, convection and radiation are considered, which act on all surfaces.
        '''
        power_x, power_y, power_on = ode_params
        T, zeta, eta = unpack_state(y)
        boundary_face_areas = graph.nodes['boundary_face_areas']
        boundary_face_centroids = graph.nodes['boundary_face_centroids']

        q_convection = np.sum(args.h_conv*(args.T_ambient - T)*boundary_face_areas, axis=1)
        q_radiation = np.sum(args.emissivity*args.SB_constant*(args.T_ambient**4 - T**4)*boundary_face_areas, axis=1)

        # 0: left surface, 1: right surface, 2: front surface, 3: back surface, 4: bottom surface, 5: top surface
        upper_face_centroids = boundary_face_centroids[:, 5, :]
        upper_face_areas = boundary_face_areas[:, 5]

        X = upper_face_centroids[:, 0] - power_x
        Y = upper_face_centroids[:, 1] - power_y
        q_laser = 2*args.power*args.power_fraction/(np.pi * args.r_beam**2) * np.exp(-2*(X**2 + Y**2)/args.r_beam**2) * upper_face_areas
        q_laser = q_laser * power_on

        q = q_convection + q_radiation + q_laser

        return q[:, None]

    def update_anisotropy():
        '''
        Determine anisotropy (see Yan paper Eq. (12))
        '''
        print("compute_anisotropy...")
        if args.case == 'fd':
            y = graph.nodes['state']
            eta = y[:, 2:]
            eta_xyz = np.reshape(eta, (args.Nz, args.Ny, args.Nx, args.num_oris))
            eta_neg_x = np.concatenate((eta_xyz[:, :, :1, :], eta_xyz[:, :, :-1, :]), axis=2)
            eta_pos_x = np.concatenate((eta_xyz[:, :, 1:, :], eta_xyz[:, :, -1:, :]), axis=2)
            eta_neg_y = np.concatenate((eta_xyz[:, :1, :, :], eta_xyz[:, :-1, :, :]), axis=1)
            eta_pos_y = np.concatenate((eta_xyz[:, 1:, :, :], eta_xyz[:, -1:, :, :]), axis=1)
            eta_neg_z = np.concatenate((eta_xyz[:1, :, :, :], eta_xyz[:-1, :, :, :]), axis=0)
            eta_pos_z = np.concatenate((eta_xyz[1:, :, :, :], eta_xyz[-1:, :, :, :]), axis=0)
            directions_xyz = np.stack((eta_pos_x - eta_neg_x, eta_pos_y - eta_neg_y, eta_pos_z - eta_neg_z), axis=-1)
            assert directions_xyz.shape == (args.Nz, args.Ny, args.Nx, args.num_oris, args.dim)
            directions = directions_xyz.reshape(-1, args.num_oris, args.dim)
            sender_directions = np.take(directions, graph.senders, axis=0)
            receivers_directions = np.take(directions, graph.receivers, axis=0)
            edge_directions = (sender_directions + receivers_directions) / 2.
        else:
            sender_centroids = np.take(centroids, graph.senders, axis=0)
            receiver_centroids = np.take(centroids, graph.receivers, axis=0)
            edge_directions = sender_centroids - receiver_centroids
            edge_directions = np.repeat(edge_directions[:, None, :], args.num_oris, axis=1) # (num_edges, num_oris, dim)
 
        assert edge_directions.shape == (len(graph.senders), args.num_oris, args.dim)
        unique_grain_directions = polycrystal.unique_grain_directions # (num_directions_per_cube, num_oris, dim)
        cosines = np.sum(unique_grain_directions[None, :, :, :] * edge_directions[:, None, :, :], axis=-1) \
                  / (np.linalg.norm(edge_directions, axis=-1)[:, None, :])
        anlges =  np.arccos(cosines) 
        anisotropy_term =  1 + args.anisotropy * np.max((np.cos(anlges)**4 + np.sin(anlges)**4), axis=1) # (num_edges, num_oris)
        assert anisotropy_term.shape == (len(graph.senders), args.num_oris)
        anisotropy_term = np.where(np.isfinite(anisotropy_term), anisotropy_term, 1 + args.anisotropy/2.)
        graph.edges['anisotropy'] = anisotropy_term
 

    def compute_energy(y, t, *ode_params):
        '''
        When you call net_fn, you are asking Jraph to compute the total free energy (grad energy + local energy) for you.
        '''
        q = heat_source(y, t, *ode_params)
        graph.nodes['state'] = y
        new_graph = net_fn(graph)
        return new_graph.edges['grad_energy'], new_graph.nodes['local_energy'], q

    grad_energy_der_fn = jax.grad(lambda y, t, *ode_params: compute_energy(y, t, *ode_params)[0])
    local_energy_der_fn = jax.grad(lambda y, t, *ode_params: compute_energy(y, t, *ode_params)[1])

    def state_rhs(y, t, *ode_params):
        '''
        Define the right-hand-side function for the ODE system
        '''
        update_anisotropy()
        _, _, q = compute_energy(y, t, *ode_params)
        T, zeta, eta = unpack_state(y)

        # If T is too large, L would be too large - solution diverges; Also, T too large is not physical.
        T = np.where(T > 2000., 2000., T)

        der_grad = grad_energy_der_fn(y, t, *ode_params)
        der_local = local_energy_der_fn(y, t, *ode_params)
        L = args.L0 * np.exp(-args.Qg / (T*args.gas_const))
        rhs_phase_field = -L * (der_grad[:, 1:]/volumes + der_local[:, 1:])
        rhs_T = (-der_grad[:, 0:1] + q)/volumes/(args.rho * args.c_h)
        rhs = np.hstack((rhs_T, rhs_phase_field))

        return rhs

    return state_rhs


def debug():
    args.case = 'fd'
    ts, xs, ys, ps = read_path(f'data/txt/single_track.txt')
    if args.case == 'gn':
        simulate(ts, xs, ys, ps, polycrystal_gn)
    else:
        simulate(ts, xs, ys, ps, polycrystal_fd)


@walltime
def simulate(ts, xs, ys, ps, func):
    polycrystal, mesh = func()
    y0, melt = default_initialization(polycrystal)
    graph = build_graph(polycrystal, y0)
    state_rhs = phase_field(graph, polycrystal)
    odeint(polycrystal, mesh, None, explicit_euler, state_rhs, y0, melt, ts, xs, ys, ps)
   

def run():
    args.case = 'fd'
    ts, xs, ys, ps = read_path(f'data/txt/single_track.txt')
    simulate(ts, xs, ys, ps, polycrystal_fd)


if __name__ == "__main__":
    # run()
    debug()
