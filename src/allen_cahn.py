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
from src.utils import unpack_state, get_unique_ori_colors, obj_to_vtu, walltime


@partial(jax.jit, static_argnums=(2,))
def rk4(state, t_crt, f, *diff_args):
    y_prev, t_prev = state
    h = t_crt - t_prev
    k1 = h * f(y_prev, t_prev, *diff_args)
    k2 = h * f(y_prev + k1/2., t_prev + h/2., *diff_args)
    k3 = h * f(y_prev + k2/2., t_prev + h/2., *diff_args)
    k4 = h * f(y_prev + k3, t_prev + h, *diff_args)
    y_crt = y_prev + 1./6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return (y_crt, t_crt), y_crt


@partial(jax.jit, static_argnums=(2,))
def explicit_euler(state, t_crt, f, *diff_args):
    y_prev, t_prev = state
    h = t_crt - t_prev
    y_crt = y_prev + h * f(y_prev, t_prev, *diff_args)
    return (y_crt, t_crt), y_crt


def odeint(polycrystal, mesh, compute_T, stepper, f, y0, ts, *diff_args):
    clean_sols()
    state = (y0, ts[0])
    melt = np.zeros(len(y0), dtype=bool)
    write_sols(polycrystal, mesh, y0, melt, 0)
    for (i, t_crt) in enumerate(ts[1:]):
        state, y = stepper(state, t_crt, f, *diff_args)
        melt = np.logical_or(melt, y[:, 0] < 0.5)
        if (i + 1) % 20 == 0:
            print(f"step {i}")
            T = compute_T(t_crt)
            inspect_T(T)
            # all_state = np.hstack((T, y))
            # print(all_state[:10, :5])
            inspect_y(y, y0)
            if not np.all(np.isfinite(y)):          
                raise ValueError(f"Found np.inf or np.nan in y - stop the program")
        if (i + 1) % 600 == 0:
            write_sols(polycrystal, mesh, y, melt, (i + 1) // 600)

    write_info(polycrystal)


def inspect_y(y, y0):
    zeta = y[:, 0]
    change_zeta = np.where(zeta < 0.5, 1, 0)
    eta0 = np.argmax(y0[:, 1:], axis=1)
    eta = np.argmax(y[:, 1:], axis=1)
    change_eta = np.where(eta0 == eta, 0, 1)
    print(f"percet of change of zeta (liquid) = {np.sum(change_zeta)/len(change_zeta)*100}%")
    print(f"percent of change of oris = {np.sum(change_eta)/len(change_eta)*100}%")


def inspect_T(T):
    change_T = np.where(T >= args.T_melt, 1, 0)
    print(f"percet of T >= T_melt = {np.sum(change_T)/len(change_T)*100}%")
    print(f"max T = {np.max(T)}")


def clean_sols():
    files_vtk = glob.glob(f"data/vtk/{args.case}/sols/*")
    files_numpy = glob.glob(f"data/numpy/{args.case}/sols/*")
    files = files_vtk + files_numpy
    for f in files:
        os.remove(f)


def write_info(polycrystal):
    onp.save(f"data/numpy/{args.case}/info/edges.npy", polycrystal.edges)
    onp.save(f"data/numpy/{args.case}/info/vols.npy", polycrystal.volumes)
    onp.save(f"data/numpy/{args.case}/info/centroids.npy", polycrystal.centroids)


def write_sols(polycrystal, mesh, y, aux, step):
    print(f"Write sols to file...")
    zeta = y[:, 0]
    eta = y[:, 1:]
    cell_ori_inds = onp.argmax(eta, axis=1)
    oris = onp.take(polycrystal.unique_oris, cell_ori_inds, axis=0)
    mesh.cell_data['zeta'] = [onp.array(zeta, dtype=onp.float32)]
    mesh.cell_data['eta'] = [oris]
    cell_ori_inds = onp.array(cell_ori_inds, dtype=onp.int32)
    mesh.cell_data['ori_inds'] = [cell_ori_inds]
    mesh.write(f"data/vtk/{args.case}/sols/u{step:03d}.vtu")
    onp.save(f"data/numpy/{args.case}/sols/cell_ori_inds_{step:03d}.npy", cell_ori_inds)
    onp.save(f"data/numpy/{args.case}/sols/melt_{step:03d}.npy", aux) 
    # onp.save(f"data/numpy/{args.case}/sols/y_{step:03d}.npy", y)  


def polycrystal_gn():
    unique_oris = get_unique_ori_colors()
    grain_oris_inds = onp.random.randint(args.num_oris, size=args.num_grains)
    cell_ori_inds = grain_oris_inds
    mesh = obj_to_vtu()

    stface = onp.loadtxt(f'data/neper/domain.stface')
    face_centroids = stface[:, :3]
    face_areas = stface[:, 3]

    edges = [[] for i in range(len(face_areas))]
    centroids = []
    volumes = []
 
    file = open(f'data/neper/domain.stcell', 'r')
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
        domains = [args.domain_length, args.domain_width, args.domain_height]
        for i, domain in enumerate(domains):
            if onp.isclose(face_centroid[i], 0., atol=1e-08):
                return 2*i
            if onp.isclose(face_centroid[i], domain, atol=1e-08):
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
    # ch_len = (domain_vol / args.num_grains)**(1./3.) * onp.ones(len(new_face_areas))

    PolyCrystal = namedtuple('PolyCrystal', ['edges', 'ch_len', 'face_areas', 'grain_distances', 'centroids', 'volumes', 'unique_oris', 
                                             'cell_ori_inds', 'boundary_face_areas', 'boundary_face_centroids'])
    polycrystal = PolyCrystal(new_edges, ch_len, new_face_areas, grain_distances, centroids, volumes, unique_oris, 
                              cell_ori_inds, boundary_face_areas, boundary_face_centroids)

    return polycrystal, mesh


def polycrystal_fd():
    filepath = f'data/neper/domain.msh'
    mesh = meshio.read(filepath)
    points = mesh.points
    cells =  mesh.cells_dict['hexahedron']
    cell_grain_inds = mesh.cell_data['gmsh:physical'][0]
    onp.save(f"data/numpy/fd/info/cell_grain_inds.npy", cell_grain_inds)
    assert args.num_grains == np.max(cell_grain_inds)

    unique_oris = get_unique_ori_colors()
    grain_oris_inds = onp.random.randint(args.num_oris, size=args.num_grains)
    cell_ori_inds = onp.take(grain_oris_inds, cell_grain_inds - 1, axis=0)

    Nx = round(args.domain_length / points[1, 0])
    Ny = round(args.domain_width / points[Nx + 1, 1])
    Nz = round(args.domain_height / points[(Nx + 1)*(Ny + 1), 2])

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

    edges = np.array(edges)
    cell_points = np.take(points, cells, axis=0)
    centroids = np.mean(cell_points, axis=1)
    domain_vol = args.domain_length*args.domain_width*args.domain_height
    volumes = domain_vol / (Nx*Ny*Nz) * np.ones(len(cells))
    ch_len = (domain_vol / len(cells))**(1./3.) * np.ones(len(edges))

    PolyCrystal = namedtuple('PolyCrystal', ['edges',  'ch_len', 'centroids', 'volumes', 'unique_oris', 'cell_ori_inds'])
    polycrystal = PolyCrystal(edges, ch_len, centroids, volumes, unique_oris, cell_ori_inds)

    return polycrystal, mesh


def build_graph(polycrystal):
    print(f"Build graph...")
    num_nodes = len(polycrystal.centroids)
    senders = polycrystal.edges[:, 0]
    receivers = polycrystal.edges[:, 1]
    n_node = np.array([num_nodes])
    n_edge = np.array([len(senders)])

    print(f"Total number nodes = {n_node[0]}, total number of edges = {n_edge[0]}")

    zeta = np.ones(num_nodes)
    eta = np.zeros((num_nodes, args.num_oris))
    eta = eta.at[np.arange(num_nodes), polycrystal.cell_ori_inds].set(1)
    senders = np.array(senders)
    receivers = np.array(receivers)
    state = np.hstack((zeta[:, None], eta))

    node_features = {'state':state, 
                     'centroids': polycrystal.centroids,
                     'volumes': polycrystal.volumes[:, None]}

    edge_features = {'ch_len': polycrystal.ch_len[:, None]}

    global_features = {'t': 0.}
    graph = jraph.GraphsTuple(nodes=node_features, edges=edge_features, senders=senders, receivers=receivers,
        n_node=n_node, n_edge=n_edge, globals=global_features)

    return graph, state


def update_graph():

    def update_edge_fn(edges, senders, receivers, globals_):
        del globals_
        sender_zeta, sender_eta = unpack_state(senders['state'])
        receiver_zeta, receiver_eta = unpack_state(receivers['state'])
        ch_len = edges['ch_len']
        grad_energy_zeta = args.kappa_p * 0.5 * np.sum((sender_zeta - receiver_zeta)**2 * ch_len)
        grad_energy_eta = args.kappa_g * 0.5 *np.sum((sender_eta - receiver_eta)**2 * ch_len)
        grad_energy = grad_energy_zeta + grad_energy_eta
        grad_energy = grad_energy * args.ad_hoc

        return {'grad_energy': grad_energy}

    def update_node_fn(nodes, sent_edges, received_edges, globals_):
        del sent_edges, received_edges

        t = globals_['t'][0]
        T = nodes['T']
        zeta, eta = unpack_state(nodes['state'])

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


def phase_field(graph):
    net_fn = update_graph()
    volumes = graph.nodes['volumes']
    centroids = graph.nodes['centroids']

    def compute_T(t):
        T_ambiant = 300.
        alpha = 5.2
        Q = 25
        kappa = 2.7*1e-2
        x0 = 0.2*args.domain_length
        vel = 0.6*args.domain_length/(2400*1e-6)
        X = centroids[:, 0] - x0 - vel * t
        Y = centroids[:, 1] - 0.5*args.domain_width
        Z = centroids[:, 2] - args.domain_height
        R = np.sqrt(X**2 + Y**2 + Z**2)
        T = T_ambiant + Q / (2 * np.pi * kappa) / (R) * np.exp(-vel / (2*alpha) * (R + X))

        # Ad-hoc: T can be too high from the analytical model, so we truncate it
        T = np.where(T > 2000, 2000, T)

        return T[:, None]

    def compute_energy(y, t):
        T = compute_T(t)
        graph.globals['t'] = t
        graph.nodes['state'] = y
        graph.nodes['T'] = T
        new_graph = net_fn(graph)
        # return new_graph.globals['total_energy'][0], T
        return new_graph.edges['grad_energy'], new_graph.nodes['local_energy'], T
 
    grad_energy_der_fn = jax.grad(lambda y, t: compute_energy(y, t)[0])
    local_energy_der_fn = jax.grad(lambda y, t: compute_energy(y, t)[1])
   

    def state_rhs(y, t, *diff_args):
        _, _, T = compute_energy(y, t)
        der_grad = grad_energy_der_fn(y, t)
        der_local = local_energy_der_fn(y, t)
        L = args.L0 * np.exp(-args.Qg / (T*args.gas_const))
        rhs = -L * (der_grad/volumes + der_local)
        return rhs

    return state_rhs, compute_T


@walltime
def simulate(ts, func):
    polycrystal, mesh = func()
    graph, y0 = build_graph(polycrystal)
    aux0 = np.zeros(len(y0), dtype=bool)
    state_rhs, compute_T = phase_field(graph)
    odeint(polycrystal, mesh, compute_T, explicit_euler, state_rhs, y0, ts)
   

def run():
    args.case = 'gn'
    # args.case = 'fd'

    if args.case == 'gn':
        args.dt = 2 * 1e-7
        ts = np.arange(0., args.dt*12001, args.dt)
        # ts = np.arange(0., args.dt*601, args.dt)
        simulate(ts, polycrystal_gn)
    else:
        args.dt = 2 * 1e-7
        ts = np.arange(0., args.dt*12001, args.dt)
        # ts = np.arange(0., args.dt*21, args.dt)
        simulate(ts, polycrystal_fd)


if __name__ == "__main__":
    run()

