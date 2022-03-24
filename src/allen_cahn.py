import jraph
import jax
import jax.numpy as np
import numpy as onp
import meshio
import os
import pickle
from functools import partial
from scipy.spatial.transform import Rotation as R
from collections import namedtuple
from matplotlib import pyplot as plt
import microstructpy as msp
from src.arguments import args
from src.plots import poly_plot, save_animation
from src.utils import unpack_state


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

    # y_crt = np.where(y_crt < 0., 0, y_crt)
    # y_crt = np.where(y_crt > 1., 1, y_crt)

    return (y_crt, t_crt), y_crt


def odeint(stepper, compute_T, f, y0, ts, *diff_args):
    ys = []
    state = (y0, ts[0])
    for (i, t_crt) in enumerate(ts[1:]):
        state, y = stepper(state, t_crt, f, *diff_args)
        if i % 20 == 0:
            print(f"step {i}")
            T = compute_T(t_crt)
            all_state = np.hstack((T, y))
            print(all_state[:20, :5])
            inspect_y(y, y0)
            inspect_T(T)
            if not np.all(np.isfinite(y)):
                print(f"Found np.inf or np.nan in y - stop the program")             
                exit()
        ys.append(y)
    ys = np.array(ys)
    return ys


def inspect_y(y, y0):
    zeta = y[:, 0]
    change_zeta = np.where(zeta < 0.1, 1, 0)

    eta0 = np.argmax(y0[:, 1:], axis=1)
    eta = np.argmax(y[:, 1:], axis=1)
    change_eta = np.where(eta0 == eta, 0, 1)

    print(f"percet of liquid = {np.sum(change_zeta)/len(change_zeta)*100}%")
    print(f"percent of change of oris = {np.sum(change_eta)/len(change_eta)*100}%")


def inspect_T(T):
    change_T = np.where(T >= args.T_melt, 1, 0)
    print(f"percet of T >= T_melt = {np.sum(change_T)/len(change_T)*100}%")
    print(f"max T = {np.max(T)}")


def construct_polycrystal():
    unique_oris = R.random(args.num_oris, random_state=0).as_euler('zxz', degrees=True)
    oris_indics = onp.random.randint(args.num_oris, size=args.num_grains)
    oris = onp.take(unique_oris, oris_indics, axis=0)
    onp.savetxt(f'data/neper/input.ori', oris)

    stface = onp.loadtxt(f'data/neper/domain.stface')
    face_centroids = stface[:, :3]
    face_areas = stface[:, 3]

    edges = [[] for i in range(len(face_areas))]
    centroids = []
    volumes = []
 
    file = open('data/neper/domain.stcell', 'r')
    lines = file.readlines()
    num_grains = len(lines)

    boundary_face_areas = onp.zeros((num_grains, 6))
    boundary_face_centroids = onp.zeros((num_grains, 6, args.dim))

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

    print(new_edges.shape)
    print(new_edges[:10])
    print(onp.sum(boundary_face_areas, axis=0))
 
    PolyCrystal = namedtuple('PolyCrystal', ['edges', 'face_areas', 'grain_distances', 'centroids', 'volumes', 'unique_oris', 
                                             'oris_indics', 'boundary_face_areas', 'boundary_face_centroids'])
    polycrystal = PolyCrystal(new_edges, new_face_areas, grain_distances, centroids, volumes, unique_oris, 
                              oris_indics, boundary_face_areas, boundary_face_centroids)

    return polycrystal


def finite_difference():

    filepath = f'data/neper/domain.msh'
    mesh = meshio.read(filepath)
    mesh.write(f'data/vtk/domain.vtk')

    points = mesh.points
    cells =  mesh.cells_dict['hexahedron']

    Nx = round(args.domain_length / points[1, 0])
    Ny = round(args.domain_width / points[Nx + 1, 1])
    Nz = round(args.domain_height / points[(Nx + 1)*(Ny + 1), 2])

    print(f"total num of cells = {len(cells)}")

    assert Nx*Ny*Nz == len(cells)

    edges = []
    for i in range(Nx):
        print(f"i = {i}")
        for j in range(Ny):
            for k in range(Nz):
                crt_ind = i + j * Nx + k * Nx * Ny
                if i != 0:
                    edges.append([crt_ind, (i - 1) + j * Nx + k * Nx * Ny])
                if i != Nx - 1:
                    edges.append([crt_ind, (i + 1) + j * Nx + k * Nx * Ny])
                if j != 0:
                    edges.append([crt_ind, i + (j - 1) * Nx + k * Nx * Ny])
                if j != Ny - 1:
                    edges.append([crt_ind, i + (j + 1) * Nx + k * Nx * Ny])
                if k != 0:
                    edges.append([crt_ind, i + j * Nx + (k - 1) * Nx * Ny])
                if k != Nz - 1:
                    edges.append([crt_ind, i + j * Nx + (k + 1) * Nx * Ny])



def build_graph(polycrystal):
    senders = []
    receivers = []

    for edge in polycrystal.edges:
        senders += list(edge)
        receivers += list(edge[::-1])

    n_node = np.array([args.num_grains])
    n_edge = np.array([len(senders)])

    print(f"Total number nodes = {n_node[0]}, total number of edges = {n_edge[0]}")

    solid_phases = np.ones(args.num_grains)

    grain_orientations = np.zeros((args.num_grains, args.num_oris))
    inds = jax.ops.index[np.arange(args.num_grains), polycrystal.oris_indics]
    grain_orientations = jax.ops.index_update(grain_orientations, inds, 1)

    senders = np.array(senders)
    receivers = np.array(receivers)

    state = np.hstack((solid_phases[:, None], grain_orientations))

    node_features = {'state':state, 
                     'centroids': polycrystal.centroids,
                     'volumes': polycrystal.volumes[:, None],
                     'boundary_face_areas': polycrystal.boundary_face_areas, 
                     'boundary_face_centroids': polycrystal.boundary_face_centroids}
    edge_features = {'face_areas': np.repeat(polycrystal.face_areas, 2)[:, None],
                     'grain_distances': np.repeat(polycrystal.grain_distances, 2)[:, None]}
    global_features = {'t': 0.}
    graph = jraph.GraphsTuple(nodes=node_features, edges=edge_features, senders=senders, receivers=receivers,
        n_node=n_node, n_edge=n_edge, globals=global_features)

    return graph, state


def update_graph():

    def update_edge_fn(edges, senders, receivers, globals_):
        del globals_
        sender_zeta, sender_eta = unpack_state(senders['state'])
        receiver_zeta, receiver_eta = unpack_state(receivers['state'])
        face_areas = edges['face_areas']
        grain_distances = edges['grain_distances']

        assert face_areas.shape == grain_distances.shape
  
        # print(f"{face_areas[face_areas < 1e-10]}")
        # print(f"max face area = {np.max(face_areas)}, min face area = {np.min(face_areas)}")
        # print(f"max grain distance = {np.max(grain_distances)}, grain distance = {np.min(grain_distances)}")
        # exit()

        # coeff_zeta = 2.77*1e-12
        # grad_energy_zeta = coeff_zeta * 0.25 * np.sum((sender_zeta - receiver_zeta)**2 * face_areas / grain_distances)
        # coeff_eta = 2.77*1e-12
        # grad_energy_eta = coeff_eta * 0.25 *np.sum((sender_eta - receiver_eta)**2 * face_areas / grain_distances)
        # grad_energy = grad_energy_zeta + grad_energy_eta

        coeff_zeta = 2.77*1e-12
        grad_energy_zeta = coeff_zeta * 0.25 * np.sum((sender_zeta - receiver_zeta)**2)
        coeff_eta = 2.77*1e-12
        grad_energy_eta = coeff_eta * 0.25 *np.sum((sender_eta - receiver_eta)**2)
        grad_energy = grad_energy_zeta + grad_energy_eta

        grad_energy = grad_energy / (1e-4)


        return {'grad_energy': grad_energy}

    def update_node_fn(nodes, sent_edges, received_edges, globals_):
        del sent_edges, received_edges

        t = globals_['t'][0]
        T = nodes['T']
        zeta, eta = unpack_state(nodes['state'])
        boundary_face_areas = nodes['boundary_face_areas']
        boundary_face_centroids = nodes['boundary_face_centroids']
        volumes = nodes['volumes']
        centroids = nodes['centroids']
 
        # m_phase = 1.2*1e-4

        # phi = 0.5 * (1 - np.tanh(1e2*(T/args.T_melt - 1)))
        # phase_energy = m_phase * np.sum(((1 - zeta)**2 * phi + zeta**2 * (1 - phi)) * volumes)

        # m_grain = 1.2*1e-4
        # gamma = 1
        # vmap_outer = jax.vmap(np.outer, in_axes=(0, 0))
        # grain_energy_1 = np.sum((eta**4/4. - eta**2/2.) * volumes)
        # graph_energy_2 = gamma * (np.sum(np.sum(vmap_outer(eta, eta)**2, axis=(1, 2))[:, None] * volumes) - np.sum(eta**4 * volumes))
        # graph_energy_3 = np.sum((1 - zeta.reshape(-1))**2 * np.sum(eta**2, axis=1) * volumes.reshape(-1))
        # grain_energy_4 = 0.25 * np.sum(volumes)
        # grain_energy = m_grain * (grain_energy_1 +  graph_energy_2 + graph_energy_3 + grain_energy_4)

        # local_energy = phase_energy + grain_energy

        m_phase = 1.2*1e-4
        phi = 0.5 * (1 - np.tanh(1e2*(T/args.T_melt - 1)))
        phase_energy = m_phase * np.sum(((1 - zeta)**2 * phi + zeta**2 * (1 - phi)))

        m_grain = 1.2*1e-4
        gamma = 1
        vmap_outer = jax.vmap(np.outer, in_axes=(0, 0))
        grain_energy_1 = np.sum((eta**4/4. - eta**2/2.))
        graph_energy_2 = gamma * (np.sum(np.sum(vmap_outer(eta, eta)**2, axis=(1, 2))[:, None]) - np.sum(eta**4))
        graph_energy_3 = np.sum((1 - zeta.reshape(-1))**2 * np.sum(eta**2, axis=1).reshape(-1))
        grain_energy = m_grain * (grain_energy_1 +  graph_energy_2 + graph_energy_3)

        local_energy = phase_energy + grain_energy

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
        T = T_ambiant + Q / (2 * np.pi * kappa) / R * np.exp(-vel / (2*alpha) * (R + X))

        # T = np.where(T > args.T_melt, 2000, T)

        # print(T)
        # exit()

        return T[:, None]

    def compute_energy(y, t):
        T = compute_T(t)
        graph.globals['t'] = t
        graph.nodes['state'] = y
        graph.nodes['T'] = T
        new_graph = net_fn(graph)
        return new_graph.globals['total_energy'][0], T

    grad_energy = jax.grad(lambda y, t: compute_energy(y, t)[0])

    def state_rhs(y, t, *diff_args):
        _, T = compute_energy(y, t)
        grads = grad_energy(y, t)

        gas_const = 8.3
 
        # Qg = 1.4*1e5
        # L = 3.5e12 * np.exp(-Qg/(T*gas_const))

        # Qg = 2.5*1e5
        # L = 6e15 * np.exp(-Qg/(T*gas_const))

        Qg = 1.4*1e5
        # L = 1e8 * np.exp(-Qg/(T*gas_const))
        # rhs = -L * grads / volumes

        L = 1e5

        rhs = -L * grads

        all_var = np.hstack((T, rhs * 2 * 1e-6))

        print(all_var[:100, :5])
        # exit()

        return rhs

    return state_rhs, compute_T


def simulate(ts):
    polycrystal = construct_polycrystal()
    graph, y0 = build_graph(polycrystal)
    state_rhs, compute_T = phase_field(graph)
    ys_ = odeint(explicit_euler, compute_T, state_rhs, y0, ts)
    ys = np.vstack((y0[None, :], ys_))  

    T_final = compute_T(ts[-1])
    zeta_final = ys[-1, :, 0]
    eta_final = ys[-1, :, 1:]

    onp.savetxt(f'data/neper/temp', T_final)
    onp.savetxt(f'data/neper/phase', zeta_final)
    oris_indics = np.argmax(eta_final, axis=1)
    oris = onp.take(polycrystal.unique_oris, oris_indics, axis=0)
    onp.savetxt(f'data/neper/oris', oris)

    return ys, T_final, polycrystal


def exp():
    dt = 2 * 1e-6
    ts = np.arange(0., dt*1201, dt)
    ys, T, polycrystal = simulate(ts)
    show_3d_scatters(ys[-1, :, :], T, polycrystal)
 
    # save_animation(ys[::20], polycrystal)


def show_3d_scatters(y, T, polycrystal):
    x1, x2, x3 = polycrystal.centroids.T
 
    zeta = y[:, 0]
    eta = y[:, 1:]
    oris_indics = np.argmax(eta, axis=1)

    cut = args.domain_width/2.

    x1_show = x1[x2 > cut]
    x2_show = x2[x2 > cut]
    x3_show = x3[x2 > cut]
    T_show = T[x2 > cut]
    zeta_show = zeta[x2 > cut]
    oris_indics_show = oris_indics[x2 > cut]
    
    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes(projection="3d")
    p = ax.scatter3D(x1_show, x2_show, x3_show, s=2, c=T_show, alpha=1, vmin=280, vmax=args.T_melt)
    ax.set_box_aspect((np.ptp(x1_show), np.ptp(x2_show), np.ptp(x3_show)))
    cbar = fig.colorbar(p)

    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes(projection="3d")
    p = ax.scatter3D(x1_show, x2_show, x3_show, s=2, c=zeta_show, alpha=1, vmin=0, vmax=1)
    ax.set_box_aspect((np.ptp(x1_show), np.ptp(x2_show), np.ptp(x3_show)))
    cbar = fig.colorbar(p)

    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes(projection="3d")
    ax.scatter3D(x1_show, x2_show, x3_show, s=2, c=oris_indics_show, alpha=1)
    ax.set_box_aspect((np.ptp(x1_show), np.ptp(x2_show), np.ptp(x3_show)))


if __name__ == "__main__":
    # exp()
    # plt.show()
    # show_3d_scatters()
    finite_difference()
