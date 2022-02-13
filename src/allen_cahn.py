import jraph
import jax
import jax.numpy as np
import numpy as onp
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


# @partial(jax.jit, static_argnums=(2,))
def explicit_euler(state, t_crt, f, *diff_args):
    y_prev, t_prev = state
    h = t_crt - t_prev
    y_crt = y_prev + h * f(y_prev, t_prev, *diff_args)
    return (y_crt, t_crt), y_crt


def odeint(stepper, f, y0, ts, *diff_args):
    ys = []
    state = (y0, ts[0])
    for (i, t_crt) in enumerate(ts[1:]):
        state, y = stepper(state, t_crt, f, *diff_args)
        if i % 20 == 0:
            print(f"step {i}")
            print(y[:10, :10])
            print(f"max T is {np.max(y[:, 0])}")
            inspect(y, y0)
            if not np.all(np.isfinite(y)):
                print(f"Found np.inf or np.nan in y - stop the program")             
                exit()
        ys.append(y)
    ys = np.array(ys)
    return ys


def inspect(y, y0):
    eta0 = np.argmax(y0[:, 2:], axis=1)
    eta = np.argmax(y[:, 2:], axis=1)
    change_eta = np.where(eta0 == eta, 0, 1)
    zeta = y[:, 1]
    change_zeta = np.where(zeta < 0.1, 1, 0)
    T = y[:, 0]
    change_T = np.where(T > args.T_melt, 1, 0)

    print(f"percet of T > T_melt = {np.sum(change_T)/len(change_T)*100}%")
    print(f"percet of liquid = {np.sum(change_zeta)/len(change_zeta)*100}%")
    print(f"percent of change of oris = {np.sum(change_eta)/len(change_eta)*100}%")


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

    temp = 300.*np.ones(args.num_grains)
    senders = np.array(senders)
    receivers = np.array(receivers)

    state = np.hstack((temp[:, None], solid_phases[:, None], grain_orientations))

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
        sender_T, sender_zeta, sender_eta = unpack_state(senders['state'])
        receiver_T, receiver_zeta, receiver_eta = unpack_state(receivers['state'])
        face_areas = edges['face_areas']
        grain_distances = edges['grain_distances']

        assert face_areas.shape == grain_distances.shape
 
        # coeff_T = 5*1e-2
        coeff_T = args.kappa/4.
        grad_energy_T = coeff_T * np.sum((sender_T - receiver_T)**2 * face_areas / grain_distances)
        coeff_zeta = 1e-2
        grad_energy_zeta = coeff_zeta * np.sum((sender_zeta - receiver_zeta)**2 * face_areas / grain_distances)
        coeff_eta = 1e-2
        grad_energy_eta = coeff_eta * np.sum((sender_eta - receiver_eta)**2 * face_areas / grain_distances)
        grad_energy = grad_energy_T + grad_energy_zeta + grad_energy_eta

        return {'grad_energy': grad_energy}

    def update_node_fn(nodes, sent_edges, received_edges, globals_):
        del sent_edges, received_edges

        t = globals_['t'][0]
        T, zeta, eta = unpack_state(nodes['state'])
        boundary_face_areas = nodes['boundary_face_areas']
        boundary_face_centroids = nodes['boundary_face_centroids']
        volumes = nodes['volumes']
        centroids = nodes['centroids']
 
        T_ambient = 300.        
        # coeff_convection = 1.
        # q_ambient_convection = np.sum(coeff_convection*(T_ambient - T)*boundary_face_areas, axis=1)
        # coeff_radiation = 1e-7

        coeff_radiation = args.emissivity * args.SB_constant
        q_ambient_radiation = coeff_radiation*np.sum((T_ambient**4 - T**4)*boundary_face_areas, axis=1)
        q_ambient = q_ambient_radiation

        q_ambient = 0.

        # coeff_laser = 1e10
        coeff_laser = args.power * 3 / (np.pi * args.r_beam**2)

        length_scale = args.domain_width/5.
        # laser_pos = np.array([args.domain_length/2., args.domain_width/2., args.domain_height])
        # laser_gate = np.where(t < 0.1, 1., 0.)
        hafl_time = 600*1e-6
        laser_pos = np.array([t/hafl_time*0.6*args.domain_length + 0.2*args.domain_length, args.domain_width/2., args.domain_height])
        laser_gate = np.where(t < hafl_time, 1., 0.)

        # q_laser = laser_gate * coeff_laser * np.exp(-3 * np.sum((centroids[:, :2] - laser_pos[None, :2])**2, axis=1) / (args.r_beam**2)) * \
        #           2 / args.h_depth * (1 - (args.domain_height - centroids[:, 2]) / args.h_depth) * volumes.reshape(-1)

        q_laser = laser_gate * coeff_laser * np.exp(-3 * np.sum((centroids - laser_pos[None, :])**2, axis=1) / (args.r_beam**2)) * \
                  volumes.reshape(-1)

        # upper_face_centroids = boundary_face_centroids[:, -1, :]
        # upper_face_areas = boundary_face_areas[:, -1]
        # q_laser = laser_gate * coeff_laser * np.exp(-3 * np.sum((upper_face_centroids[:, :2] - laser_pos[None, :2])**2, axis=1) / (args.r_beam**2)) * \
        #           upper_face_areas

 
        # print(f"q_laser bound = {coeff_laser} x {np.max(volumes)} = {coeff_laser * np.max(volumes)}")
        print(f"np.max(q_laser) = {np.max(q_laser)}, np.mean(q_laser) = {np.mean(q_laser)}")

        # exit()

        # upper_face_centroids = boundary_face_centroids[:, -1, :]
        # upper_face_areas = boundary_face_areas[:, -1]
        # q_laser = laser_gate * coeff_laser * np.exp(-np.sum((upper_face_centroids - laser_pos[None, :])**2, axis=1) / (2 * length_scale**2)) * upper_face_areas

        q = q_ambient + q_laser # q shape (args.num_grains,)

        m_phase = 1e5
        phi = 0.5 * (1 - np.tanh(1e1*(T/args.T_melt - 1)))
        phase_energy = m_phase * np.sum(((1 - zeta)**2 * phi + zeta**2 * (1 - phi)) * volumes)
        m_grain = 1e5
        gamma = 1


        # beta = 1
        beta = 0.

        vmap_outer = jax.vmap(np.outer, in_axes=(0, 0))
        grain_energy_1 = np.sum((eta**4/4. - eta**2/2.) * volumes)
        graph_energy_2 = gamma * (np.sum(np.sum(vmap_outer(eta, eta)**2, axis=(1, 2))[:, None] * volumes) - np.sum(eta**4 * volumes))
        graph_energy_3 = beta * np.sum((1 - zeta.reshape(-1))**2 * np.sum(eta**2, axis=1) * volumes.reshape(-1))
 
        grain_energy = m_grain * (grain_energy_1 +  graph_energy_2 + graph_energy_3)
        local_energy = phase_energy + grain_energy

        return {'heat_source': q, 'local_energy': local_energy}

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
    # TODO: seems ugly
    volumes = graph.nodes['volumes']

    # print(np.min(volumes))
    # print(np.max(volumes))
    # exit()


    def compute_energy(y, t):
        graph.globals['t'] = t
        graph.nodes['state'] = y
        new_graph = net_fn(graph)
        return new_graph.globals['total_energy'][0], new_graph.nodes['heat_source']

    grad_energy = jax.grad(lambda y, t: compute_energy(y, t)[0])

    def state_rhs(y, t, *diff_args):
        _, source = compute_energy(y, t)
        grads = grad_energy(y, t)

        T_rhs = (source - grads[:, 0])[:, None] / (args.rho * args.c_h * volumes)


        a = source[:, None] / (args.rho * args.c_h * volumes) * 1e-6
        b = grads[:, 0][:, None] / (args.rho * args.c_h * volumes) * 1e-6
 
        print(f"max a = {np.max(a)}, mean a = {np.mean(a)},")
        print(f"max b = {np.max(b)}, mean b = {np.mean(b)},")
        # print(f"bound a = {args.power * 3 / (np.pi * args.r_beam**2) * 1e-6 / (args.rho * args.c_h)}")

        # exit()

        rhs = np.hstack((T_rhs, -grads[:, 1:]))
        return rhs

    return state_rhs


def simulate(ts):
    polycrystal = construct_polycrystal()
    graph, y0 = build_graph(polycrystal)
    state_rhs = phase_field(graph)
    ys_ = odeint(explicit_euler, state_rhs, y0, ts)
    ys = np.vstack((y0[None, :], ys_))  


    T_final = ys[-1, :, 0]
    zeta_final = ys[-1, :, 1]
    eta_final = ys[-1, :, 2:]

    onp.savetxt(f'data/neper/temp', T_final)
    onp.savetxt(f'data/neper/phase', zeta_final)
    oris_indics = np.argmax(eta_final, axis=1)
    oris = onp.take(polycrystal.unique_oris, oris_indics, axis=0)
    onp.savetxt(f'data/neper/oris', oris)

    return ys, polycrystal


def exp():
    # dt = 1e-4
    # ts = np.arange(0., dt*2001, dt)

    dt = 1e-6
    ts = np.arange(0., dt*1201, dt)

    ys, polycrystal = simulate(ts)

    # show_3d_scatters(ys[-1, :, :], polycrystal)
 
    show_3d_scatters(ys[1000, :, :], polycrystal)
    show_3d_scatters(ys[2000, :, :], polycrystal)

    # save_animation(ys[::20], polycrystal)


def show_3d_scatters(y, polycrystal):
    x1, x2, x3 = polycrystal.centroids.T
    T = y[:, 0]
    zeta = y[:, 1]
    eta = y[:, 2:]
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
    p = ax.scatter3D(x1_show, x2_show, x3_show, s=2, c=T_show, alpha=1, vmin=280, vmax=2000)
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
    exp()
    plt.show()
    # show_3d_scatters()
