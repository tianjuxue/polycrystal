import jraph
import jax
import jax.numpy as np
import numpy as onp
import os
import pickle
from functools import partial
from matplotlib import pyplot as plt
import microstructpy as msp
from src.arguments import args
from src.plots import poly_plot, plot_polygon_mesh, save_animation
from src.utils import unpack_state


def compute_centroids(polygon_mesh):
    '''
    Find the centroids of grains
    The package MicroStructPy does not provide this function, so we write it by ourselves.
    Code is adapted from
    https://github.com/kip-hart/MicroStructPy/blob/master/src/microstructpy/meshing/polymesh.py#L154
    '''
    centroids = []
    n = len(polygon_mesh.points[0])
    for i, region in enumerate(polygon_mesh.regions):
        cen = onp.array(polygon_mesh.points)[polygon_mesh.facets[region[0]][0]]
        centroid = onp.zeros(n)
        vol = 0.
        for f_num in region:
            facet = onp.array(polygon_mesh.facets[f_num])
            j_max = len(facet) - n + 2
            for j in range(1, j_max):
                inds = onp.append(onp.arange(j, j + n - 1), 0)
                simplex = facet[inds]
                facet_pts = onp.array(polygon_mesh.points)[simplex]
                rel_pos = facet_pts - cen
                vertices = onp.vstack((facet_pts, cen[None, :]))
                sub_vol = onp.abs(onp.linalg.det(rel_pos))
                centroid += onp.mean(vertices, axis=0)*sub_vol
                vol += sub_vol
        centroids.append(centroid/vol)
 
    return onp.array(centroids)


def generate_microsctructure():
    phase = {'shape': 'circle', 'size': 0.1}
    domain = msp.geometry.Rectangle(length=5., width=1.)

    pickle_path = f'data/pickle/seeds.pkl'
    cache = os.path.isfile(pickle_path)
    if cache:
        with open(pickle_path, 'rb') as handle:
            seeds = pickle.load(handle)
    else:
        seeds = msp.seeding.SeedList.from_info(phase, domain.area)
        seeds.position(domain)
        with open(pickle_path, 'wb') as handle:
            pickle.dump(seeds, handle, protocol=pickle.HIGHEST_PROTOCOL)   

    polygon_mesh = msp.meshing.PolyMesh.from_seeds(seeds, domain)
    # Add a few features that we need
    polygon_mesh.num_orientations = 20
    polygon_mesh.orientations = onp.random.randint(polygon_mesh.num_orientations, size=len(polygon_mesh.regions))
    polygon_mesh.centroids = compute_centroids(polygon_mesh)

    # plot_polygon_mesh(polygon_mesh, variable='phase')

    return polygon_mesh


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


def odeint(polygon_mesh, stepper, f, y0, ts, *diff_args):
    ys = []
    state = (y0, ts[0])
    for (i, t_crt) in enumerate(ts[1:]):

        state, y =  stepper(state, t_crt, f, *diff_args)
        if i % 20 == 0:
            print(f"step {i}")
            if not np.all(np.isfinite(y)):
                print(f"Found np.inf or np.nan in y - stop the program")             
                exit()
        ys.append(y)
    ys = np.array(ys)
    return ys


def build_graph(polygon_mesh):
    num_grains = len(polygon_mesh.regions)
    num_orientations = polygon_mesh.num_orientations
    senders = []
    receivers = []

    for edge in polygon_mesh.facet_neighbors:
        if edge[0] >=0 and edge[1] >= 0:
            senders += list(edge)
            receivers += list(edge[::-1])

    n_node = np.array([num_grains])
    n_edge = np.array([len(senders)])

    print(f"Total number nodes = {n_node[0]}, total number of edges = {n_edge[0]}")

    solid_phases = np.ones(num_grains)
    grain_orientations = np.zeros((num_grains, num_orientations))
    inds = jax.ops.index[np.arange(num_grains), polygon_mesh.orientations]
    grain_orientations = jax.ops.index_update(grain_orientations, inds, 1)
    temp = 300.*np.ones(num_grains)
    senders = np.array(senders)
    receivers = np.array(receivers)

    state = np.hstack((temp[:, None], solid_phases[:, None], grain_orientations))

    node_features = {'state':state, 'centroids': polygon_mesh.centroids}
    global_features = {'t': 0.}
    graph = jraph.GraphsTuple(nodes=node_features, edges={}, senders=senders, receivers=receivers,
        n_node=n_node, n_edge=n_edge, globals=global_features)

    return graph, state


def update_graph():

    def update_edge_fn(edges, senders, receivers, globals_):
        del edges, globals_
        sender_T, sender_zeta, sender_eta = unpack_state(senders['state'])
        receiver_T, receiver_zeta, receiver_eta = unpack_state(receivers['state'])

        coeff_T = 1e-2
        grad_energy_T = coeff_T * np.sum((sender_T - receiver_T)**2)
        coeff_zeta = 1e-3
        grad_energy_zeta = coeff_zeta * np.sum((sender_zeta - receiver_zeta)**2)
        coeff_eta = 1e-3
        grad_energy_eta = coeff_eta * np.sum((sender_eta - receiver_eta)**2)
        grad_energy = grad_energy_T + grad_energy_zeta + grad_energy_eta

        return {'grad_energy': grad_energy}

    def update_node_fn(nodes, sent_edges, received_edges, globals_):
        del sent_edges, received_edges

        t = globals_['t'][0]
        T, zeta, eta = unpack_state(nodes['state'])
        centroids = nodes['centroids']

        T_melt = 800.
        T_ambient = 300.        
        coeff_ambient = 5.
        q_ambient = coeff_ambient*(T_ambient - T).reshape(-1)
        coeff_laser = 2*1e5
        length_scale = 0.15
        # laser_pos = np.array([0., 0.])
        # laser_gate = np.where(t < 0.02, 1., 0.)
        laser_pos = np.array([t/0.1*4 - 2., 0.])
        laser_gate = np.where(t < 0.1, 1., 0.)
        q_laser = laser_gate * coeff_laser * np.exp(-np.sum((centroids - laser_pos[None, :])**2, axis=1) / (2 * length_scale**2))
        q = q_ambient + q_laser

        m_phase = 1e1
        phi = 0.5 * (1 - np.tanh(1e1*(T/T_melt - 1)))
        phase_energy = m_phase * np.sum(((1 - zeta)**2 * phi +  zeta**2 * (1 - phi)))
        m_grain = 1e1
        gamma = 1.
        vmap_outer = jax.vmap(np.outer, in_axes=(0, 0))
        grain_energy = m_grain * (np.sum(eta**4/4. - eta**2/2.) + gamma*(np.sum(vmap_outer(eta, eta)**2) - np.sum(eta**4)) + 
                       np.sum((1 - zeta.reshape(-1))**2 * np.sum(eta**2, axis=1)))
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

    def compute_energy(y, t):
        graph.globals['t'] = t
        graph.nodes['state'] = y
        new_graph = net_fn(graph)
        return new_graph.globals['total_energy'][0], new_graph.nodes['heat_source']

    grad_grad = jax.grad(lambda y, t: compute_energy(y, t)[0])

    def state_rhs(y, t, *diff_args):
        _, source = compute_energy(y, t)
        grads = grad_grad(y, t)
        rhs = np.hstack(((source - grads[:, 0])[:, None], -grads[:, 1:]))
        return rhs

    return state_rhs


def simulate(ts):
    polygon_mesh = generate_microsctructure()
    graph, y0 = build_graph(polygon_mesh)
    state_rhs = phase_field(graph)
    ys_ = odeint(polygon_mesh, explicit_euler, state_rhs, y0, ts)
    ys = np.vstack((y0[None, :], ys_))  
    return ys, polygon_mesh


def exp():
    dt = 1e-4
    ts = np.arange(0., dt*2001, dt)
    ys, polygon_mesh = simulate(ts)
    save_animation(ys[::20], polygon_mesh)


if __name__ == "__main__":
    exp()
    plt.show()


