import numpy as onp
import jraph
import jax.numpy as np
import jax

def test():

    def update_edge_fn(edges, senders, receivers, globals_):
        energy = np.sum(senders['state']) + np.sum(receivers['state'])
        return {'energy': energy}


    def update_node_fn(node_features, aggregated_sender_edge_features, aggregated_receiver_edge_features, globals_):
        energy = np.sum(node_features['state'])
        return {'energy': 0.}


    def update_global_fn(nodes, edges, globals_):
        total_energy = edges['energy'] + nodes['energy']
        return {'total_energy': total_energy}


    node_features = {'state': np.array([[0.], [1.], [2.]])}

    senders = np.array([0, 1, 2])
    receivers = np.array([1, 2, 0])

    n_node = np.array([3])
    n_edge = np.array([3])
 
    graph = jraph.GraphsTuple(nodes=node_features, senders=senders, receivers=receivers,
    edges={}, n_node=n_node, n_edge=n_edge, globals={})

    net_fn = jraph.GraphNetwork(update_edge_fn=update_edge_fn, update_node_fn=update_node_fn, update_global_fn=update_global_fn)

    def func(nf):
        graph.nodes['state'] = nf
        new_graph = net_fn(graph)
        return new_graph.edges['energy']
        # return new_graph.globals['total_energy'][0]

    func_grad = jax.grad(func)

    nf = np.array([[0.], [1.], [2.]])
    print(func(nf))
    print(func_grad(nf))


if __name__ == "__main__":
    test()

