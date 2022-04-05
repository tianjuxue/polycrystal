import numpy as onp
import jraph
import jax.numpy as np
import jax
import pymesh
import meshio
import gmsh
import os
import fenics as fe
import mshr


def test_jraph():

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




def selected_cube():
    os.system(f'neper -T -n 1 -reg 0 -o data/neper/inspect/simple -format tess')
    os.system(f'neper -M data/neper/inspect/simple.tess -rcl 0.3 -order 2')


def convert():
    pass


# This function will be deprecated!
def convert():
    filepath_raw = f'data/neper/debug/simple.msh'
    mesh = meshio.read(filepath_raw)
    points = mesh.points
    cells =  mesh.cells_dict['hexahedron']
    cell_grain_inds = mesh.cell_data['gmsh:physical'][0]
    cell_points = onp.take(points, cells, axis=0)
    centroids = onp.mean(cell_points, axis=1)

    min_x, min_y, min_z = onp.min(points[:, 0]), onp.min(points[:, 1]), onp.min(points[:, 2])
    max_x, max_y, max_z = onp.max(points[:, 0]), onp.max(points[:, 1]), onp.max(points[:, 2])
    domain_length = max_x - min_x
    domain_width = max_y - min_y
    domain_height = max_z - min_z

    Nx = round(domain_length / (points[1, 0] - min_x))
    Ny = round(domain_width / (points[Nx + 1, 1]) - min_y)
    Nz = round(domain_height / (points[(Nx + 1)*(Ny + 1), 2]) - min_z)
    tick_x, tick_y, tick_z =  domain_length / Nx, domain_width / Ny, domain_height / Nz  

    assert Nx*Ny*Nz == len(cells)

    print(cell_grain_inds.shape)

    # TODO: Why not just use order2 mesh?
    filepath_order1 = f'data/neper/inspect/v1.msh'
    mesh = meshio.read(filepath_order1)
    points = mesh.points
    cells =  mesh.cells_dict['tetra']
    cell_points = onp.take(points, cells, axis=0)
    order2_tet_centroids = onp.mean(cell_points, axis=1)

    indx = onp.round((order2_tet_centroids[:, 0] - min_x - tick_x / 2.) / tick_x)
    indy = onp.round((order2_tet_centroids[:, 1] - min_y - tick_y / 2.) / tick_y)
    indz = onp.round((order2_tet_centroids[:, 2] - min_z - tick_z / 2.) / tick_z)
    total_ind = onp.array(indx + indy * Nx + indz * Nx * Ny, dtype=np.int32)
    order2_cell_grain_inds = onp.take(cell_grain_inds, total_ind, axis=0)

    print(onp.min(total_ind))
    print(onp.max(total_ind))
    print(order2_cell_grain_inds)
    print(len(order2_cell_grain_inds))

    filepath_order2 = f'data/neper/inspect/v2.msh'
    points = []
    cells = []
    order2_file = open(filepath_order2, 'r')
    lines = order2_file.readlines()
    mode = None
    for i, line in enumerate(lines):
        l = line.split()
        if l[0] == '$Nodes':
            mode = 'points'
        if l[0] == '$Elements':
            mode = 'cells'
        if mode == 'points' and len(l) > 1:
            points.append([float(item) for item in l[1:]])
        if mode == 'cells' and len(l) > 1:
            cells.append([int(item) for item in l[5:]])

    print(f"len(points) = {len(points)}")
    print(f"len(cells) = {len(cells)}")

    assert len(cells) == len(order2_cell_grain_inds)

    points = onp.array(points)
    EPS = 1e-10
    x0 = (onp.argwhere(points[:, 0] < min_x + EPS)).reshape(-1)
    x1 = (onp.argwhere(points[:, 0] > max_x - EPS)).reshape(-1)
    y0 = (onp.argwhere(points[:, 1] < min_y + EPS)).reshape(-1)
    y1 = (onp.argwhere(points[:, 1] > max_y - EPS)).reshape(-1)
    z0 = (onp.argwhere(points[:, 2] < min_z + EPS)).reshape(-1)
    z1 = (onp.argwhere(points[:, 2] > max_z - EPS)).reshape(-1)

    domain_face_names = ['x0', 'x1', 'y0', 'y1', 'z0', 'z1'] 
    domain_face_values = [x0, x1, y0, y1, z0, z1]

    def get_order(line_pts, vertex_pts, ref_v):
        line_v1 = line_pts[1] - line_pts[0]
        line_v2 = line_pts[2] - line_pts[0]
        vertex_v1 = vertex_pts[1] - vertex_pts[0]
        vertex_v2 = vertex_pts[2] - vertex_pts[0]
        a, b, c = [0, 1, 2] if onp.dot(onp.cross(line_v1, line_v2), ref_v) > 0  else [0, 2, 1]
        d, e, f = [0, 1, 2] if onp.dot(onp.cross(vertex_v1, vertex_v2), ref_v) > 0 else [0, 2, 1]
        line_v_in_order = line_pts[b] - line_pts[a]
        tmp_v1 = vertex_pts[d] - line_pts[a]
        tmp_v2 = vertex_pts[e] - line_pts[a]
        tmp_v3 = vertex_pts[f] - line_pts[a]
        if onp.dot(onp.cross(line_v_in_order, tmp_v1), ref_v) < 0:
            return a, b, c, d, e, f
        elif onp.dot(onp.cross(line_v_in_order, tmp_v2), ref_v) < 0:
            return a, b, c, e, f, d
        elif onp.dot(onp.cross(line_v_in_order, tmp_v3), ref_v) < 0:
            return a, b, c, f, d, e
        else:
            raise ValueError

    face_x0 = []
    face_x1 = []
    face_y0 = []
    face_y1 = []
    face_z0 = []
    face_z1 = []

    faces = [face_x0, face_x1, face_y0, face_y1, face_z0, face_z1]

    cells = onp.array(cells) - 1
    cell_points = onp.take(points, cells, axis=0)
    ref_vs = np.array([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.], [0., 0., 1.], [0., 0., -1.]])

    print(cell_points.shape)

    for i in range(len(cells)):
        cell = cells[i]
        crt_cell_pts = cell_points[i]
        x0_inds = (onp.argwhere(crt_cell_pts[:, 0] < min_x + EPS)).reshape(-1)
        x1_inds = (onp.argwhere(crt_cell_pts[:, 0] > max_x - EPS)).reshape(-1)
        y0_inds = (onp.argwhere(crt_cell_pts[:, 1] < min_y + EPS)).reshape(-1)
        y1_inds = (onp.argwhere(crt_cell_pts[:, 1] > max_y - EPS)).reshape(-1)
        z0_inds = (onp.argwhere(crt_cell_pts[:, 2] < min_z + EPS)).reshape(-1)
        z1_inds = (onp.argwhere(crt_cell_pts[:, 2] > max_z - EPS)).reshape(-1)
        inds_list = [x0_inds, x1_inds, y0_inds, y1_inds, z0_inds, z1_inds]
        for j in range(len(inds_list)):
            inds = inds_list[j]
            face = faces[j]
            if len(inds) == 6:
                selected_pts = onp.take(crt_cell_pts, inds, axis=0)
                new_order = get_order(selected_pts[3:, :], selected_pts[:3, :], ref_vs[j])
                new_inds = [inds[3:][i] for i in new_order[:3]] + [inds[:3][i] for i in new_order[3:]]
                actual_inds = [i] + [cell[i] for i in new_inds]
                face.append(actual_inds)

    print(f"Starting writing to files...")

    fepx_file = f'data/neper/inspect/simulation.msh'
    with open(fepx_file, 'w') as f:
        f.write(f'$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n{len(points)}\n')
        for i, line in enumerate(points):
            f.write(str(i + 1))
            for coo in line:
                f.write(" " + str(coo))
            f.write('\n')
        f.write(f"$EndNodes\n$Elements\n{len(cells)}\n")
        for i, line in enumerate(cells):
            f.write(f"{i + 1} 11 3 {order2_cell_grain_inds[i]} {order2_cell_grain_inds[i]} 0")
            for ind in line:
                f.write(" " + str(ind + 1))
            f.write('\n')
        f.write(f"$EndElements\n$NSets\n6\n")

        for i, domain_face_name in enumerate(domain_face_names):
            domain_face_value = domain_face_values[i]
            f.write(f"{domain_face_name}\n{len(domain_face_value)}\n")
            for ind in domain_face_value:
                f.write(f"{ind + 1}\n")
        f.write(f"$EndNSets\n$Fasets\n6\n")

        for i, domain_face_name in enumerate(domain_face_names):
            domain_face_inds = faces[i]
            f.write(f"{domain_face_name}\n{len(domain_face_inds)}\n")
            for actual_inds in domain_face_inds:
                f.write(f"{actual_inds[0] + 1}")
                for ind in actual_inds[1:]:
                    f.write(f" {ind + 1}")
                f.write("\n")
        f.write(f"$EndFasets\n")
      
    # TODO: grain orienations

if __name__ == "__main__":
    # hex2tet()
    selected_cube()
    # convert()


