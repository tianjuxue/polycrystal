import numpy as onp
import jax
import jax.numpy as np
import os
from src.utils import obj_to_vtu, read_path, walltime
from src.arguments import args
from src.allen_cahn import polycrystal_gn, PolyCrystal, build_graph, phase_field, odeint, explicit_euler
import copy
import meshio


def set_params():
    args.num_grains = 100000
    args.domain_length = 2.
    args.domain_width = 2.
    args.domain_height = 0.025
    args.write_sol_interval = 10000


def neper_domain():
    set_params()
    os.system(f'neper -T -n {args.num_grains} -id 1 -domain "cube({args.domain_length},{args.domain_width},{args.domain_height})" \
                -o data/neper/multi_layer/domain -format tess,obj,ori')
    os.system(f'neper -T -loadtess data/neper/multi_layer/domain.tess -statcell x,y,z,vol,facelist -statface x,y,z,area')
   

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
    # Current lower layer is previous upper layer
    y_down = np.load(f'data/numpy/{args.case}/sols/layer_{args.layer - 1:03d}/y_final_top.npy')
    melt_down = np.load(f'data/numpy/{args.case}/sols/layer_{args.layer - 1:03d}/melt_final_top.npy')
    return np.vstack((y_down, y_top)), np.hstack((melt_down, melt_top))


def lift_poly(poly, delta_z):
    poly.boundary_face_centroids[:, :, 2] = poly.boundary_face_centroids[:, :, 2] + delta_z
    poly.centroids[:, 2] = poly.centroids[:, 2] + delta_z
    poly.meta_info[2] = poly.meta_info[2] + delta_z


def flip_poly(poly, base_z):
    new_boundary_face_areas = onp.hstack((poly.boundary_face_areas[:, :4], poly.boundary_face_areas[:, 5:6], poly.boundary_face_areas[:, 4:5]))
    poly.boundary_face_areas[:] = new_boundary_face_areas

    new_boundary_face_centroids = onp.array(poly.boundary_face_centroids)
    new_boundary_face_centroids[:, :, 2] = 2*base_z - new_boundary_face_centroids[:, :, 2]
    new_boundary_face_centroids = onp.concatenate((new_boundary_face_centroids[:, :4, :], new_boundary_face_centroids[:, 5:6, :], 
        new_boundary_face_centroids[:, 4:5, :]), axis=1)
    poly.boundary_face_centroids[:] = new_boundary_face_centroids

    poly.centroids[:, 2] = 2*base_z - poly.centroids[:, 2] 
    poly.meta_info[2] = 2*base_z -  poly.meta_info[2] -  poly.meta_info[5]


def lift_mesh(mesh, delta_z):
    mesh.points[:, 2] = mesh.points[:, 2] + delta_z


def flip_mesh(mesh, base_z):
    mesh.points[:, 2] = 2*base_z - mesh.points[:, 2]  


def merge_mesh(mesh1, mesh2):
    '''
    Merge two meshes
    '''
    print("Merge two meshes...")
    points1 = mesh1.points
    points2 = mesh2.points 
    cells1 = mesh1.cells_dict['polyhedron']
    cells2 = mesh2.cells_dict['polyhedron']

    num_points1 = len(points1)

    for cell in cells2:
        for face in cell:
            for i in range(len(face)):
                face[i] += num_points1

    points_merged = onp.vstack((points1, points2))
    cells_merged = [('polyhedron', onp.concatenate((cells1, cells2)))]
    mesh_merged = meshio.Mesh(points_merged, cells_merged)
    return mesh_merged


def merge_poly(poly1, poly2):
    '''
    Merge two polycrystals: poly2 should exactly sits on top of poly1
    '''
    print("Merge two polycrystal domains...")

    poly1_top_z = poly1.meta_info[2] + poly1.meta_info[5] 
    poly2_bottom_z = poly2.meta_info[2]
    num_nodes1 = len(poly1.volumes)
    num_nodes2 = len(poly2.volumes)

    assert onp.isclose(poly1_top_z, poly2_bottom_z, atol=1e-8)

    inds1 = onp.argwhere(poly1.boundary_face_areas[:, 5] > 0).reshape(-1)
    inds2 = onp.argwhere(poly2.boundary_face_areas[:, 4] > 0).reshape(-1)

    face_areas1 = onp.take(poly1.boundary_face_areas[:, 5], inds1)
    face_areas2 = onp.take(poly2.boundary_face_areas[:, 4], inds2)

    assert onp.isclose(onp.sum(onp.absolute(face_areas1 - face_areas2)), 0., atol=1e-8)

    grain_distances = 2 * (poly1_top_z - onp.take(poly1.centroids[:, 2], inds1))
    ch_len_interface = face_areas1 / grain_distances
    edges_interface = onp.stack((inds1, inds2 + num_nodes1)).T
    edges_merged =  onp.vstack((poly1.edges, poly2.edges + num_nodes1, edges_interface)) 
    ch_len_merged = onp.hstack((poly1.ch_len, poly2.ch_len, ch_len_interface))

    boundary_face_areas1 = onp.hstack((poly1.boundary_face_areas[:, :5], onp.zeros((num_nodes1, 1))))
    boundary_face_areas2 = onp.hstack((poly2.boundary_face_areas[:, :4], onp.zeros((num_nodes2, 1)), poly2.boundary_face_areas[:, 5:6]))
    boundary_face_areas_merged = onp.vstack((boundary_face_areas1, boundary_face_areas2))

    boundary_face_centroids_merged = onp.concatenate((poly1.boundary_face_centroids, poly2.boundary_face_centroids), axis=0)
    volumes_merged = onp.hstack((poly1.volumes, poly2.volumes))
    centroids_merged = onp.vstack((poly1.centroids, poly2.centroids))
    cell_ori_inds_merged = onp.hstack((poly1.cell_ori_inds, poly2.cell_ori_inds))

    meta_info = onp.hstack((poly1.meta_info[:5], poly1.meta_info[5:] + poly2.meta_info[5:]))

    poly_merged = PolyCrystal(edges_merged, ch_len_merged, centroids_merged, volumes_merged, poly1.unique_oris_rgb, poly1.unique_grain_directions,
                              cell_ori_inds_merged, boundary_face_areas_merged, boundary_face_centroids_merged, meta_info)

    return poly_merged


def randomize_oris(poly, seed):
    onp.random.seed(seed)
    cell_ori_inds = onp.random.randint(args.num_oris, size=len(poly.volumes))
    poly.cell_ori_inds[:] = cell_ori_inds


# @walltime
def run_helper(path):
    print(f"Merge into poly layer")
    poly1, mesh1  = polycrystal_gn('multi_layer')
    N_random = 100
    randomize_oris(poly1, N_random*args.layer + 1)
    poly2 = copy.deepcopy(poly1)
    randomize_oris(poly2, N_random*args.layer + 2)

    mesh2 = copy.deepcopy(mesh1)
    flip_poly(poly2, poly1.meta_info[2] + poly1.meta_info[5])
    flip_mesh(mesh2, poly1.meta_info[2] + poly1.meta_info[5])
    mesh_layer1 = merge_mesh(mesh1, mesh2)
    poly_layer1 = merge_poly(poly1, poly2)
    args.layer_num_dofs = len(poly_layer1.volumes)
    args.layer_height = poly_layer1.meta_info[5]

    # mesh_layer1.write(f'data/vtk/part/domain.vtu')

    print(f"Merge into poly sim")
    poly_layer2 = copy.deepcopy(poly_layer1)
    randomize_oris(poly_layer2, N_random*args.layer + 3)

    mesh_layer2 = copy.deepcopy(mesh_layer1)
    lift_poly(poly_layer2, poly_layer1.meta_info[5])
    lift_mesh(mesh_layer2, poly_layer1.meta_info[5])
    mesh_sim = merge_mesh(mesh_layer1, mesh_layer2)
    poly_sim = merge_poly(poly_layer1, poly_layer2)

    poly_top_layer = poly_layer2
    bottom_mesh = mesh_layer1

    lift_val = (args.layer - 1) * args.layer_height
    lift_poly(poly_sim, lift_val)
    lift_poly(poly_top_layer, lift_val)
    lift_mesh(mesh_sim, lift_val)
    lift_mesh(bottom_mesh, lift_val)

    if args.layer == 1:
        y0, melt = default_initialization(poly_sim)
    else:
        y0, melt = layered_initialization(poly_top_layer)

    graph = build_graph(poly_sim, y0)
    state_rhs = phase_field(graph, poly_sim)
    # This is how you generate NU.txt
    # traveled_time = onp.cumsum(onp.array([0., 0.6, (0.6**2 + 0.3**2)**0.5, 0.6, 0.2, 0.6, 0.3, 0.6, 0.4]))/500.
    ts, xs, ys, ps = read_path(path)
    odeint(poly_sim, mesh_sim,  bottom_mesh, explicit_euler, state_rhs, y0, melt, ts, xs, ys, ps)


def write_info():
    args.case = 'gn_multi_layer_scan_2'
    set_params()
    print(f"Merge into poly layer")
    poly1, mesh1  = polycrystal_gn('multi_layer')
    poly2 = copy.deepcopy(poly1)
    flip_poly(poly2, poly1.meta_info[2] + poly1.meta_info[5])
    poly_layer1 = merge_poly(poly1, poly2)
    args.layer_height = poly_layer1.meta_info[5]

    crt_layers = copy.deepcopy(poly_layer1)

    args.num_total_layers = 10
    for i in range(1, args.num_total_layers):
        print(f"Merge layer {i + 1} into current {i} layers")
        poly_layer_new = copy.deepcopy(poly_layer1)
        lift_poly(poly_layer_new, args.layer_height * i)
        crt_layers = merge_poly(crt_layers, poly_layer_new)

    onp.save(f"data/numpy/{args.case}/info/edges.npy", crt_layers.edges)
    onp.save(f"data/numpy/{args.case}/info/vols.npy", crt_layers.volumes)
    onp.save(f"data/numpy/{args.case}/info/centroids.npy", crt_layers.centroids)


def run_NU():
    args.case = 'gn_multi_layer_NU'
    set_params()
    args.num_total_layers = 20
    for i in range(num_total_layers):
        print(f"\nLayer {i + 1}...")
        args.layer = i + 1
        onp.random.seed(args.layer)
        run_helper(f'data/txt/{args.case}.txt')


def run_scans_1():
    args.case = 'gn_multi_layer_scan_1'
    set_params()
    args.num_total_layers = 10
    for i in range(args.num_total_layers - 1, args.num_total_layers):
        print(f"\nLayer {i + 1}...")
        args.layer = i + 1
        onp.random.seed(args.layer)
        run_helper(f'data/txt/{args.case}.txt')


def rotate(points, angle, center):
    rot_mat = onp.array([[onp.cos(angle), -onp.sin(angle)], [onp.sin(angle), onp.cos(angle)]])
    return onp.matmul(rot_mat, (points - center[None, :]).T).T + center[None, :]


def run_scans_2():
    args.case = 'gn_multi_layer_scan_2'
    set_params()
    path1 = f'data/txt/{args.case}-1.txt'
    path2 = f'data/txt/{args.case}-2.txt'
    path_info = onp.loadtxt(path1)
    center = onp.array([args.domain_length/2., args.domain_width/2.])
    rotated_points = rotate(path_info[:, 1:3], onp.pi/2, center)
    onp.savetxt(path2, onp.hstack((path_info[:, :1], rotated_points, path_info[:, -1:])), fmt='%.5f')

    args.num_total_layers = 10
    for i in range(args.num_total_layers - 1, args.num_total_layers):
        print(f"\nLayer {i + 1}...")
        args.layer = i + 1
        onp.random.seed(args.layer)
        if i % 2 == 0:
            run_helper(path1)
        else:
            run_helper(path2)


if __name__ == "__main__":
    # neper_domain()
    # write_info()
    # run_NU()
    run_scans_1()
    run_scans_2()
