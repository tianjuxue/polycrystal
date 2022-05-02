import numpy as onp
import os
from src.utils import obj_to_vtu, read_path, walltime
from src.arguments import args
from src.allen_cahn import polycrystal_gn, PolyCrystal, build_graph, phase_field, odeint, explicit_euler, default_initialization, layered_initialization
import copy
import meshio


def set_params():
    args.num_grains = 100000
    args.domain_length = 2.
    args.domain_width = 2.
    args.domain_height = 0.025
    args.write_sol_interval = 5000
    args.case = 'part'    


def neper_domain():
    set_params()
    os.system(f'neper -T -n {args.num_grains} -id 1 -domain "cube({args.domain_length},{args.domain_width},{args.domain_height})" \
                -o data/neper/part/domain -format tess,obj,ori')
    os.system(f'neper -T -loadtess data/neper/part/domain.tess -statcell x,y,z,vol,facelist -statface x,y,z,area')
   

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
    print("Merging two meshes...")
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
    print("Merging two polycrystal domains...")

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
def run_helper():
    print(f"Merge into poly layer")
    poly1, mesh1  = polycrystal_gn('part')
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
    ts, xs, ys, ps = read_path(f'data/txt/NU.txt')
    odeint(poly_sim, mesh_sim,  bottom_mesh, explicit_euler, state_rhs, y0, melt, ts, xs, ys, ps)


def run():
    set_params()
    for i in range(0, 20):
        print(f"\nLayer {i + 1}...")
        args.layer = i + 1
        onp.random.seed(args.layer)
        run_helper()


# def exp():
#     set_params()
#     poly1, mesh1 = polycrystal_gn('part')
#     poly2 = copy.deepcopy(poly1)
#     mesh2 = copy.deepcopy(mesh1)
#     lift_poly(poly2, args.domain_height)
#     flip_mesh(mesh2, args.domain_height)
#     mesh_merged = merge_mesh(mesh1, mesh2)
#     poly_merged = merge_poly(poly1, poly2)
#     # print(len(poly_merged.volumes))
#     print(f"write to solutions")


if __name__ == "__main__":
    # neper_domain()
    run()
