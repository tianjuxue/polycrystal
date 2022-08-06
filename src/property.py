'''
We tried to use Neper sister software FEPX (https://fepx.info/) or DAMASK for crystal plasticity anlysis.
'''

def selected_cube_hex():
    '''
    This function produces input files for DAMASK (and for the OSU folks).
    Since we're using MOOSE now instead of DAMASK, this function should be deprecated.
    See https://github.com/tianjuxue/cp_gnn
    '''
    property_name = 'property_damask'
    
    offset_x = 0.5
    offset_y = 0.1
    offset_z = 0.05

    neper_create_cube = True
    if neper_create_cube:
        select_length = 0.2
        select_width = 0.2
        select_height = 0.05

        os.system(f'neper -T -n 1 -reg 0 -domain "cube({select_length},{select_width},{select_height})" -o data/neper/{property_name}/simple -format tess')
        os.system(f'neper -M -rcl 0.1 -elttype hex data/neper/{property_name}/simple.tess ')


    filepath_raw = f'data/neper/single_layer/domain.msh'

    mesh = meshio.read(filepath_raw)
    points = mesh.points
    cells =  mesh.cells_dict['hexahedron']
    
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

    filepath_neper = f'data/neper/{property_name}/simple.msh'
    mesh = meshio.read(filepath_neper)
    points = mesh.points

    cells =  mesh.cells_dict['hexahedron']
    cell_points = onp.take(points, cells, axis=0)
    order2_hex_centroids = onp.mean(cell_points, axis=1)
    indx = onp.round((order2_hex_centroids[:, 0] + offset_x - min_x - tick_x / 2.) / tick_x)
    indy = onp.round((order2_hex_centroids[:, 1] + offset_y - min_y - tick_y / 2.) / tick_y)
    indz = onp.round((order2_hex_centroids[:, 2] + offset_z - min_z - tick_z / 2.) / tick_z)
    total_ind = onp.array(indx + indy * Nx + indz * Nx * Ny, dtype=np.int32)


    def helper(case, step):
        print(f"Processing case {case} and step {step}")
        if case == 'fd_single_layer':
            cell_ori_inds = onp.load(f"data/numpy/{case}/sols/cell_ori_inds_{step:03d}.npy")
        else:
            grain_oris_inds = onp.load(f"data/numpy/{case}/sols/cell_ori_inds_{step:03d}.npy")
            cell_grain_inds = onp.load(f"data/numpy/fd_single_layer/info/cell_grain_inds.npy")
            cell_ori_inds = onp.take(grain_oris_inds, cell_grain_inds, axis=0)

        order2_cell_ori_inds = onp.take(cell_ori_inds, total_ind, axis=0)

        file_to_read = open(filepath_neper, 'r')
        lines = file_to_read.readlines()
        new_lines = []
        flag = True
        for i, line in enumerate(lines):
            l = line.split()
            # TODO: dirty
            if len(l) == 14:
                if flag: 
                    offset_cell_ind = int(l[0])
                    flag = False

                ori_ind = order2_cell_ori_inds[int(l[0]) - offset_cell_ind]
                l[3] = str(ori_ind)
                l[4] = str(ori_ind)
                new_line = " ".join(l) + "\n"
            else:
                new_line = line
            new_lines.append(new_line)

        ori_quat = onp.load(f"data/numpy/quat.npy")
        file_to_write = f'data/neper/{property_name}/simulation_{case}_{step:03d}.msh'
        with open(file_to_write, 'w') as f:
            for i, line in enumerate(new_lines): 
                l = line.split()
                if l[0] == "$ElsetOrientations":
                    break
                # TODO: dirty
                elif len(l) == 4:
                    l[1] = str(float(l[1]) + offset_x)
                    l[2] = str(float(l[2]) + offset_y)
                    l[3] = str(float(l[3]) + offset_z)
                    f.write(" ".join(l) + "\n")
                else:
                    f.write(line)
            f.write(f"$ElsetOrientations\n{len(ori_quat)} quaternion:active\n")
            for i, line in enumerate(ori_quat):
                f.write(f"{i + 1}")
                for q in line:
                    f.write(f" {q}")
                f.write("\n")
            f.write(f"$EndElsetOrientations\n")

        mesh = meshio.read(file_to_write)
        mesh.write(f"data/neper/{property_name}/simulation_{case}_{step:03d}.vtu")

    helper('gn_single_layer', 30)
    helper('fd_single_layer', 30)
    helper('fd_single_layer', 0)


if __name__ == "__main__":
    selected_cube_hex()
