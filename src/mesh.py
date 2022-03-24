import meshio


def exp():
    # filepath = f'data/neper/debug/simple.msh'
    # mesh = meshio.read(filepath)
    # mesh.write(f'data/vtk/simple.vtk')

    filepath = f'data/neper/domain.msh'
    mesh = meshio.read(filepath)
    mesh.write(f'data/vtk/domain.vtk')

if __name__ == "__main__":
    exp()
  