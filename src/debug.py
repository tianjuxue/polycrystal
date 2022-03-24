import meshio
import jax
import jax.numpy as np


def exp():
    # filepath = f'data/neper/debug/simple.msh'
    # mesh = meshio.read(filepath)
    # mesh.write(f'data/vtk/simple.vtk')

    # filepath = f'data/neper/domain.msh'
    # mesh = meshio.read(filepath)
    # mesh.write(f'data/vtk/domain.vtk')
    

    filepath = f'data/vtk/sols/u_final.vtk'
    mesh = meshio.read(filepath)
    mesh.write(filepath)
    

if __name__ == "__main__":
    exp()
  