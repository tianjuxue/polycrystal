import fenics as fe
import numpy as np
from src.arguments import args


def simulation():
    domain_x = 1.
    domain_y = 0.2
    domain_z = 0.1
    total_t = 2400*1e-6
    ambient_T = args.T_ambient

    rho = args.rho

    Cp = args.c_h

    k = args.kappa_T

    h = args.h_conv

    eta = args.power_fraction
    r = args.r_beam
    P = args.power

    x0 = 0.2*args.domain_length
    y0 = 0.5*args.domain_width
    vel = 0.6*args.domain_length/total_t

    finer = False
    ele_size = 0.01
    dt = 2*1e-7
 
    EPS = 1e-8
    ts = np.arange(0., total_t + dt, dt)

    # Building mesh, see https://fenicsproject.org/olddocs/dolfin/1.5.0/python/programmers-reference/cpp/mesh/BoxMesh.html
    mesh = fe.BoxMesh(fe.Point(0., 0., 0.), fe.Point(domain_x, domain_y, domain_z), 
                      round(domain_x/ele_size), round(domain_y/ele_size), round(domain_z/ele_size))

    # Save mesh to local file, optional, just for inspection
    mesh_file = fe.File(f'data/vtk/fem/mesh.pvd')
    mesh_file << mesh

    # Define bottom surface 
    class Bottom(fe.SubDomain):
        def inside(self, x, on_boundary):
            # The condition for a point x to be on bottom side is that x[2] < EPS
            return on_boundary and x[2] < EPS

    # Define top surface
    class Top(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[2] > domain_z - EPS

    # Define the other four surfaces
    class SurroundingSurfaces(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and (x[0] < EPS or x[0] > domain_x - EPS or x[1] < EPS or x[1] > domain_y - EPS)

    # The following few lines mark different boundaries with different numbers
    # For example, the top surface is marked with the integer number 2
    bottom = Bottom()
    top = Top()
    surrounding_surfaces = SurroundingSurfaces()
    boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    bottom.mark(boundaries, 1)
    top.mark(boundaries, 2)
    surrounding_surfaces.mark(boundaries, 3)
    ds = fe.Measure('ds')(subdomain_data=boundaries)

    # Define FEM function space to be first order continuous Galerkin (the most commonly used)
    V = fe.FunctionSpace(mesh, 'CG', 2)

    # u_crt is the temperature field we want to solve 
    # u_crt = fe.Function(V)
    u_crt = fe.interpolate(fe.Constant(ambient_T), V)

    # u_pre is the temperature from the previous step
    # We initialize u_pre to be a constant field = ambient_T (assign initial values)
    u_pre = fe.interpolate(fe.Constant(ambient_T), V)

    # v is the test function in FEM
    v = fe.TestFunction(V)

    # If theta = 0., we recover implicit Eulear; if theta = 1., we recover explicit Euler; theta = 0.5 seems to be a good choice.
    theta = 1.
    u_rhs = theta*u_pre + (1 - theta)*u_crt

    # Define Dirichlet boundary conditions for the bottom surface to be always at ambient temperature
    bcs = [fe.DirichletBC(V, fe.Constant(ambient_T), bottom)]

    # Define the laser heat source, note that t is a changeble parameter
    class LaserExpression(fe.UserExpression):
        def __init__(self, t):
            # Construction method of base class has to be called first
            super(LaserExpression, self).__init__()
            self.t = t

        def eval(self, values, x):
            t = self.t
            values[0] = 2*P*eta/(np.pi*r**2) * np.exp(-2*((x[0] - x0 - vel*t)**2 + (x[1] - y0)**2) / r**2)
    
        def value_shape(self):
            return ()

    q_laser = LaserExpression(None)
    q_convection = h * (u_rhs - ambient_T)

    # For the top surface, we will consider both convection and laser heating
    q_top = q_convection + q_laser 
    # For the four side surfaces, we will only consider convection
    q_surr = q_convection

    # Deine the weak form residual
    # For the terms with fe.dx, they are volume integrals
    # Note that ds(2) means that it is a surface integral only computed on surface number 2 (the top surface), which we defined previously!
    residual = rho*Cp/dt*(u_crt - u_pre) * v * fe.dx + k * fe.dot(fe.grad(u_rhs), fe.grad(v)) * fe.dx \
                - q_top * v * ds(2) - q_surr * v * ds(3)

    # Open a pvd file to store results 
    u_vtk_file = fe.File(f'data/vtk/fem/u.pvd')

    # Store solution at the 0th step
    u_vtk_file << u_pre
 
    # for i in range(len(ts) - 1):

    for i in range(101):

        print(f"step {i + 1}, time = {ts[i + 1]}")

        # Update the time parameter in laser
        q_laser.t = theta*ts[i] + (1 - theta)*ts[i + 1]

        # Solve the problem at this time step
        solver_parameters = {'newton_solver': {'maximum_iterations': 20, 'linear_solver': 'mumps'}}
        fe.solve(residual == 0, u_crt, bcs, solver_parameters=solver_parameters)

        # After solving, update u_pre so that it is equal to the newly solved u_crt
        u_pre.assign(u_crt)

        # Store solution at this step
        u_vtk_file << u_pre

        print(f"min T = {np.min(np.array(u_pre.vector()))}")
        print(f"max T = {np.max(np.array(u_pre.vector()))}")


if __name__ == '__main__':
    simulation()
