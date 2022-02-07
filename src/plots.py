import numpy as np
import random
import shutil
import os
from matplotlib import collections
from matplotlib import patches
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from src.arguments import args
from src.utils import unpack_state


def kp_loop(kp_pairs):
    loop = list(kp_pairs[0])
    kp_arr = np.array(kp_pairs[1:])
    while kp_arr.shape[0] > 0:
        kp_find = loop[-1]
        has_kp = np.any(kp_arr == kp_find, axis=1)
        row = kp_arr[has_kp]

        loop.append(row[row != kp_find][0])
        kp_arr = kp_arr[~has_kp]
    assert loop[0] == loop[-1]
    return loop[:-1]


def poly_plot(polygon_mesh, variable, **kwargs):
    """Plot the mesh.
    This function plots the polygon mesh.
    In 2D, this creates a class:`matplotlib.collections.PolyCollection`
    and adds it to the current axes.
    In 3D, it creates a
    :class:`mpl_toolkits.mplot3d.art3d.Poly3DCollection` and
    adds it to the current axes.
    The keyword arguments are passed though to matplotlib.
    Args:
        index_by (str): *(optional)* {'facet' | 'material' | 'seed'}
            Flag for indexing into the other arrays passed into the
            function. For example,
            ``plot(index_by='material', color=['blue', 'red'])`` will plot
            the regions with ``phase_number`` equal to 0 in blue, and
            regions with ``phase_number`` equal to 1 in red. The facet
            option is only available for 3D plots. Defaults to 'seed'.
        material (list): *(optional)* Names of material phases. One entry
            per material phase (the ``index_by`` argument is ignored).
            If this argument is set, a legend is added to the plot with
            one entry per material.
        loc (int or str): *(optional)* The location of the legend,
            if 'material' is specified. This argument is passed directly
            through to :func:`matplotlib.pyplot.legend`. Defaults to 0,
            which is 'best' in matplotlib.
        **kwargs: Keyword arguments for matplotlib.
    """
    n_dim = len(polygon_mesh.points[0])
    if n_dim == 2:
        ax = plt.gca()
    else:
        ax = plt.gcf().gca(projection=Axes3D.name)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    if n_dim == 2:
        # create vertex loops for each poly
        vloops = [kp_loop([polygon_mesh.facets[f] for f in r]) for r in
                  polygon_mesh.regions]
        # create poly input
        xy = [np.array([polygon_mesh.points[kp] for kp in lp]) for lp in vloops]

        pc = collections.PolyCollection(xy, **kwargs)
        image = ax.add_collection(pc)
        ax.autoscale_view()
    elif n_dim == 3:
        if n_obj > 0:
            zlim = ax.get_zlim()
        else:
            zlim = [float('inf'), -float('inf')]
        polygon_mesh.plot_facets(polygon_mesh, **kwargs)

    else:
        raise NotImplementedError('Cannot plot in ' + str(n_dim) + 'D.')

    if variable == 'temp':
        # colors = 100 * np.random.rand(len(polygon_mesh.regions))
        pc.set_array(polygon_mesh.temp)
        plt.colorbar(pc, shrink=0.5, ax=ax)
        image.set_clim(vmin=273, vmax=1000)

    # Adjust Axes
    mins = np.array(polygon_mesh.points).min(axis=0)
    maxs = np.array(polygon_mesh.points).max(axis=0)
    xlim = (min(xlim[0], mins[0]), max(xlim[1], maxs[0]))
    ylim = (min(ylim[0], mins[1]), max(ylim[1], maxs[1]))
    if n_dim == 2:
        plt.axis('square')
        plt.xlim(xlim)
        plt.ylim(ylim)
    elif n_dim == 3:
        zlim = (min(zlim[0], mins[2]), max(zlim[1], maxs[2]))
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)



def plot_facets(polygon_mesh, hide_interior=True, **kwargs):
    """Plot PolyMesh facets.
    This function plots the facets of the polygon mesh, rather than the
    regions.
    In 2D, it adds a :class:`matplotlib.collections.LineCollection` to the
    current axes.
    In 3D, it adds a
    :class:`mpl_toolkits.mplot3d.art3d.Poly3DCollection`
    with ``facecolors='none'``.
    The keyword arguments are passed though to matplotlib.
    Args:
        index_by (str): *(optional)* {'facet' | 'material' | 'seed'}
            Flag for indexing into the other arrays passed into the
            function. For example,
            ``plot(index_by='material', color=['blue', 'red'])`` will plot
            the regions with ``phase_number`` equal to 0 in blue, and
            regions with ``phase`` equal to 1 in red. The facet option is
            only available for 3D plots. Defaults to 'seed'.
        hide_interior (bool): If True, removes interior facets from the
            output plot. This avoids occasional matplotlib issue where
            interior facets are shown in output plots.
        **kwargs (dict): Keyword arguments for matplotlib.
    """
    n_dim = len(polygon_mesh.points[0])
    if n_dim == 2:
        ax = plt.gca()
    else:
        ax = plt.gcf().gca(projection=Axes3D.name)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    if n_dim == 2:
        xy = [np.array([polygon_mesh.points[kp] for kp in f]) for f in polygon_mesh.facets]

        pc = collections.LineCollection(xy, kwargs)
        ax.add_collection(pc)
        ax.autoscale_view()
    else:

        if ax.has_data:
            zlim = ax.get_zlim()
        else:
            zlim = [float('inf'), -float('inf')]

        if hide_interior:
            f_mask = [min(fn) < 0 for fn in polygon_mesh.facet_neighbors]
            xy = [np.array([polygon_mesh.points[kp] for kp in f]) for m, f in
                  zip(f_mask, polygon_mesh.facets) if m]
            list_kws = [k for k, vl in f_kwargs.items()
                        if isinstance(vl, list)]
            plt_kwargs = {k: vl for k, vl in f_kwargs.items() if
                          k not in list_kws}
            for k in list_kws:
                v = [val for val, m in zip(f_kwargs[k], f_mask) if m]
                plt_kwargs[k] = v
        else:
            xy = [np.array([polygon_mesh.points[kp] for kp in f]) for f in
                  polygon_mesh.facets]
            plt_kwargs = f_kwargs
        pc = Poly3DCollection(xy, **plt_kwargs)
        ax.add_collection(pc)

    # Adjust Axes
    mins = np.array(polygon_mesh.points).min(axis=0)
    maxs = np.array(polygon_mesh.points).max(axis=0)

    xlim = (min(xlim[0], mins[0]), max(xlim[1], maxs[0]))
    ylim = (min(ylim[0], mins[1]), max(ylim[1], maxs[1]))
    if n_dim == 2:
        plt.axis('square')
        plt.xlim(xlim)
        plt.ylim(ylim)
    if n_dim == 3:
        zlim = (min(zlim[0], mins[2]), max(zlim[1], maxs[2]))
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)


def plot_polygon_mesh(orientation_colors, polygon_mesh, variable):
    region_colors = [orientation_colors[o] for o in polygon_mesh.orientations]

    if variable == 'phase':
        poly_plot(polygon_mesh, variable, facecolor=region_colors, edgecolor='k')
    else:
        poly_plot(polygon_mesh, variable, edgecolor='k')

    plt.plot(polygon_mesh.centroids[:, 0], polygon_mesh.centroids[:, 1], 'o', color='black', markersize=0.5)
    plt.axis('image')
    plt.axis([-2.6, 2.6, -0.6, 0.6])
    # plt.show()
    # exit()


def get_orientations(zeta, eta):
    orientations = np.argmax(eta, axis=1)
    orientations = np.where(zeta.reshape(-1) < 0.1, args.num_orientations, orientations)
    return orientations


def save_animation(ys, polygon_mesh):
    orientation_colors = [f"#{random.randint(0, 0xFFFFFF):06x}" for _ in range(polygon_mesh.num_orientations)]
    orientation_colors.append('#000000')

    Ts, zetas, etas = unpack_state(ys)

    tmp_root_path = f'data/png/tmp/'
    shutil.rmtree(tmp_root_path, ignore_errors=True)
    os.mkdir(tmp_root_path)

    max_temp = []

    for i in range(len(Ts)):
        if i % 20 == 0:
            print(f"i = {i}")
        fig = plt.figure(figsize=(20, 6))
        T = Ts[i]
        zeta = zetas[i]
        eta = etas[i]

        polygon_mesh.temp = T.reshape(-1)
        polygon_mesh.orientations = get_orientations(zeta, eta)

        plot_polygon_mesh(orientation_colors, polygon_mesh, variable='temp')
        # plot_polygon_mesh(orientation_colors, polygon_mesh, variable='phase')

        fig.savefig(tmp_root_path + f'{i:05}.png', bbox_inches='tight')
        plt.close(fig)

        max_temp.append(np.max(T))

    fig = plt.figure()
    plt.plot(max_temp, marker='o', markersize=2, linestyle="-", linewidth=1, color='black')

    os.system('ffmpeg -y -framerate 25 -i data/png/tmp/%05d.png -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" ' + 'data/mp4/temp.mp4')

