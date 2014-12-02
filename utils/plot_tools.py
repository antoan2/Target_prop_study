from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d, Axes3D
import numpy as np


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def plot_curves(x, fig_title, plot_titles, logs=None):

    fig = plt.figure(fig_title)
    
    n_plots = x.shape[1]
    
    for i in xrange(n_plots):
        ax = fig.add_subplot(n_plots, 1, i+1)
        ax.set_title(plot_titles[i])
        ax.set_xlabel('batch_seen')
        if not logs == None:
            if logs[i]:
                ax.semilogy(x[:, i])
            else:
                ax.plot(x[:, i])
        else:
            ax.plot(x[:, i])

def plot_scatter_labels(x, y, fig_title):

    fig = plt.figure(fig_title)

    c = ['r', 'b']
    m = ['o', '^']

    if x.shape[1] == 2:
        ax = fig.add_subplot(111)
        for i in xrange(2):
            ax.scatter(x[y==i, 0], x[y==i, 1], c=c[i], marker=m[i])
    else:
        ax = fig.add_subplot(111, projection='3d')
        for i in xrange(2):
            x_c = x[y==i, ...]
            ax.scatter3D(x_c[:, 0], x_c[:, 1], x_c[:, 2], c=c[i], marker=m[i])

def plot_scatter(x, fig_title):

    fig = plt.figure(fig_title)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(x[:, 0], x[:, 1], x[:, 2])

def plot_scatter_targets(h, hh, fig_title, norm_target=False):

    fig = plt.figure(fig_title)

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(hh[:, 0], hh[:, 1], hh[:, 2], edgecolor='r', marker='*', alpha=1.)
    ax.scatter3D(h[:, 0], h[:, 1], h[:, 2], edgecolor=None, c='b', marker='+', alpha=1.)
    
    if norm_target:
        delta = hh-h
        norms = np.linalg.norm(hh-h, axis=1).mean()
        norms = norms[..., np.newaxis]
        hhh = h + 1.*delta/norms
        ax.scatter3D(hhh[:, 0], hhh[:, 1], hhh[:, 2], edgecolor=None, marker='*', c='g', alpha=1.)

    for i in xrange(h.shape[0]):
        if norm_target:
            a = Arrow3D([h[i, 0], hhh[i, 0]], [h[i, 1], hhh[i, 1]], [h[i, 2], hhh[i, 2]], mutation_scale=4, lw=1, arrowstyle='-|>', color='b')
        else:
            a = Arrow3D([h[i, 0], hh[i, 0]], [h[i, 1], hh[i, 1]], [h[i, 2], hh[i, 2]], mutation_scale=4, lw=1, arrowstyle='-|>', color='b')
        ax.add_artist(a)
