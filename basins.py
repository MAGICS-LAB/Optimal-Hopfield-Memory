import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from utils import energy, entmax, normmax_bisect

from feature_map import *

Num_points = 5

def generate_data():

    D = 2
    N = Num_points

    patterns = []
    queries = []

    for _ in range(10):
        patterns.append(torch.randn(N, D, dtype=torch.float32))
        queries.append(torch.randn(D, 1, dtype=torch.float32))

    return patterns[0], queries[0]

def main():

    D = 2
    N = Num_points

    temp = .05
    normalize = True
    n_samples = 100
    thresh = 0.001

    torch.random.manual_seed(42)

    Iter = 5
    which = [1, 2, Iter]
    nplots = len(which)+1

    fig, axes = plt.subplots(nplots, 3, figsize=(10, 6),
                             constrained_layout=True)

    X, query = generate_data()

    if normalize:
        print(torch.sqrt(torch.sum(X*X, dim=1)))
        X = X / torch.sqrt(torch.sum(X*X, dim=1)).unsqueeze(1)

        xmin, xmax = X[:, 0].min(), X[:, 0].max()
        ymin, ymax = X[:, 1].min(), X[:, 1].max()

        xmin -= .1
        ymin -= .1
        xmax += .1
        ymax += .1

        xx = np.linspace(xmin, xmax, n_samples)
        yy = np.linspace(ymin, ymax, n_samples)

        mesh_x, mesh_y = np.meshgrid(xx, yy)

        Q = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])
        Q = torch.from_numpy(Q).float()

        for k, alpha in enumerate([1, 1.5, 2]):
            num_iters = 5

            # X is n by d. Xi is m by d.

            Xi = Q
            for _ in range(num_iters):
                p = entmax(Xi @ X.T / temp, alpha=alpha, dim=-1)
                Xi = p @ X

            dists = torch.cdist(Xi, X)

            response = torch.zeros_like(dists[:, 0])
            for pp in range(len(X)):
                response[dists[:, pp] < thresh] = pp+1

            cols = ['w', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5'][:len(X)+1]
            cmap = matplotlib.colors.ListedColormap(cols)

            for pp in range(len(X)):
                response = response.reshape(*mesh_x.shape)
                axes[0,k].pcolormesh(mesh_x, mesh_y, response,
                                     vmin=0, vmax=len(X)+1,
                                     cmap=cmap)

        for ax in axes[0]:
            for pp in range(len(X)):
                ax.plot(X[pp, 0], X[pp, 1],
                        's',
                        markerfacecolor=f'C{pp}',
                        markeredgecolor='k',
                        markeredgewidth=1,
                        markersize=5)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_xticks(())
            ax.set_yticks(())

        axes[0, 0].set_ylabel(f"N = 0", fontsize=14)

    w = FeatureMap(D, D)
    opt = torch.optim.Adam(w.parameters(), 0.1) # , momentum=0.0)
    count = 0

    for i, n in enumerate(range(Iter)):

        opt.zero_grad()

        phi_x = w(X)
        loss = separation_loss(phi_x, tau=1)
        loss.backward()
        opt.step()

        if n+1 in which:
            count += 1
            with torch.no_grad():
                wX = F.normalize(w(X).cpu(), dim=-1)

            xmin, xmax = wX[:, 0].min()-0.1, wX[:, 0].max()+0.1
            ymin, ymax = wX[:, 1].min()-0.1, wX[:, 1].max()+0.1

            xx = np.linspace(xmin, xmax, n_samples)
            yy = np.linspace(ymin, ymax, n_samples)

            mesh_x, mesh_y = np.meshgrid(xx, yy)

            Q = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])
            Q = torch.from_numpy(Q).float()
            wQ = Q

            for k, alpha in enumerate([1, 1.5, 2]):
                num_iters = 5

                Xi = Q
                for _ in range(num_iters):
                    p = entmax(Xi @ wX.T / temp, alpha=alpha, dim=-1)
                    Xi = p @ wX

                dists = torch.cdist(Xi, wX)

                response = torch.zeros_like(dists[:, 0])
                for pp in range(len(wX)):
                    response[dists[:, pp] < thresh] = pp+1

                cols = ['w', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5'][:len(X)+1]
                cmap = matplotlib.colors.ListedColormap(cols)

                for pp in range(len(X)):
                    response = response.reshape(*mesh_x.shape)
                    axes[count,k].pcolormesh(mesh_x, mesh_y, response,
                                        vmin=0, vmax=len(X)+1,
                                        cmap=cmap)

            for ax in axes[count]:
                for pp in range(len(wX)):
                    ax.plot(wX[pp, 0], wX[pp, 1],
                            's',
                            markerfacecolor=f'C{pp}',
                            markeredgecolor='k',
                            markeredgewidth=1,
                            markersize=5)
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                ax.set_xticks(())
                ax.set_yticks(())
            axes[count, 0].set_ylabel(f"N = {n+1}", fontsize=14)

    axes[0,0].set_title("$1$-entmax")
    axes[0,1].set_title("$1.5$-entmax")
    axes[0,2].set_title("$2$-entmax")
    plt.show()
    plt.savefig("basins.png", dpi=600)


if __name__ == '__main__':
    main()



