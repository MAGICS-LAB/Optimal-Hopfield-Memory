import torch
import numpy as np
import matplotlib.pyplot as plt

import torch.nn.functional as F

from utils import energy, entmax, normmax_bisect
from feature_map import *


Num_points = 2


def energy2(Q: torch.tensor, X: torch.tensor, beta: float):

    # Q: (B, D)
    # X: (M, D)
    term1 = 0.5 * (Q * Q).sum(-1)
    term2 = -1 * torch.log(torch.exp(beta * Q @ X.T).sum(-1))
    return term1 + term2


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

    normalize = True
    n_samples = 40

    Iter = 5
    which = [1, 2, Iter]
    nplots = len(which) + 1

    fig, axes = plt.subplots(nplots, 3, figsize=(10, 10), constrained_layout=True)
    if N == 2:
        temp = 0.9
        torch.random.manual_seed(111)
        fig.suptitle("2 Points", fontsize=16)

    else:
        temp = 0.05
        torch.random.manual_seed(1111)
        fig.suptitle("4 Points", fontsize=16)

    X, query = generate_data()
    if normalize:
        X = X / torch.sqrt(torch.sum(X * X, dim=1)).unsqueeze(1)

    # queries[0].zero_()

    # initialized energy landscape
    xmin, xmax = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    ymin, ymax = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

    xx = np.linspace(xmin, xmax, n_samples)
    yy = np.linspace(ymin, ymax, n_samples)
    mesh_x, mesh_y = np.meshgrid(xx, yy)
    Q = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])
    Q = torch.from_numpy(Q).float()
    cmap = "hot"

    E1 = energy2(Q, X, beta=1 / temp).reshape(*mesh_x.shape)
    axes[0, 0].contourf(mesh_x, mesh_y, E1, cmap=cmap)
    E15 = energy(Q, X, alpha=1.5, beta=1 / temp).reshape(*mesh_x.shape)
    axes[0, 1].contourf(mesh_x, mesh_y, E15, cmap=cmap)
    E2 = energy(Q, X, alpha=2, beta=1 / temp).reshape(*mesh_x.shape)
    axes[0, 2].contourf(mesh_x, mesh_y, E2, cmap=cmap)

    for ax in axes[0]:

        ax.plot(
            X[:, 0],
            X[:, 1],
            "s",
            markerfacecolor="w",
            markeredgecolor="k",
            markeredgewidth=1,
            markersize=5,
            label="$ξ_μ$",
        )
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(())
        ax.set_yticks(())

    axes[0, 0].set_ylabel("N = 0", fontsize=14)

    w = FeatureMap(D, D)
    opt = torch.optim.SGD(w.parameters(), 0.1, momentum=0.0)

    count = 0

    # for i in range(nplots-1):
    for i, n in enumerate(range(Iter)):

        opt.zero_grad()

        phi_x = w(X)
        loss = separation_loss(phi_x, tau=1.0)
        loss.backward()
        opt.step()

        if n + 1 in which:

            count += 1
            with torch.no_grad():
                # wQ = F.normalize(w(Q.cuda()).cpu(), dim=-1)
                wX = F.normalize(w(X).cpu(), dim=-1)

            xmin, xmax = wX[:, 0].min() - 0.1, wX[:, 0].max() + 0.1
            ymin, ymax = wX[:, 1].min() - 0.1, wX[:, 1].max() + 0.1

            xx = np.linspace(xmin, xmax, n_samples)
            yy = np.linspace(ymin, ymax, n_samples)
            mesh_x, mesh_y = np.meshgrid(xx, yy)
            Q = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])
            Q = torch.from_numpy(Q).float()
            wQ = Q

            E1 = energy2(Q, wX, beta=1 / temp).reshape(*mesh_x.shape)
            axes[count, 0].contourf(mesh_x, mesh_y, E1, cmap=cmap)
            E15 = energy(Q, wX, alpha=1.5, beta=1 / temp).reshape(*mesh_x.shape)
            axes[count, 1].contourf(mesh_x, mesh_y, E15, cmap=cmap)
            E2 = energy(Q, wX, alpha=2, beta=1 / temp).reshape(*mesh_x.shape)
            axes[count, 2].contourf(mesh_x, mesh_y, E2, cmap=cmap)
            axes[count, 0].plot(
                wX[:, 0],
                wX[:, 1],
                "s",
                markerfacecolor="w",
                markeredgecolor="k",
                markeredgewidth=1,
                markersize=5,
                label="$ξ_μ$",
            )
            axes[count, 1].plot(
                wX[:, 0],
                wX[:, 1],
                "s",
                markerfacecolor="w",
                markeredgecolor="k",
                markeredgewidth=1,
                markersize=5,
                label="$ξ_μ$",
            )
            axes[count, 2].plot(
                wX[:, 0],
                wX[:, 1],
                "s",
                markerfacecolor="w",
                markeredgecolor="k",
                markeredgewidth=1,
                markersize=5,
                label="$ξ_μ$",
            )

            for ax in axes[count]:

                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                ax.set_xticks(())
                ax.set_yticks(())
            axes[count, 0].set_ylabel(f"N = {n+1}", fontsize=14)

    axes[0, 0].set_title("Softmax", fontsize=14)
    axes[0, 1].set_title("$1.5$-entmax", fontsize=14)
    axes[0, 2].set_title("Sparsemax", fontsize=14)
    axes[0, 0].legend()

    plt.tight_layout()
    plt.savefig(f"{Num_points}points.png", dpi=800)
    plt.show()


if __name__ == "__main__":
    main()
